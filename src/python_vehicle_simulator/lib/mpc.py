from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from typing import List, Any, Tuple, Dict, Literal
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from python_vehicle_simulator.utils.math_fn import R_casadi
from python_vehicle_simulator.lib.physics import RHO, GRAVITY
import numpy as np, casadi as ca, matplotlib.pyplot as plt
from python_vehicle_simulator.lib.path import PWLPath
from math import pi
from matplotlib.axes import Axes


class MPCPathTrackingRevolt(IControl):
    """
    Model-Predictive Controller for path tracking of the ReVolt ASV, based on 

    Reinforcement learning-based NMPC for tracking control of ASVs:Theory and experiments

    and 

    MPC-based Reinforcement Learning for a Simplified Freight Mission of Autonomous Surface Vehicles


    STATES:
    eta = [n, e, psi]
    nu = [u, v, r]

    INPUTS:
    u   = [alphas, forces]
        = [
            a1, a2, a3,
            f1, f2, f3
        ]
    
    """

    AlphaActualOpt:ca.DM = None
    ForceActualOpt:ca.DM = None
    UActualOpt:ca.DM = None
    AlphaSetpointOpt:ca.DM = None
    ForceSetpointOpt:ca.DM = None
    USetpointOpt:ca.DM = None
    XOpt:ca.DM = None

    # Decision variables
    ## States
    X:ca.SX = None

    ## Command inputs
    Alpha:ca.SX = None # Azimuth angles
    Force:ca.SX = None # Thruster forces

    def __init__(
            self,
            vessel_params:Any,
            actuator_params:Any,
            *args,
            horizon:int=20,
            dt:float=0.2,
            tau_ext:Tuple=(0., 0., 0.),
            gamma:float=1.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vessel_params = vessel_params
        self.actuator_params = actuator_params
        self.horizon = horizon
        self.dt = dt

        self.Nu = 6 # 3 forces + 3 azimuth angles
        self.Nx = 6 # north, east, down + derivatives

        # State constraints
        self.lbx = self.vessel_params.lbx
        self.ubx = self.vessel_params.ubx

        # External disturbances
        self.tau_ext = np.array(tau_ext)

        ## Actuators
        self.lx = self.actuator_params.xy[:, 0]
        self.ly = self.actuator_params.xy[:, 1]
        self.Ti = self.actuator_params.Ti
        self.T = self.actuator_params.T
        self.delta_prev = None
        # self.efficiency = np.array([1.0, 1.0, 1.0]) # for FTC

        # Force constraints
        self.lbf = self.actuator_params.f_min
        self.ubf = self.actuator_params.f_max

        # Angle constraints
        self.lba = self.actuator_params.lba
        self.uba = self.actuator_params.uba

        # Complete command constraints
        self.lbu = np.concatenate([self.lbf, self.lba])
        self.ubu = np.concatenate([self.ubf, self.uba])

        # Time constant
        self.actuators_time_constant = self.actuator_params.time_constant
        self.actuators_dt = np.array([min(time_constant, self.dt) for time_constant in self.actuator_params.time_constant])

        # Hyperparameters
        self.huber_penalty_slope = 50 # 10 # delta
        self.huber_penalty_weight = 30 # q_x,y
        self.heading_penalty_weight = 100 # 50 # q_psi
        self.singular_value_penalty = 1e-3 # epsilon -> for nonsigular thruster configuration
        self.singular_value_weight = 1e-5 # 1e-6 # 1e-5 # 8e-4 # 1e-9 #  5e-4 # 1e-5 # rho -> for nonsigular thruster configuration

        self.Q = np.array([
            [1, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ]) * 1e0 # Velocity weight matrix

        self.Ra = np.eye(3) * 1e-2 # Azimuth weight matrix
        self.Rf = np.eye(3) * 1e-5 # 1e-1 # Force weight matrix

        self.Theta_v = 0*np.diag([10, 10, 100, 1, 10, 10])*(gamma**self.horizon) # Terminal weight matrix

        self.gamma = gamma
        
        self.init_nlp()

    def actuators_dynamics(self, uk:ca.SX, efficiency:ca.SX, *args, **kwargs) -> ca.SX:
        """
        Compute the total generalized force based on thruster's forces and azimuth angles
        """

        # This should be in the actuator configuration, as a single matrix
        tau_a = ca.SX([0., 0., 0.])
        for i in range(3): 
            tau_a += self.Ti(uk[i+3], self.lx[i], self.ly[i]) * uk[i] * efficiency[i]
        return tau_a

    def dynamics(self, xk:ca.SX, uk:ca.SX, efficiency:ca.SX, *args, **kwargs) -> ca.SX:
        """
        xk : etak, nuk = n, e, psi, u, v, r
        uk : fk, alphak = f1, f2, f3, a1, a2, a3

        Input commands are the ACTUAL VALUES ***NOT*** SETPOINTS
        """
        # Actuators
        tau_a = self.actuators_dynamics(uk, efficiency, *args, **kwargs)
        C = self.vessel_params.CRB(xk[3:6]) + self.vessel_params.CA(xk[3:6])
        D = self.vessel_params.D

        # Vessel dynamics
        nu_dot = self.vessel_params.Minv @ (
            tau_a +
            self.tau_ext -
            C @ xk[3:6] -
            D @ xk[3:6]
        )
        # nu_dot = self.vessel_params.dynamics(xk, tau_a-self.tau_ext)

        # Euler integration
        nukp1 = xk[3:6] + nu_dot * self.dt
        eta_dot = ca.mtimes(R_casadi(xk[2]), xk[3:6])
        etakp1 = xk[0:3] + eta_dot * self.dt
        
        # Return next states
        return ca.vertcat(etakp1, nukp1)
    
    def __lagrange__(self, xk:ca.SX, uk:ca.SX, pk_des:Tuple[float, float, float], nu_des:np.ndarray, *args, sk:ca.SX=None, k:int=None, **kwargs) -> ca.SX:
        """
        k can be used if l(xk, uk) changes at each stage -> e.g. path tracking, discount factor, etc..

        from 
        
        Reinforcement learning-based NMPC for tracking control of ASVs:Theory and experiments (https://www.sciencedirect.com/science/article/pii/S0967066121002823)
        """
        return (self.gamma**k)*(self.huber_penalty_weight * self.huber_cost(xk, pk_des) + self.heading_penalty_weight * self.heading_cost(xk, pk_des) + self.singular_value_weight * self.singularity_cost(uk) + self.speed_cost(xk, nu_des) + self.alpha_cost(uk) + self.force_cost(uk))
        # return ca.mtimes(ca.transpose(xk[0:3]-np.array(pk_des)), xk[0:3]-np.array(pk_des))

    def __mayer__(self, xf:ca.SX, pf_des:Tuple[float, float, float], nu_des:np.ndarray, *args, sf:ca.SX=None, **kwargs) -> ca.SX:
        xd = np.concatenate([pf_des, nu_des])
        return (xf-xd).T @ self.Theta_v @ (xf-xd) #1 * self.__lagrange__(xf, np.zeros((self.Nu,)), pf_des, nu_des, *args, sk=sf, **kwargs)

    def __format_commands__(self, commands:np.ndarray, *args, **kwargs) -> List[np.ndarray]:
        return [np.array([commands[i+3], np.sqrt(commands[i]/self.actuator_params.k_pos[i]) if commands[i] >= 0 else -np.sqrt(-commands[i]/self.actuator_params.k_neg[i])]) for i in range(3)]

    def huber_cost(self, xk:Any, pk_des:Tuple[float, float, float]) -> Any:
        return self.huber_penalty_slope**2 * (ca.sqrt(1 + ((xk[0]-pk_des[0])**2 + (xk[1]-pk_des[1])**2) / self.huber_penalty_slope **2) - 1 ) 
    
    def heading_cost(self, xk:Any, pk_des:Tuple[float, float, float]) -> Any:
        return 0.5 * (1 - ca.cos(xk[2] - pk_des[2])) # It's not this weight that makes it crazy
    
    def singularity_cost(self, uk:Any) -> Any:
        T = self.actuator_params.T(uk[3], uk[4], uk[5])
        return 1 / (self.singular_value_penalty + ca.det(T @ T.T))

    def speed_cost(self, xk:Any, nu_des:np.ndarray) -> Any:
        return (xk[3:6]-nu_des).T @ self.Q @ (xk[3:6]-nu_des)
    
    def alpha_cost(self, uk:Any) -> Any:
        return uk[3:6].T @ self.Ra @ uk[3:6]
    
    def force_cost(self, uk:Any) -> Any:
        return uk[0:3].T @ self.Rf @ uk[0:3]
    
    def eval_current_state(self, p0_des:np.ndarray, nu_des:np.ndarray) -> Dict:
        x0, u0 = self.get_x_opt(0), self.get_u_actual_opt(0)
        return {
            'huber': self.huber_penalty_weight*self.huber_cost(x0, p0_des),
            'heading': self.heading_penalty_weight*self.heading_cost(x0, p0_des),
            'singularity': self.singular_value_weight*self.singularity_cost(u0),
            'speed': self.speed_cost(x0, nu_des),
            'alpha': self.alpha_cost(u0),
            'force': self.force_cost(u0)
        }

    def eval_solution(self, nu_des:np.ndarray, path:List[Tuple[float, float, float]]) -> Dict:
        """
        Evaluate the current solution by computing the total cost of each term
        """

        huber, heading, singularity, speed, alpha, force = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for k in range(self.horizon):
            # print('\n################## k: ', k)
            uk = self.get_u_actual_opt(k)
            xk = self.get_x_opt(k)
            pk_des = path[k]
            # print("SHAPE: ", uk.shape, xk.shape)
            
            huber += (self.gamma**k)*self.huber_cost(xk, pk_des)
            # print('huber: ', huber)
            
            heading += (self.gamma**k)*self.heading_cost(xk, pk_des)
            # print('heading: ', heading)

            singularity += (self.gamma**k)*self.singularity_cost(uk)
            # print('singularity: ', singularity)

            speed += (self.gamma**k)*self.speed_cost(xk, nu_des)
            # print('speed: ', speed)

            alpha += (self.gamma**k)*self.alpha_cost(uk)
            # print('alpha: ', alpha)

            force += (self.gamma**k)*self.force_cost(uk)
            # print('force: ', force)

        cost = {
            'huber': self.huber_penalty_weight*huber,
            'heading': self.heading_penalty_weight*heading,
            'singularity': self.singular_value_weight*singularity,
            'speed': speed,
            'alpha': alpha,
            'force': force
        }

        return cost       

    def set_dynamic_constraints(self, *args, **kwargs) -> None:
        """
        
        """
        ## Vessel
        self.dynamic_constraints = ca.SX.nan(self.Nx) # self.get_x(0) # Reserve space for constraint x(0) = 0 to be set at runtime
        self.dynamic_actuator_constraints = ca.SX.nan(self.Nu) #self.get_u(0) # Reserve space for initial (delta) input constraint
        self.efficiency_constraints = ca.SX.nan(3)
        for k in range(0, self.horizon):
            self.dynamic_constraints = ca.vertcat(self.dynamic_constraints, self.get_x(k+1) - self.dynamics(self.get_x(k), self.get_u_actual(k), self.efficiency, *args, **kwargs))
        
        # Consider time constant for actuator dynamics
        for k in range(0, self.horizon-1):
            self.dynamic_actuator_constraints = ca.vertcat(self.dynamic_actuator_constraints, self.get_u_actual(k+1) - self.get_u_actual(k) - self.actuators_dt * (self.get_u_setpoint(k+1) - self.get_u_actual(k)) / self.actuators_time_constant)

        self.LBDynamics = np.array([self.Nx*[0.0]]).repeat(self.horizon + 1, axis=0).flatten().tolist() # Lower Bound Dynamics, including 0 <= x(0) - x0 <= 0
        self.UBDynamics = np.array([self.Nx*[0.0]]).repeat(self.horizon + 1, axis=0).flatten().tolist() # Upper Bound Dynamics, including 0 <= x(0) - x0 <= 0
        self.LBActuatorsDynamics = np.array([self.Nu*[0.0]]).repeat(self.horizon, axis=0).flatten().tolist()
        self.UBActuatorsDynamics = np.array([self.Nu*[0.0]]).repeat(self.horizon, axis=0).flatten().tolist()
        self.LBActuatorEfficiencyConstraints = [0, 0, 0]
        self.UBActuatorEfficiencyConstraints = [0, 0, 0]


    def set_states_and_commands_constraints(self, *args, **kwargs) -> None:
        # States
        self.LBX = self.lbx[None, :].repeat(self.horizon + 1, axis=0).flatten().tolist()
        self.UBX = self.ubx[None, :].repeat(self.horizon + 1, axis=0).flatten().tolist()
        # Actual forces & angles
        self.LBUA = self.lbu[None, :].repeat(self.horizon, axis=0).flatten().tolist()
        self.UBUA = self.ubu[None, :].repeat(self.horizon, axis=0).flatten().tolist()
        # Efficiency
        self.LBEfficiency = [0, 0, 0]
        self.UBEfficiency = [1, 1, 1]


    def init_nlp(self, *args, **kwargs) -> None:
        """
        
        """
        # Decision Variables
        self.X = ca.SX.sym("X", self.Nx * (self.horizon + 1))   # States
        self.UActual = ca.SX.sym("UActual", self.Nu * self.horizon)   # Actual command input
        self.USetpoint = ca.SX.sym("USetpoint", self.Nu * self.horizon) # Setpoint command input
        self.efficiency = ca.SX.sym("efficiency", 3)

        # Constraints
        self.set_states_and_commands_constraints(*args, **kwargs)   # x_k \in X, u_k \in U
        self.set_dynamic_constraints(*args, **kwargs)               # x_k+1 = f(x_k, u_k) || Time constant for actuators


    def set_cost(self, path:List[Tuple[float, float, float]], nu_des:np.ndarray, *args, **kwargs) -> None:
        """
        
        """
        self.cost = ca.DM(0.0)
        for k in range(0, self.horizon):
            xk, uk = self.get_x(k), self.get_u_actual(k) # X_steps[k], U_steps[k]
            self.cost += self.__lagrange__(xk, uk, path[k], nu_des, *args, k=k, **kwargs)
        self.cost += self.__mayer__(self.get_x(k+1), path[k+1], nu_des, *args, **kwargs)

    def get_initial_guess(self, x0:np.ndarray, u_prev_actual:np.ndarray, efficiency:np.ndarray, *args, mode:Literal['constant input', 'previous solution']='constant input', **kwargs) -> Tuple[ca.DM, ca.DM]:        
        match mode:
            case 'constant input':
                x_prev = x0.copy()
                X0 = x_prev

                U0Actual = []
                for _ in range(0, self.horizon):
                    x = self.dynamics(x_prev, u_prev_actual, efficiency)
                    X0 = ca.vertcat(X0, x)
                    U0Actual = ca.vertcat(U0Actual, u_prev_actual)

                return ca.DM(ca.vertcat(X0, U0Actual, U0Actual, efficiency)) # Assumption: Actual command = setpoint (steady state)
            case 'previous solution':
                if self.XOpt is None:
                    return self.get_initial_guess(x0, u_prev_actual, efficiency, *args, mode='constant input', **kwargs)
                X0 = ca.vertcat(self.XOpt[1::].reshape(-1), self.dynamics(self.XOpt[-1], self.UActualOpt[-1], efficiency)) # Assume N-th input is equal to N-1-th input
                U0Actual = ca.vertcat(self.UActualOpt[1::].reshape(-1), self.UActualOpt[-1])
                U0Setpoint = ca.vertcat(self.USetpointOpt[1::].reshape(-1), self.USetpointOpt[-1])
                return ca.DM(ca.vertcat(X0, U0Actual, U0Setpoint, efficiency))

    def set_initial_constraints(self, x0:np.ndarray, u_prev_actual:np.ndarray, efficiency:np.ndarray, *args, **kwargs) -> None:       
        self.dynamic_constraints[0:self.Nx] = self.get_x(0) - x0 # Replace first constraint
        self.dynamic_actuator_constraints[0:self.Nu] = self.get_u_actual(0) - u_prev_actual - self.actuators_dt * (self.get_u_setpoint(0) - u_prev_actual) / self.actuators_time_constant 
        self.efficiency_constraints[0:3] = self.efficiency - efficiency

    def solve(self, initial_guess, options:dict={'ipopt.print_level':0, 'print_time':0}, *args, **kwargs) -> Dict:
        nlp = {
            "x": ca.vertcat(self.X, self.UActual, self.USetpoint, self.efficiency),
            "f": self.cost,
            "g": ca.vertcat(self.dynamic_constraints, self.dynamic_actuator_constraints, self.efficiency_constraints)
        }

        solver_in = {
            "x0": initial_guess,
            "lbx": self.LBX + self.LBUA + self.LBUA + self.LBEfficiency, #  + self.LBFS + self.LBAS,
            "ubx": self.UBX + self.UBUA + self.UBUA + self.UBEfficiency, # + self.UBFS + self.UBAS,
            "lbg": self.LBDynamics + self.LBActuatorsDynamics + self.LBActuatorEfficiencyConstraints, # + self.LBObstacles,
            "ubg": self.UBDynamics + self.UBActuatorsDynamics + self.UBActuatorEfficiencyConstraints # + self.UBObstacles
        }

        self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        # print("OUT: ", solver_out)
        X = solver_out['x'].full().flatten()
        g = solver_out['g'].full().flatten()
        self.XOpt = X[0:self.Nx*(self.horizon+1)].reshape(-1, self.Nx)
        self.UActualOpt = X[self.Nx*(self.horizon+1): self.Nx*(self.horizon+1) + self.Nu*self.horizon].reshape(-1, self.Nu)
        self.USetpointOpt = X[self.Nx*(self.horizon+1) + self.Nu*self.horizon: self.Nx*(self.horizon+1) + 2*self.Nu*self.horizon].reshape(-1, self.Nu)
        self.DynamicsConstraintsOpt = g[0:(self.horizon+1)*self.Nx]
        self.ActuatorDynamicsConstraintsOpt = g[(self.horizon+1)*self.Nx:(self.horizon+1)*self.Nx+self.horizon*self.Nu]
        return solver_out

    def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, path:List[Tuple[float, float, float]], *args, delta=np.array([1, 1, 1]), sigma:np.ndarray=None, initial_guess=None, u_prev_actual:np.ndarray=None, **kwargs) -> List[np.ndarray]:
        # delta = np.array([1, 1, 1]) # disable adaptive MPC

        x0 = np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]])
        nu_des = np.array([nu_des[0], nu_des[1], nu_des[5]])
        if self.u_prev_actual is None:
            if u_prev_actual is None:
                u_prev_actual = np.array(self.Nu * [0.0])
        else:
            u_prev_actual = self.u_prev_actual.copy()

        # path = self.path.get_target_wpts_from(eta[0], eta[1], 0.1, self.horizon+1)
        initial_guess = initial_guess or self.get_initial_guess(x0, u_prev_actual, delta, *args, **kwargs)

        # if sigma is not None:
        #     self.singular_value_weight = 0.5 * (1e-3 * np.linalg.norm(sigma) / 0.1) + 0.5 * self.singular_value_weight

        # print(self.singular_value_weight)

        self.set_cost(path, nu_des, *args, **kwargs) # stage & terminal cost
        # print("u_prev_actual:", u_prev_actual)
        self.set_initial_constraints(x0, u_prev_actual, efficiency=delta, *args, **kwargs)
        self.solve(initial_guess, *args, **kwargs) # Solve for X, U, V
        cost = self.eval_solution(nu_des, path)
        current_cost = self.eval_current_state(path[0], nu_des)
        info = {'horizon': self.horizon, 'psi_des': path[0][2], 'cost': cost, 'current_cost': current_cost, 'prediction': self.XOpt}
        return self.__format_commands__(self.u_prev_setpoint, *args, **kwargs), info
    
    def reset(self):
        pass

    def get_x(self, k) -> ca.SX:
        return self.X[k*self.Nx:(k+1)*self.Nx]
    
    def get_u_actual(self, k) -> ca.SX:
        return self.UActual[k*self.Nu:(k+1)*self.Nu]
    
    def get_u_setpoint(self, k) -> ca.SX:
        return self.USetpoint[k*self.Nu:(k+1)*self.Nu]
    
    def get_x_opt(self, k) -> ca.SX:
        return self.XOpt[k]
    
    def get_u_actual_opt(self, k) -> ca.SX:
        return self.UActualOpt[k]
    
    def get_u_setpoint_opt(self, k) -> ca.SX:
        return self.USetpointOpt[k]

    @property
    def u_prev_actual(self) -> np.ndarray:
        if self.UActualOpt is not None:
            return self.UActualOpt[0]
        else:
            return None
        
    @property
    def u_prev_setpoint(self) -> np.ndarray:
        if self.USetpointOpt is not None:
            return self.USetpointOpt[0]
        else:
            return None
        
    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.XOpt is None:
            return ax
        ax.plot(self.XOpt[:, 1], self.XOpt[:, 0], c='green', linewidth=2)
        return ax
        
    def plot_results(self):
        """
        Plot MPC results:
        - Input setpoint commands (as steps, dashed)
        - Actual actuator values (as lines)
        - Command constraints (shaded region)
        - Psi values
        - 2D trajectory (north, east)
        - Speed values (u, v, r)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        XOpt = np.array(self.XOpt)  # shape: (horizon+1, Nx)
        USetpointOpt = np.array(self.USetpointOpt)  # shape: (horizon, Nu)
        UActualOpt = np.array(self.UActualOpt)      # shape: (horizon, Nu)

        timesteps_x = np.arange(XOpt.shape[0]) * self.dt
        timesteps_u = np.arange(USetpointOpt.shape[0]) * self.dt

        # Colors for forces and angles (consistent for actual/setpoint)
        force_colors = ['tab:blue', 'tab:orange', 'tab:green']
        angle_colors = ['tab:blue', 'tab:orange', 'tab:green']

        # Figure 1: Input commands (forces and angles)
        fig1, axs1 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot command constraints as shaded regions
        for i in range(3):
            axs1[0].axhspan(self.lbu[i], self.ubu[i], color=force_colors[i], alpha=0.08)
        for i in range(3):
            axs1[1].axhspan(self.lbu[i+3], self.ubu[i+3], color=angle_colors[i], alpha=0.08)

        # Forces
        for i in range(3):
            axs1[0].plot(timesteps_u, UActualOpt[:, i], label=f'Force {i+1} Actual', color=force_colors[i])
            axs1[0].step(timesteps_u, USetpointOpt[:, i], where='post', linestyle='--', label=f'Force {i+1} Setpoint', color=force_colors[i])
        axs1[0].set_ylabel('Force [N]')
        axs1[0].legend()
        axs1[0].grid(True)

        # Angles
        for i in range(3):
            axs1[1].plot(timesteps_u, UActualOpt[:, i+3], label=f'Angle {i+1} Actual', color=angle_colors[i])
            axs1[1].step(timesteps_u, USetpointOpt[:, i+3], where='post', linestyle='--', label=f'Angle {i+1} Setpoint', color=angle_colors[i])
        axs1[1].set_ylabel('Azimuth [rad]')
        axs1[1].set_xlabel('Time [s]')
        axs1[1].legend()
        axs1[1].grid(True)
        fig1.suptitle('Actuator Commands (Actual & Setpoint) with Constraints', fontsize=14)
        fig1.tight_layout(rect=[0, 0, 1, 0.97])

        # Figure 2: Psi values
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(timesteps_x, XOpt[:, 2], color='tab:blue')
        ax2.set_title('Yaw Angle (Psi) over Time', fontsize=14)
        ax2.set_ylim([-np.pi, np.pi])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Psi [rad]')
        ax2.grid(True)

        # Figure 3: 2D Trajectory (east, north) -> east on x, north on y
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        ax3.plot(XOpt[:, 1], XOpt[:, 0], marker='o', color='tab:green', label='Trajectory')
        ax3.set_title('2D Trajectory (East-North)', fontsize=14)
        ax3.set_xlabel('East [m]')
        ax3.set_ylabel('North [m]')
        ax3.grid(True)
        ax3.axis('equal')
        ax3.legend()

        # Figure 4: Speed values (u, v, r)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(timesteps_x, XOpt[:, 3], label='Surge (u)', color='tab:red')
        ax4.plot(timesteps_x, XOpt[:, 4], label='Sway (v)', color='tab:orange')
        ax4.plot(timesteps_x, XOpt[:, 5], label='Yaw rate (r)', color='tab:purple')
        ax4.set_title('Speed Components over Time', fontsize=14)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Speed [m/s] / Yaw rate [rad/s]')
        ax4.legend()
        ax4.grid(True)

        plt.show()

    def animate_heading(self, interval=50, arrow_length=0.5):
        """
        Animates the ship's position and heading (psi) in time according to the optimized states.
        The ship is shown as an arrow indicating heading in the east-north plane.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np

        XOpt = np.array(self.XOpt)
        east = XOpt[:, 1]
        north = XOpt[:, 0]
        psi = XOpt[:, 2]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(east, north, 'k--', alpha=0.3, label='Trajectory')
        ship_point, = ax.plot([], [], 'ro', markersize=8, label='Ship')
        arrow = None  # Will hold the current arrow
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_title('Ship Position & Heading Animation')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        def init():
            ship_point.set_data([], [])
            return ship_point,

        def update(frame):
            nonlocal arrow
            ship_point.set_data([east[frame]], [north[frame]])
            # Remove previous arrow if it exists
            if arrow is not None:
                arrow.remove()
            # Draw new arrow for heading
            print("psi: ", psi[frame]*180/np.pi)
            arrow = ax.arrow(
                east[frame], north[frame],
                arrow_length * np.sin(psi[frame]),
                arrow_length * np.cos(psi[frame]),
                head_width=0.2, head_length=0.2, fc='r', ec='r'
            )
            return ship_point, arrow

        ani = FuncAnimation(fig, update, frames=len(east), init_func=init,
                            blit=False, interval=interval, repeat=False)
        plt.show()


    
def test() -> None:
    from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF, RevoltThrusterParameters
    mpc = MPCPathTrackingRevolt(
        RevoltParameters3DOF(),
        RevoltThrusterParameters(),
        horizon=200,
        dt=0.05
    )

    mpc(None, np.array([0.0]+5*[0.0]), np.array(6*[0.0]), np.array(6*[0.0]), None, None, None, None, np.array([(mpc.horizon+1)*(2, 0.5, np.pi/2)]).reshape(-1, 3).tolist())
    # mpc(None, None, np.array(3*[1.0]), np.array(3*[0.0]), None, None, None, None, np.array([(mpc.horizon+1)*(2, 0.5, np.pi/2)]).reshape(-1, 3).tolist())
    # mpc(None, None, np.array(3*[-1.0]), np.array(3*[0.0]), None, None, None, None, np.array([(mpc.horizon+1)*(2, 0.5, np.pi/2)]).reshape(-1, 3).tolist())

    # 
    mpc.animate_heading()
    mpc.plot_results()

def test_full_system() -> None:
    from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF, RevoltThrusterParameters


if __name__ == "__main__":
    test()






# class MPCPathTrackingRevolt(IControl):
#     """
#     Model-Predictive Controller for path tracking of the ReVolt ASV, based on 

#     Reinforcement learning-based NMPC for tracking control of ASVs:Theory and experiments

#     and 

#     MPC-based Reinforcement Learning for a Simplified Freight Mission of Autonomous Surface Vehicles
    
#     """

#     Uopt:ca.DM = None
#     Xopt:ca.DM = None
#     Sopt:ca.DM = None
#     X:ca.SX = None
#     U:ca.SX = None
#     S:ca.SX = None

#     def __init__(
#             self,
#             lbx:Tuple,
#             ubx:Tuple,
#             lbu:Tuple,
#             ubu:Tuple,
#             lbdu:Tuple,
#             ubdu:Tuple,
#             *args,
#             horizon:int=20,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.lbx = np.array(lbx)
#         self.ubx = np.array(ubx)
#         self.lbu = np.array(lbu)
#         self.ubu = np.array(ubu)
#         self.lbdu = np.array(lbdu)
#         self.ubdu = np.array(ubdu)
#         self.horizon = horizon
#         self.Nu = len(lbu)
#         self.Nx = len(lbx)
#         self.init_nlp()

#     @abstractmethod
#     def __model__(self, xk:ca.SX, uk:ca.SX, *args, **kwargs) -> ca.SX:
#         return []
    
#     @abstractmethod
#     def __lagrange__(self, xk:ca.SX, uk:ca.SX, *args, sk:ca.SX=None, k:int=None, **kwargs) -> ca.SX:
#         """
#         k can be used if l(xk, uk) changes at each stage -> e.g. path tracking, discount factor, etc..
#         """
#         pass

#     @abstractmethod
#     def __mayer__(self, xf:ca.SX, *args, sf:ca.SX=None, **kwargs) -> ca.SX:
#         pass

#     @abstractmethod
#     def __format_commands__(self, commands:np.ndarray) -> List[np.ndarray]:
#         return [command for command in commands]

#     def set_delta_constraints(self, *args, **kwargs) -> None:
#         """
        
#         """
#         self.delta = self.get_u(0) # Reserve space for constraint du- <= u-u_prev <= du+ to be set at runtime
#         for k in range(0, self.horizon-1):
#             self.delta = ca.vertcat(self.delta, self.get_u(k+1) - self.get_u(k))
#         self.LBDelta = self.lbdu[None, :].repeat(self.horizon) # Lower Bound of Delta u, including du- <= u0 - u_prev <= du+
#         self.UBDelta = self.ubdu[None, :].repeat(self.horizon) # Upper Bound of Delta u, including du- <= u0 - u_prev <= du+

#     def set_dynamic_constraints(self, *args, **kwargs) -> None:
#         """
        
#         """
#         self.dynamics = self.get_x(0) # Reserve space for constraint x(0) = 0 to be set at runtime
#         for k in range(0, self.horizon):
#             self.dynamics = ca.vertcat(self.dynamics, self.get_x(k+1) - self.__model__(self.get_x(k), self.get_u(k), *args, **kwargs))

#         self.LBDynamics = np.array([self.Nx*[0.0]]).repeat(self.horizon + 1, axis=0).flatten() # Lower Bound Dynamics, including 0 <= x(0) - x0 <= 0
#         self.UBDynamics = np.array([self.Nx*[0.0]]).repeat(self.horizon + 1, axis=0).flatten() # Upper Bound Dynamics, including 0 <= x(0) - x0 <= 0

#     def set_states_and_commands_constraints(self, *args, **kwargs) -> None:
#         self.LBX = self.lbx[None, :].repeat(self.horizon + 1, axis=0).flatten()
#         self.UBX = self.ubx[None, :].repeat(self.horizon + 1, axis=0).flatten()
#         self.LBU = self.lbu[None, :].repeat(self.horizon, axis=0).flatten()
#         self.UBU = self.ubu[None, :].repeat(self.horizon, axis=0).flatten()

#     def init_nlp(self, *args, **kwargs) -> None:
#         """
        
#         """
#         # Decision Variables
#         self.X = ca.SX.sym("X", self.Nx * (self.horizon + 1))   # States
#         self.U = ca.SX.sym("U", self.Nu * self.horizon)         # Command input

#         # Constraints
#         self.set_states_and_commands_constraints(*args, **kwargs)   # x_k \in X, u_k \in U
#         self.set_dynamic_constraints(*args, **kwargs)               # x_k+1 = f(x_k, u_k)


#         # Initialize NLP  

#     def set_cost(self, *args, **kwargs) -> None:
#         """
        
#         """
#         self.cost = ca.DM(0.0)
#         # X_steps = ca.vertsplit(self.X, self.Nx)
#         # U_steps = ca.vertsplit(self.U, self.Nu)
#         for k in range(0, self.horizon):
#             xk, uk = self.get_x(k), self.get_u(k) # X_steps[k], U_steps[k]
#             self.cost += self.__lagrange__(xk, uk, *args, k=k, **kwargs)
#         self.cost += self.__mayer__(self.get_x(k+1), *args, **kwargs)

#     def get_initial_guess(self, eta:np.ndarray, nu:np.ndarray) -> Tuple[ca.DM, ca.DM]:
#         if self.u0_opt is None:
#             self.u0_opt = np.array(self.Nu * [0.0])
#         X0 = ca.DM()
#         U0 = ca.DM()
#         return X0, U0   

#     def set_initial_constraints(self, eta:np.ndarray, nu:np.ndarray, *args, u_prev:np.ndarray=None, **kwargs) -> None:       
#         self.dynamics[0:self.Nx] = self.get_x(0) - np.concatenate([eta, nu]) # Replace first constraint
#         self.delta[0:self.Nu] = self.get_u(0) - u_prev

#     def solve(self, initial_guess, options:dict={'ipopt.print_level':0, 'print_time':0}, *args, **kwargs) -> Dict:
#         nlp = {
#             "x": ca.vertcat(self.X, self.U),
#             "f": self.cost,
#             "g": ca.vertcat(self.dynamics, self.delta)
#         }

#         solver_in = {
#             "x0": initial_guess,
#             "lbx": self.LBX + self.LBU,
#             "ubx": self.UBX + self.UBU,
#             "lbg": self.LBDynamics + self.LBDelta,
#             "ubg": self.UBDynamics + self.UBDelta
#         }

#         self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, options)
#         solver_out = self.solver(**solver_in)
#         arr = solver_out['x'].full().flatten()
#         self.Xopt = arr[0:(self.horizon+1)*self.Nx].reshape((self.horizon+1, self.Nx)).T
#         self.Uopt = arr[(self.horizon+1)*self.Nx:].reshape((self.horizon, self.Nu)).T

#         # Remove last constraint (i.e. x[0] = x0) after solve is done, since it will change for next nlp
#         self.G = self.G[0:-self.Nx]
#         self.LBG = self.LBG[0:-self.Nx]
#         self.UBG = self.UBG[0:-self.Nx] 

#         return {}

#     def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, initial_guess=None, u_prev:np.ndarray=None, **kwargs) -> List[np.ndarray]:
#         initial_guess = initial_guess or self.get_initial_guess(eta, nu, *args, **kwargs)
#         self.set_cost(*args, **kwargs) # stage & terminal cost
#         self.set_initial_constraints(eta, nu, *args, **kwargs)
#         self.solve(initial_guess, *args, **kwargs) # Solve for X, U, V
#         return self.__format_commands__(self.u0_opt, *args, **kwargs)
    
#     def reset(self):
#         pass

#     def get_x(self, k) -> ca.SX:
#         return self.X[k*self.Nx:(k+1)*self.Nx]
    
#     def get_u(self, k) -> ca.SX:
#         return self.U[k*self.Nu:(k+1)*self.Nu]

#     @property
#     def u0_opt(self) -> np.ndarray:
#         if self.Uopt is not None:
#             return self.Uopt[0:self.Nu]
#         else:
#             raise RuntimeError(f"Uopt is empty. First call solve() or get() to get a valid solution.")

# class PathTrackingMPC(IMPC):
#     def __init__(
#             self,
#             *args,
#             Qx:np.ndarray=None,
#             Ru:np.ndarray=None,
#             Rdu:np.ndarray=None,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.Qx = Qx or np.eye(self.Nx)
#         self.Ru = Ru or np.eye(self.Nu)
#         self.Rdu = Rdu or np.eye(self.Nu)

#     def __model__(self, xk:ca.SX, uk:ca.SX) -> ca.SX:
#         return xk
    
#     def __lagrange__(self, xk:ca.SX, uk:ca.SX, path:np.ndarray, k:int, sk:ca.SX=None) -> ca.SX:
#         """
#         path is a (N, Nx) array describing the path to be followed
#         """
#         xdk = ...

#     def __mayer__(self, xf:ca.SX, sf:ca.SX=None) -> ca.SX:
#         pass

#     def __format_commands__(self, commands:np.ndarray) -> List[np.ndarray]:
#         return super().__format_commands__(commands)

#     def set_delta_constraints(self, *args, **kwargs) -> None:
#         return super().set_delta_constraints(*args, **kwargs)