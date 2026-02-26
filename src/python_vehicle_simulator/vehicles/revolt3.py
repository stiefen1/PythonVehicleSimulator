from math import pi
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from python_vehicle_simulator.lib.physics import GRAVITY, RHO
from python_vehicle_simulator.utils.math_fn import R_casadi
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.diagnosis import IDiagnosis
from python_vehicle_simulator.lib.dynamics import IDynamics
from python_vehicle_simulator.lib.thruster import THRUSTER_LENGTH_DEFAULT, THRUSTER_WIDTH_DEFAULT, THRUSTER_GEOMETRY, ROTATION_MATRIX
import casadi as cs, math, numpy as np, numpy.typing as npt
from python_vehicle_simulator.utils.math_fn import Rzyx


INF = float('inf')

@dataclass
class RevoltBowThrusterParams: # front
    """
    Parameters for ReVolt bow (front) thruster configuration.
    
    Defines thrust coefficients, force limits, speed limits, 
    and azimuth constraints for the bow thruster.
    """
    ## Propellers       
    T_n: float = 1.0                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 1.518e-3                                     # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 6.172e-4                                     # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 14                                           # Max positive force, one propeller
    f_min: float = 0 # -6.1                                     # Max negative force, one propeller
    speed_max: float = math.sqrt(f_max/k_pos)                   # Max (positive) propeller speed
    speed_min: float = -math.sqrt(-f_min/k_neg)
    alpha_max: float = math.pi                                  # Max (positive) propeller speed
    alpha_min: float = -math.pi                                 # Min (negative) propeller speed
    max_radians_per_step: float = math.pi/6
    max_newton_per_step: float = 10
    geometry = THRUSTER_GEOMETRY(THRUSTER_LENGTH_DEFAULT, THRUSTER_WIDTH_DEFAULT)

@dataclass
class RevoltSternThrusterParams: # back
    """
    Parameters for ReVolt stern (back) thruster configuration.
    
    Defines thrust coefficients, force limits, speed limits,
    and azimuth constraints for the stern thrusters.
    """
    ## Propellers       
    T_n: float = 1.0                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 2.7e-3                                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 2.7e-3                                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 25                                           # Max positive force, one propeller
    f_min: float = 0 # -25                                      # Max negative force, one propeller
    speed_max: float = math.sqrt(f_max/k_pos)                   # Max (positive) propeller speed
    speed_min: float = -math.sqrt(-f_min/k_neg)
    alpha_max: float = math.pi                                      # Max (positive) propeller speed
    alpha_min: float = -math.pi                                     # Min (negative) propeller speed
    max_radians_per_step: float = math.pi/6
    max_newton_per_step: float = 10
    geometry = THRUSTER_GEOMETRY(THRUSTER_LENGTH_DEFAULT, THRUSTER_WIDTH_DEFAULT)

@dataclass
class RevoltThrusterParameters:
    """
    Complete thruster configuration for ReVolt vessel.
    
    Aggregates parameters for all thrusters (2 stern + 1 bow)
    and provides thrust allocation matrix for force/torque computation.
    """
    ## Propellers       
    thrusters: List = field(init=False)                         
    k_pos: npt.NDArray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: npt.NDArray = field(init=False)                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: npt.NDArray = field(init=False)                       # Max positive force, one propeller
    f_min: npt.NDArray = field(init=False)                       # Max negative force, one propeller
    speed_max: npt.NDArray = field(init=False)                   # Max (positive) propeller speed
    speed_min: npt.NDArray = field(init=False)
    alpha_min: npt.NDArray = field(init=False)                   # Max (positive) propeller speed
    alpha_max: npt.NDArray = field(init=False)                   # Min (negative) propeller speed
    xy: npt.NDArray = field(init=False)                          # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    time_constant: npt.NDArray = field(init=False)
    geometries: List = field(init=False)

    def __post_init__(self):
        """
        Initialize thruster arrays and thrust allocation matrix.
        
        Creates arrays for all thruster parameters and defines
        the thrust allocation matrix T for force/torque computation.
        """
        self.thrusters = [RevoltSternThrusterParams(), RevoltSternThrusterParams(), RevoltBowThrusterParams()]  
        self.k_pos: npt.NDArray = np.array([thruster.k_pos for thruster in self.thrusters])      # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: npt.NDArray = np.array([thruster.k_neg for thruster in self.thrusters])      # Negative Bollard, one propeller
        self.f_max: npt.NDArray = np.array([thruster.f_max for thruster in self.thrusters])      # Max positive force, one propeller
        self.f_min: npt.NDArray = np.array([thruster.f_min for thruster in self.thrusters])      # np.array([-25, -25, -6.1]) 
        
        self.speed_max = np.array([thruster.speed_max for thruster in self.thrusters])          # Thruster speed constraints
        self.speed_min = np.array([thruster.speed_min for thruster in self.thrusters])
        self.alpha_min = np.array([thruster.alpha_min for thruster in self.thrusters])          # Azimuth angles constraints
        self.alpha_max = np.array([thruster.alpha_max for thruster in self.thrusters])
        
        self.xy = np.array([[-1.65, -0.15], [-1.65, 0.15], [1.15, 0.0]])                        # Azimuth thruster positions
        self.T_n = np.array([thruster.T_n for thruster in self.thrusters])
        self.T_a = np.array([thruster.T_a for thruster in self.thrusters])

        self.Ti = lambda alpha, lx, ly : np.array([                                              # Unused
            cs.cos(alpha),
            cs.sin(alpha),
            lx * cs.sin(alpha) - ly * cs.cos(alpha)
        ])

        ####### WARNING : IF YOU CHANGE Alpha YOU HAVE TO DO IT AS WELL IN RL ENVIRONMENTS, IT IS NOT LINKED ######
        self.Alpha = lambda a1, a2, a3 : cs.vertcat(
            cs.horzcat(cs.cos(a1), cs.sin(a1), self.xy[0, 0]*cs.sin(a1) - self.xy[0, 1] * cs.cos(a1)),
            cs.horzcat(cs.cos(a2), cs.sin(a2), self.xy[1, 0]*cs.sin(a2) - self.xy[1, 1] * cs.cos(a2)),
            cs.horzcat(cs.cos(a3), cs.sin(a3), self.xy[2, 0]*cs.sin(a3) - self.xy[2, 1] * cs.cos(a3))
        ).T

        self.geometries = [thruster.geometry for thruster in self.thrusters]

@dataclass
class RevoltParameters3DOF:
    """
    Physical and hydrodynamic parameters for ReVolt vessel in 3DOF.
    
    Defines vessel geometry, mass properties, hydrodynamic coefficients,
    and constraints for 3DOF motion (surge, sway, yaw).
    """
    Nx: int = 3
    loa: float = 3.0                                    # Length Over All (m) assumed equal to LPP
    beam: float = 0.72                                  # Beam (m)
    initial_draft:float = 0.25                          # Initial draft
    volume:float = 0.268                                # m^3 volume
    volume_iz:float = 0.310                             # m^5 volument moment of inertia

    R44: float = 0.36 * beam                             # radii of gyration (m)
    R55: float = 0.26 * loa
    R66: float = 0.26 * loa

    ## Time constants
    T_yaw: float = 3.0                                  # Time constant in yaw (s)

    ## Mass & Payload
    m: float = 157.0                                    # Mass (kg) from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.63)
    mp: float = 100.0                                   # Payload mass (kg)
    rg: npt.NDArray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))  
    rp: npt.NDArray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))          # Location of payload (m)

    ## State constraints
    lbx: npt.NDArray = field(default_factory=lambda: np.array([-INF, -INF, -pi, -1, -1, -pi/6]))
    ubx: npt.NDArray = field(default_factory=lambda: np.array([INF, INF, pi, knot_to_m_per_sec(3), 1, pi/6]))


    def __post_init__(self):
        """
        Compute derived parameters and matrices.
        
        Calculates total mass, corrected center of gravity,
        hydrodynamic coefficients, and system matrices (M, C, D).
        """
        self.m_tot = self.m + self.mp
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp) # corrected center of gravity with payload

        # Basic calculations
        self.Xudot, self.Yvdot, self.Yrdot, self.Nvdot = (-self.volume*RHO*np.array([0.0253, 0.1802, 0.0085 * self.loa, 0.0099 * self.loa**2])).tolist()
        self.Xu, self.Yv, self.Yr, self.Nv, self.Nr = (-np.array([
            0.102 * RHO * self.volume * GRAVITY / self.loa,
            1.212 * GRAVITY / self.loa,
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.0601 * RHO * self.volume * np.sqrt(GRAVITY * self.loa)
            ])).tolist()
        self.Iz = self.m_tot / self.volume * self.volume_iz

        ## Inertia Matrix
        MRB = np.array([
            [self.m_tot, 0, 0],
            [0, self.m_tot, self.m_tot*self.rg[0]],
            [0, self.m_tot*self.rg[0], self.Iz]
        ])
        MA = np.array([
            [-self.Xudot, 0, 0],
            [0, -self.Yvdot, -self.Yrdot],
            [0, -self.Yrdot, -self.Nvdot]
        ])
        self.Minv = np.linalg.inv(MA + MRB)

        ## Coriolis-Centripetal Matrix
        self.CRB = lambda nu : np.array([
            [0, 0, -self.m_tot * (self.rg[0]*nu[2] + nu[1])],
            [0, 0, self.m_tot * nu[0]],
            [self.m_tot * (self.rg[0]*nu[2] + nu[1]), -self.m_tot * nu[0], 0]
        ])
        self.CA = lambda nu_r : np.array([
            [0, 0, self.Yvdot * nu_r[1] + self.Yrdot * nu_r[2]],
            [0, 0, -self.Xudot * nu_r[0]],
            [-self.Yvdot * nu_r[1]-self.Yrdot * nu_r[2], self.Xudot * nu_r[0], 0]
        ])

        ## Damping Matrix
        self.D = np.array([
            [-self.Xu, 0, 0],
            [0, -self.Yv, -self.Yr],
            [0, -self.Nv, -self.Nr]
        ])

class ReVolt3Dynamics(IDynamics):
    actuators_params = RevoltThrusterParameters()
    vessel_params = RevoltParameters3DOF()

    def __init__(
            self,
            dt: float,
            *args, 
            **kwargs
    ):
        """
        ReVolt 3DOF dynamics model with azimuth thrusters.
        
        dt:     Sampling time
        
        States: [eta, nu, azimuth_angles, thruster_speeds] (12,)
        Inputs: [azimuth_setpoints, speed_setpoints] (6,)

        Azimuth thrusters are organized in the following order:
        - Stern port        (xy=[-1.65, -0.15])
        - Stern starboard   (xy=[-1.65, 0.15])
        - Bow               (xy=[1.15, 0.0])
        """
        nx, nu, np, nd = 18, 6, 6, 3 # states, commands, parameters, disturbances
        super().__init__(nx, nu, np, nd, dt, *args, **kwargs)

    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: cs.SX, *args, **kwargs) -> cs.SX:
        """
        x:              states: eta, nu, thruster speeds, azimuth angles    (nx,)
        u:              control commands                                    (nu,)
        theta:          parameters (e.g. faults)                            (np,)
        disturbance:    disturbance (e.g. wind)                             (nd,)

        Azimuth thrusters are organized in the following order:
        - Stern port        (xy=[-1.65, -0.15])
        - Stern starboard   (xy=[-1.65, 0.15])
        - Bow               (xy=[1.15, 0.0])
        """
        x_dot = cs.SX.zeros(self.nx, 1) # type: ignore

        eta_3dofs, nu_3dofs = cs.vcat([x[0], x[1], x[5]]), cs.vcat([x[6], x[7], x[11]])
        azimuth, thruster_speed = x[12:15], x[15:18]
        azimuth_setpoint = cs.fmax(cs.fmin(u[0:3], self.actuators_params.alpha_max), self.actuators_params.alpha_min)
        speed_setpoint = cs.fmax(cs.fmin(u[3:6], self.actuators_params.speed_max), self.actuators_params.speed_min)
        azimuth_stucked, propeller_effectiveness = theta[0:3], theta[3:6]

        # Generalized force generated by actuators
        thrust = propeller_effectiveness * self.actuators_params.k_pos * thruster_speed * thruster_speed
        tau_actuators = self.actuators_params.Alpha(azimuth[0], azimuth[1], azimuth[2]) @ thrust

        # Hull dynamics (body frame)
        Minv, C, D = self.vessel_params.Minv, self.vessel_params.CA(nu_3dofs) + self.vessel_params.CRB(nu_3dofs), self.vessel_params.D
        nu_dot_3dofs = Minv @ (tau_actuators + disturbance - C @ nu_3dofs - D @ nu_3dofs)
        x_dot[6] = nu_dot_3dofs[0]
        x_dot[7] = nu_dot_3dofs[1]
        x_dot[11] = nu_dot_3dofs[2]

        # Ship's kinematics (body -> NED)
        eta_dot_3dofs = cs.mtimes(R_casadi(eta_3dofs[2]), nu_3dofs)
        x_dot[0] = eta_dot_3dofs[0]
        x_dot[1] = eta_dot_3dofs[1]
        x_dot[5] = eta_dot_3dofs[2]

        # Actuator's dynamics: Low-pass
        x_dot[12:15] = azimuth_stucked * (azimuth_setpoint - azimuth) / self.actuators_params.T_a
        x_dot[15:18] = (speed_setpoint - thruster_speed) / self.actuators_params.T_n

        return x_dot


class ReVolt3(IVessel):
    vessel_params = RevoltParameters3DOF()
    actuator_params = RevoltThrusterParameters()

    def __init__(
            self,
            dt: float,
            eta: Tuple = (0, 0, 0),
            nu: Tuple = (0, 0, 0),
            thruster_speeds: Tuple = (0, 0, 0),
            azimuth_angles: Tuple = (0, 0, 0),
            guidance: Optional[IGuidance] = None,
            navigation: Optional[INavigation] = None,
            control: Optional[IControl] = None,
            diagnosis: Optional[IDiagnosis] = None,
            mmsi: Optional[str] = None,
            verbose_level: int = 0
    ):
        """
        ReVolt autonomous surface vessel with 3DOF dynamics.
        
        dt:             Sampling time
        eta:            Initial position [x, y, psi]        (3,)
        nu:             Initial velocity [u, v, r]          (3,)
        guidance:       Guidance system (optional)
        navigation:     Navigation system (optional)
        control:        Control system (optional)
        diagnosis:      Diagnosis system (optional)
        mmsi:           Maritime Mobile Service Identity
        verbose_level:  Verbosity level for logging
        """
        super().__init__(
            self.vessel_params.loa,
            self.vessel_params.beam,
            ReVolt3Dynamics(dt),
            states=(eta[0], eta[1], 0, 0, 0, eta[2], nu[0], nu[1], 0, 0, 0, nu[2], *azimuth_angles, *thruster_speeds),
            guidance=guidance,
            navigation=navigation,
            control=control,
            diagnosis=diagnosis,
            name='ReVolt3',
            mmsi=mmsi,
            verbose_level=verbose_level,
        )
    
    def __dynamics__(self, control_commands:npt.NDArray, current:Current, wind:Wind, *args, theta: Optional[npt.NDArray] = None, **kwargs) -> np.ndarray:
        """
        Vessel dynamics step with environmental disturbances.
        
        control_commands:   Thruster commands [azimuth, speeds]  (6,)
        current:            Current disturbance model
        wind:               Wind disturbance model
        theta:              Fault parameters (optional)          (6,)
        
        Returns:
            List: [next eta, next nu, next alpha, next thruster speed]

        Azimuth thrusters are organized in the following order:
        - Stern port        (xy=[-1.65, -0.15])
        - Stern starboard   (xy=[-1.65, 0.15])
        - Bow               (xy=[1.15, 0.0])
        """
        disturbance = np.array([0, 0, 0]) # Define it as a function of current, wind
        theta = theta if theta is not None else np.array(self.dynamics.nt * [1.0])
        x = self.dynamics.fd(self.states, control_commands, theta, disturbance).flatten()
        return x
    
    def __plot__(self, ax, *args, verbose:int=0, **kwargs):
        """
        x = East
        y = North
        z = -depth
        """
        ax.scatter(self.eta[1], self.eta[0], *args, **kwargs)
        ax.plot(*self.geometry_for_2D_plot, *args, **kwargs)
        for i in range(3):
            envelope = (ROTATION_MATRIX(self.states[12 + i]) @ self.actuator_params.geometries[i].T) + self.actuator_params.xy[i].reshape(-1, 1)
            envelope_in_ned_frame = Rzyx(*self.eta.to_numpy()[3:6].tolist())[0:2, 0:2] @ envelope + self.eta.to_numpy()[0:2, None]
            ax.plot(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :], *args, **kwargs)
        return ax
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt, time
    revolt = ReVolt3(0.1, nu=(0.7, 0, 0), thruster_speeds=(200, 200, 200))

    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    for t in np.linspace(0, 100, int(100//0.1)+1):
        revolt.step(None, None, [], [], control_commands=np.array([np.pi/2, np.pi/2, np.pi/2 * np.sin(2*np.pi*0.2*t), 200, 200, 200]))
        ax.cla()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        revolt.plot(ax=ax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)
        
        