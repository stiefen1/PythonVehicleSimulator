from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Callable, Tuple
import matplotlib.pyplot as plt
from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter

"""

Diagnosis object takes as input:
    state measurements, u_actual and parameters of the object to diagnose

stores:
    previous states, u_actual and diagnosis

returns
    new diagnosis as a dictionnary

"""

# According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

#         heading +- 0.2°
#         position +- 1cm
#         u, v +- 0.05 m/s
#         r not specified, assuming it is very low according to graph. let's say r +- 0.05 deg/s as well
Q_REVOLT = np.diag([0.3**2, 0.3**2, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 15*np.pi/180/3600])
R_REVOLT = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2])


class IDiagnosis(ABC):
    def __init__(
            self,
            states: np.ndarray,
            params,
            dt:float,
            *args,
            **kwargs
    ):
        self.params = params # PARAMETERS DEPEND ON WHAT WE WANT TO DIAGNOSE
        self.states = states
        self.dt = dt
        self.prev = {'diagnosis': None, 'info': None}

    def get(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        diagnosis, info = self.__get__(states, *args, **kwargs)
        self.states = states.copy()
        self.prev = {'diagnosis': diagnosis, 'info': info}
        return diagnosis, info # diagnosis and info

    @abstractmethod
    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def __call__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        return self.get(states, *args, **kwargs)
    
class Diagnosis(IDiagnosis):
    def __init__(
            self,
            states: np.ndarray,
            params,
            dt:float,
            *args,
            **kwargs
    ):
        self.params = params # PARAMETERS DEPEND ON WHAT WE WANT TO DIAGNOSE
        self.states = states
        self.dt = dt

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        return super().__get__(states, *args, **kwargs)
    
def deep_simplify(expr):
    """Try multiple simplification strategies"""
    import sympy as sp
    try:
        # Convert floats to rationals first
        expr = sp.nsimplify(expr, rational=True)
        # Then simplify
        expr = sp.simplify(expr)
        # Try factoring
        expr = sp.factor(expr)
        return expr
    except:
        return expr
class DiagnosisOfRevolt3Actuators(IDiagnosis):
    Jtheta_lambda:Callable
    f_lambda:Callable

    def __init__(
            self,
            states:np.ndarray,
            dt:float,
            *args,
            delta0:np.ndarray=np.ones((3,)),
            **kwargs
    ) -> None:
        super().__init__(
            states=states.copy(),
            params=delta0.copy(),
            dt=dt,
            *args,
            **kwargs
        )
        self.init_model()

    # Not required but here for testing purpose
    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        return {}, {}

    def init_model(self) -> None:
        import sympy as sp
        from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF
        from python_vehicle_simulator.vehicles.revolt3 import RevoltThrusterParameters
        params = RevoltParameters3DOF()
        actuators_params = RevoltThrusterParameters()
    
        n, e, psi, u, v, r = sp.symbols('n e psi u v r')
        n1, n2, n3 = sp.symbols('n1 n2 n3')
        a1, a2, a3 = sp.symbols('a1 a2 a3')
        d1, d2, d3 = sp.symbols('d1, d2, d3', real=True, finite=True)
        n_hat, e_hat, psi_hat, u_hat, v_hat, r_hat = sp.symbols('n_hat e_hat psi_hat u_hat v_hat r_hat')
        delta = sp.Matrix([d1, d2, d3])
        eta = sp.Matrix([n, e, psi])
        nu = sp.Matrix([u, v, r])
        x_hat = sp.Matrix([n_hat, e_hat, psi_hat, u_hat, v_hat, r_hat])
        # x = sp.Matrix([n, e, psi, u, v, r])  # Full state vector



        ######### I must add impact of input to get complete model!!! ########

        # Kinematic equations: eta_dot = J(psi) * nu
        J_psi = sp.Matrix([
            [sp.cos(psi), -sp.sin(psi), 0],
            [sp.sin(psi), sp.cos(psi), 0],
            [0, 0, 1]
        ])
        eta_dot = J_psi @ nu
        eta_kp1 = eta + eta_dot * self.dt 

        # Dynamic equations: nu_dot = M^(-1) * (tau - C*nu - D*nu)
        MRB = sp.Matrix([
            [params.m_tot, 0, 0],
            [0, params.m_tot, params.m_tot*params.rg[0]],
            [0, params.m_tot*params.rg[0], params.Iz]
        ])
        MA = sp.Matrix([
            [-params.Xudot, 0, 0],
            [0, -params.Yvdot, -params.Yrdot],
            [0, -params.Yrdot, -params.Nvdot]
        ])
        CRB = sp.Matrix([
            [0, 0, -params.m_tot * (params.rg[0]*r + v)],
            [0, 0, params.m_tot * u],
            [params.m_tot * (params.rg[0]*r + v), -params.m_tot * u, 0]
        ])
        CA = sp.Matrix([
            [0, 0, params.Yvdot * v + params.Yrdot * r],
            [0, 0, -params.Xudot * u],
            [-params.Yvdot * v - params.Yrdot * r, params.Xudot * u, 0]
        ])
        D = sp.Matrix([
            [-params.Xu, 0, 0],
            [0, -params.Yv, -params.Yr],
            [0, -params.Nv, -params.Nr]
        ])


        # tau u
        tau_u = d1 * actuators_params.k_pos[0] * sp.cos(a1) * n1**2 # * sp.Abs(n1) # th1 *n1 + ...
        tau_u += d2 * actuators_params.k_pos[1] * sp.cos(a2) * n2**2 # * sp.Abs(n2)
        tau_u += d3 * actuators_params.k_pos[2] * sp.cos(a3) * n3**2 # * sp.Abs(n3)

        # tau v
        tau_v = d1 * actuators_params.k_pos[0] * sp.sin(a1) * n1**2 # * sp.Abs(n1) # th1 *n1 + ...
        tau_v += d2 * actuators_params.k_pos[1] * sp.sin(a2) * n2**2 # * sp.Abs(n2)
        tau_v += d3 * actuators_params.k_pos[2] * sp.sin(a3) * n3**2 # * sp.Abs(n3)

        # tau r
        tau_r = d1 * actuators_params.k_pos[0] * (actuators_params.xy[0, 0]*sp.sin(a1) - actuators_params.xy[0, 1] * sp.cos(a1)) * n1**2 # * sp.Abs(n1)
        tau_r += d2 * actuators_params.k_pos[1] * (actuators_params.xy[1, 0]*sp.sin(a2) - actuators_params.xy[1, 1] * sp.cos(a2)) * n2**2 # * sp.Abs(n2)
        tau_r += d3 * actuators_params.k_pos[2] * (actuators_params.xy[2, 0]*sp.sin(a3) - actuators_params.xy[2, 1] * sp.cos(a3)) * n3**2 # * sp.Abs(n3)

        tau = sp.Matrix([tau_u, tau_v, tau_r])

        nu_dot = (MA + MRB).inv() @ (tau - (CA + CRB) @ nu - D @ nu)
        nu_kp1 = nu + nu_dot * self.dt

        # Complete dynamics: f = [eta_dot; nu_dot]
        f = sp.Matrix([
            eta_kp1[0, 0],  # n_k+1
            eta_kp1[1, 0],  # e_k+1
            eta_kp1[2, 0],  # psi_k+1
            nu_kp1[0, 0],   # u_k+1
            nu_kp1[1, 0],   # v_k+1
            nu_kp1[2, 0]    # r_k+1
        ])

        # To compute exact value
        self.FT_lambda = sp.lambdify([a1, n1, a2, n2, a3, n3, d1, d2, d3], tau, 'numpy')

        # Now compute the Jacobian
        J_delta = f.jacobian(delta)
        Alin = f.jacobian([n, e, psi, u, v, r])
        Blin = f.jacobian([a1, n1, a2, n2, a3, n3])
        Elin = f.jacobian([d1, d2, d3])

        # Convert to lambdified function
        self.Jdelta_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3], J_delta, 'numpy')
        self.f_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3, d1, d2, d3], f, 'numpy')
        self.Alin_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3, d1, d2, d3], Alin, 'numpy')
        self.Blin_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3, d1, d2, d3], Blin, 'numpy')
        self.Elin_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3, d1, d2, d3], Elin, 'numpy')

class GradientDescentDiagnosisRevolt3Actuators(DiagnosisOfRevolt3Actuators):
    def __init__(
            self,
            states:np.ndarray,
            dt:float,
            *args,
            lr:float = 1e3, # learning rate
            delta0:np.ndarray=np.ones((3,)),
            **kwargs
    ) -> None:
        super().__init__(states=states, dt=dt, *args, delta0=delta0, **kwargs)
        self.lr = lr

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        eta_prev, nu_prev = self.states[0:6], self.states[6:12]
        a1, n1 = states[12], states[15] 
        a2, n2 = states[13], states[16]
        a3, n3 = states[14], states[17]
        x = np.array([states[0], states[1], states[5], states[6], states[7], states[11]])[:, None]
        error = self.f_lambda(eta_prev[0], eta_prev[1], eta_prev[5], nu_prev[0], nu_prev[1], nu_prev[5], a1, n1, a2, n2, a3, n3, self.params[0], self.params[1], self.params[2]) - x
        dfddelta = self.Jdelta_lambda(eta_prev[0], eta_prev[1], eta_prev[5], nu_prev[0], nu_prev[1], nu_prev[5], a1, n1, a2, n2, a3, n3)
        gradient = 2 * error.T @ dfddelta

        # # Coordinate descent method: Gauss-Southwell -> improves A LOT without noise but struggles in real conditions
        # keep_max = np.zeros_like(gradient)
        # keep_max[0, np.argmax(gradient, axis=1)] = 1
        # gradient = keep_max * gradient
        
        self.params = np.clip((self.params - self.lr * gradient)[0], 0, 1)
        info = {}
        return {'delta': self.params}, info
    
class AugmentedEKFDiagnosisRevolt3Actuators(DiagnosisOfRevolt3Actuators, IExtendedKalmanFilter):
    """
    Augmented EKF for joint state & fault estimation since the relation between efficiency and output is linear
    """
    def __init__(
            self,
            states_0:np.ndarray,
            dt:float,
            *args,
            Q:np.ndarray = np.eye(9), # Process noise -> Small means we trust our model A LOT                               # When using 1e-5 for Both Q and R results are okay
            R:np.ndarray = R_REVOLT, # Measurement noise -> Small means we trust our measurements A LOT
            P0:np.ndarray = np.eye(9)*1e-5,
            efficiency_0:np.ndarray=np.array([1, 1, 1]),
            **kwargs
    ) -> None:
        Q[0:6, 0:6] = Q_REVOLT
        DiagnosisOfRevolt3Actuators.__init__(self, states=states_0, dt=dt, *args, delta0=efficiency_0, **kwargs)
        eta_0_3dof = np.array([states_0[0], states_0[1], states_0[5]])
        nu_0_3dof = np.array([states_0[6], states_0[7], states_0[11]])
        IExtendedKalmanFilter.__init__(self, Q=Q, R=R, x0=np.concatenate([eta_0_3dof, nu_0_3dof, efficiency_0]), P0=P0, dt=dt)
        self.init_model()

    def init_model(self) -> None:
        import sympy as sp
        from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF
        from python_vehicle_simulator.vehicles.revolt3 import RevoltThrusterParameters
        params = RevoltParameters3DOF()
        actuators_params = RevoltThrusterParameters()
    
        n, e, psi, u, v, r = sp.symbols('n e psi u v r')
        n1, n2, n3 = sp.symbols('n1 n2 n3')
        a1, a2, a3 = sp.symbols('a1 a2 a3')
        d1, d2, d3 = sp.symbols('d1, d2, d3', real=True, finite=True)
        delta = sp.Matrix([d1, d2, d3])
        eta = sp.Matrix([n, e, psi])
        nu = sp.Matrix([u, v, r])
        x = sp.Matrix([n, e, psi, u, v, r, d1, d2, d3])  # Full state vector



        ######### I must add impact of input to get complete model!!! ########

        # Kinematic equations: eta_dot = J(psi) * nu
        J_psi = sp.Matrix([
            [sp.cos(psi), -sp.sin(psi), 0],
            [sp.sin(psi), sp.cos(psi), 0],
            [0, 0, 1]
        ])
        eta_dot = J_psi @ nu
        eta_kp1 = eta + eta_dot * self.dt 
        delta_kp1 = delta

        # Dynamic equations: nu_dot = M^(-1) * (tau - C*nu - D*nu)
        MRB = sp.Matrix([
            [params.m_tot, 0, 0],
            [0, params.m_tot, params.m_tot*params.rg[0]],
            [0, params.m_tot*params.rg[0], params.Iz]
        ])
        MA = sp.Matrix([
            [-params.Xudot, 0, 0],
            [0, -params.Yvdot, -params.Yrdot],
            [0, -params.Yrdot, -params.Nvdot]
        ])
        CRB = sp.Matrix([
            [0, 0, -params.m_tot * (params.rg[0]*r + v)],
            [0, 0, params.m_tot * u],
            [params.m_tot * (params.rg[0]*r + v), -params.m_tot * u, 0]
        ])
        CA = sp.Matrix([
            [0, 0, params.Yvdot * v + params.Yrdot * r],
            [0, 0, -params.Xudot * u],
            [-params.Yvdot * v - params.Yrdot * r, params.Xudot * u, 0]
        ])
        D = sp.Matrix([
            [-params.Xu, 0, 0],
            [0, -params.Yv, -params.Yr],
            [0, -params.Nv, -params.Nr]
        ])

        # tau u
        tau_u = d1 * actuators_params.k_pos[0] * sp.cos(a1) * n1**2 # * sp.Abs(n1) # th1 *n1 + ...
        tau_u += d2 * actuators_params.k_pos[1] * sp.cos(a2) * n2**2 # * sp.Abs(n2)
        tau_u += d3 * actuators_params.k_pos[2] * sp.cos(a3) * n3**2 # * sp.Abs(n3)

        # tau v
        tau_v = d1 * actuators_params.k_pos[0] * sp.sin(a1) * n1**2 # * sp.Abs(n1) # th1 *n1 + ...
        tau_v += d2 * actuators_params.k_pos[1] * sp.sin(a2) * n2**2 # * sp.Abs(n2)
        tau_v += d3 * actuators_params.k_pos[2] * sp.sin(a3) * n3**2 # * sp.Abs(n3)

        # tau r
        tau_r = d1 * actuators_params.k_pos[0] * (actuators_params.xy[0, 0]*sp.sin(a1) - actuators_params.xy[0, 1] * sp.cos(a1)) * n1**2 # * sp.Abs(n1)
        tau_r += d2 * actuators_params.k_pos[1] * (actuators_params.xy[1, 0]*sp.sin(a2) - actuators_params.xy[1, 1] * sp.cos(a2)) * n2**2 # * sp.Abs(n2)
        tau_r += d3 * actuators_params.k_pos[2] * (actuators_params.xy[2, 0]*sp.sin(a3) - actuators_params.xy[2, 1] * sp.cos(a3)) * n3**2 # * sp.Abs(n3)

        tau = sp.Matrix([tau_u, tau_v, tau_r])

        nu_dot = (MA + MRB).inv() @ (tau - (CA + CRB) @ nu - D @ nu)
        nu_kp1 = nu + nu_dot * self.dt

        # Complete dynamics: f = [eta_dot; nu_dot]
        f = sp.Matrix([
            eta_kp1[0, 0],  # n_k+1
            eta_kp1[1, 0],  # e_k+1
            eta_kp1[2, 0],  # psi_k+1
            nu_kp1[0, 0],   # u_k+1
            nu_kp1[1, 0],   # v_k+1
            nu_kp1[2, 0],   # r_k+1
            delta_kp1[0, 0],
            delta_kp1[1, 0],
            delta_kp1[2, 0]
        ])


        # Now compute the Jacobian
        J_x = f.jacobian(x)

        # Convert to lambdified function
        self.Jx_lambda = sp.lambdify([n, e, psi, u, v, r, d1, d2, d3, a1, n1, a2, n2, a3, n3], J_x, 'numpy')
        self.f_lambda = sp.lambdify([n, e, psi, u, v, r, d1, d2, d3, a1, n1, a2, n2, a3, n3], f, 'numpy')

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x_+ = f(x, u) + v | Here us is the actuator's generalized force
        """
        xkp1 = self.f_lambda(*(x.tolist() + u.tolist())).squeeze()
        # xkp1[6:9] = np.clip(xkp1[6:9] + np.random.normal(0, np.array(3*[0.1])), 0, 1) # random walk using process noise
        return xkp1
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        dfdx = self.Jx_lambda(*(x.tolist()+u.tolist()))
        return dfdx
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        z = x[0:6] # All states are measured except faults
        return z

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.hstack([np.eye(6), np.zeros((6, 3))])


    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        eta = np.array([states[0], states[1], states[5]]) # Measurements
        nu = np.array([states[6], states[7], states[11]])
        a1, n1 = states[12], states[15] 
        a2, n2 = states[13], states[16]
        a3, n3 = states[14], states[17]
        prediction = self.predict(np.array([a1, n1, a2, n2, a3, n3]))
        # print("EKF Fault Diagnosis: ", prediction[0:6]-np.concatenate([states]))
        print("Prediction covariance: ", np.diag(self.P)[6:9])
        self.update(np.concatenate([states]))
        self.params = self.x[6:9]
        info = {}
        return {'delta': self.params}, info

    
def test() -> None:
    import numpy as np


    x0 = np.array(6*[0])
    u0 = np.array([1, 20, 0.1, 20, -0.2, 20])
    d0 = np.array(3*[1])

    x = x0 + 0.5
    u = u0 * 1.5
    d = d0.copy() 

    diagnoser = DiagnosisOfRevolt3Actuators(
        np.array(18*[0]),
        dt=0.01
    )

    A0 = diagnoser.Alin_lambda(*(x0.tolist()+u0.tolist()+d.tolist()))
    B0 = diagnoser.Blin_lambda(*(x0.tolist()+u0.tolist()+d.tolist()))
    E0 = diagnoser.Elin_lambda(*(x0.tolist()+u0.tolist()+d.tolist()))
    f0 = diagnoser.f_lambda(*(x0.tolist()+u0.tolist()+d.tolist()))
    f = diagnoser.f_lambda(*(x.tolist()+u.tolist()+d.tolist()))

    print("Shapes: \n")
    print(A0.shape, B0.shape, E0.shape, f0.shape, f.shape)

    f_approx = f0 + A0 @ (x-x0)[:, None] + B0 @ (u-u0)[:, None] + E0 @ (d-d0)[:, None]

    print(f)
    print(f_approx)

if __name__ == "__main__":
    test()