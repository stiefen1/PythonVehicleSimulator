import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class KalmanFilter:
    """
    Linear Kalman Filter

    Built from https://www.geeksforgeeks.org/python/kalman-filter-in-python/
    """
    def __init__(
            self,
            F, B, H, Q, R, x0, P0: np.ndarray,
            *args,
            **kwargs
    ):
        self.F:np.ndarray = F # Linear system model -> x' = F@x + B@u
        self.B:np.ndarray = B # Input matrix
        self.H:np.ndarray = H # Observation matrix -> y = H@x
        self.Q:np.ndarray = Q # Process noise covariance
        self.R:np.ndarray = R # Measurement noise covariance
        self.x:np.ndarray = x0 # state
        self.P:np.ndarray = P0 # Error covariance
    
    def predict(self, u:np.ndarray) -> np.ndarray:
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z:np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate
        """
        S = self.H @ self.P @ self.H.T + self.R # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.H @ self.x # Residuals
        self.x = self.x + K @ y # Update state estimate through innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x
    
class IExtendedKalmanFilter(ABC):
    """
    Extended Kalman Filter (EKF) == Kalman Filter with linearized model at each time-step

    Mainly based on:
    https://en.wikipedia.org/wiki/Extended_Kalman_filter#Discrete-time_predict_and_update_equations


    Model is 
    x'  = f(x, u) + v   # v is the process noise
    y   = h(x) + w     # w is the measurement noise

    """

    def __init__(
            self,
            Q:np.ndarray,
            R:np.ndarray,
            x0:np.ndarray, 
            P0:np.ndarray,
            dt:float,
            *args,
            **kwargs
            ):
        self.Q:np.ndarray = Q # Process noise covariance
        self.R:np.ndarray = R # Measurment noise covariance
        self.x:np.ndarray = x0 # States
        self.P:np.ndarray = P0 # Expected Error Covariance
        self.dt = dt

    def __call__(self, u:np.ndarray, z:np.ndarray, *args, **kwargs) -> np.ndarray:
        self.predict(u)
        return self.update(z)

    @abstractmethod
    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """
        return x
    
    @abstractmethod
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        return np.eye(*x.shape)
    
    @abstractmethod
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        z = x # All states are measured
        return z

    @abstractmethod
    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.eye(x.shape[0])
    
    def predict(self, u:np.ndarray) -> np.ndarray:
        dFdx = self.dfdx(self.x, u)
        self.P = dFdx @ self.P @ dFdx.T + self.Q
        self.x = self.f(self.x, u)
        return self.x
    
    def update(self, z:np.ndarray) -> np.ndarray:
        """
        Returns updated state estimate based on measurement z
        """
        dHdx = self.dhdx(self.x)
        S = dHdx @ self.P @ dHdx.T + self.R # Residual covariance -> Expected combined uncertainty of prediction & measurement
        K = self.P @ dHdx.T @ np.linalg.inv(S) # Kalman Gain -> balance factor for blending prediction and measurements
        y = z - self.h(self.x) # Residuals between measurement and measurement model
        self.x = self.x + K @ y # Update state estimate through innovation
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ dHdx) @ self.P
        return self.x
    

class EKFRevolt3(IExtendedKalmanFilter):
    """
    3Dofs Extended Kalman Filter for the Revolt vessel


    CURRENTLY THIS KALMAN TAKES GENREALIZED FROCES AS INPUT BUT IT PREVENTS IT FROM BEING TESTED FOR ACTUATOR FAULTS, BECAUSE IT DOES NOT
    INCLUDE THE ACTUATION MODEL. -> WE HAVE TO CHANGE IT.

    According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

    heading +- 0.2°
    position +- 1cm
    u, v, r += 0.05
    
    """
    def __init__(
            self,
            x0:np.ndarray,
            dt:float,
            Q:np.ndarray,
            R:np.ndarray,
            P0:np.ndarray,
            *args,
            **kwargs
            ):
        super().__init__(Q, R, x0, P0, dt, *args, **kwargs)
        self.init_model()

    def init_model(self) -> None:
        import sympy as sp
        from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF
        params = RevoltParameters3DOF()
    
        n, e, psi, u, v, r = sp.symbols('n e psi u v r')
        tau_u, tau_v, tau_r = sp.symbols('tau_u tau_v tau_r')
        eta = sp.Matrix([n, e, psi])
        nu = sp.Matrix([u, v, r])
        x = sp.Matrix([n, e, psi, u, v, r])  # Full state vector
        tau = sp.Matrix([tau_u, tau_v, tau_r])

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

        # Now compute the Jacobian
        J_x = f.jacobian(x)

        # Convert to lambdified function
        self.Jx_lambda = sp.lambdify([n, e, psi, u, v, r], J_x, 'numpy')
        self.f_lambda = sp.lambdify([n, e, psi, u, v, r] + [tau_u, tau_v, tau_r], f, 'numpy')

    def f(self, x:np.ndarray, u:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v | Here us is the actuator's generalized force
        """
        return self.f_lambda(*(x.tolist() + u.tolist())).squeeze()
    
    def dfdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        return self.Jx_lambda(*x.tolist())
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        z = x # All states are measured
        return z

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.eye(*x.shape)

def test():
    F = np.array([[1, 1], [0, 1]]) 
    B = np.array([[0.5], [1]])
    H = np.array([[1, 0]])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])
    x0 = np.array([[0], [1]])
    P0 = np.array([[1, 0], [0, 1]])
    kf = KalmanFilter(F, B, H, Q, R, x0, P0)
    u = np.array([[1]])
    e = np.array([[1]])

    predicted_state = kf.predict(u)
    print("Predicted state:\n", predicted_state)

    updated_state = kf.update(e)
    print("Updated state:\n", updated_state)

def test_ekf() -> None:
    Q = np.eye(6, 6)
    R = np.eye(6, 6)
    x0 = np.ones((6,)) * 0.1
    P0 = np.ones((6, 6)) * 0.1
    ekf = EKFRevolt3(Q, R, x0, P0, dt=0.1)
    prediction = ekf.predict(np.array(6*[0.]))
    updated_state = ekf.update(np.array([0.0]*6))
    print("EKF Prediction:\n", prediction)
    print("EKF Update:\n", updated_state)
    
     
if __name__ == "__main__":
    test_ekf()


