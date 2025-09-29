from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Callable, Tuple

"""

Diagnosis object takes as input:
    state measurements, u_actual and parameters of the object to diagnose

stores:
    previous states, u_actual and diagnosis

returns
    new diagnosis as a dictionnary

"""

class IDiagnosis(ABC):
    def __init__(
            self,
            eta:np.ndarray,
            nu:np.ndarray,
            params,
            dt:float,
            *args,
            u_actual:List[np.ndarray]=None,
            **kwargs
    ):
        self.params = params # PARAMETERS DEPEND ON WHAT WE WANT TO DIAGNOSE
        self.eta = eta
        self.nu = nu
        self.dt = dt
        self.u_actual = u_actual
        self.prev = {'diagnosis': None, 'info': None}

    def get(self, eta:np.ndarray, nu:np.ndarray, u_actual:List[np.ndarray], *args, **kwargs) -> Tuple[Dict, Dict]:
        diagnosis, info = self.__get__(eta, nu, u_actual, *args, **kwargs)
        self.eta = eta.copy()
        self.nu = nu.copy()
        self.u_actual = u_actual
        self.prev = {'diagnosis': diagnosis, 'info': info}
        return diagnosis, info # diagnosis and info

    @abstractmethod
    def __get__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List[np.ndarray], *args, **kwargs) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def __call__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List[np.ndarray], *args, **kwargs) -> Tuple[Dict, Dict]:
        return self.get(eta, nu, u_actual, *args, **kwargs)
    
class Diagnosis(IDiagnosis):
    def __init__(
            self,
            eta:np.ndarray,
            nu:np.ndarray,
            params,
            dt:float,
            *args,
            u_actual:List[np.ndarray]=None,
            **kwargs
    ):
        self.params = params # PARAMETERS DEPEND ON WHAT WE WANT TO DIAGNOSE
        self.eta = eta
        self.nu = nu
        self.dt = dt
        self.u_actual = u_actual

    def __get__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List[np.ndarray], *args, **kwargs) -> Tuple[Dict, Dict]:
        return super().__get__(eta, nu, u_actual, *args, **kwargs)
    
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
            eta:np.ndarray,
            nu:np.ndarray,
            dt:float,
            *args,
            delta0:np.ndarray=np.ones((3,)),
            **kwargs
    ) -> None:
        super().__init__(
            eta=eta.copy(),
            nu=nu.copy(),
            params=delta0.copy(),
            dt=dt,
            *args,
            **kwargs
        )
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

        # TODO: IMPLEMENT ACTUATORS DYNAMICS USING ACTUAL VALUES
        # T = lambda alpha, lx, ly : np.array([
        #     ca.cos(alpha),
        #     ca.sin(alpha),
        #     lx*ca.sin(alpha) - ly * ca.cos(alpha)
        # ])

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

        # Now compute the Jacobian
        J_delta = f.jacobian(delta)

        # Convert to lambdified function
        self.Jdelta_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3], J_delta, 'numpy')
        self.f_lambda = sp.lambdify([n, e, psi, u, v, r, a1, n1, a2, n2, a3, n3, d1, d2, d3], f, 'numpy')

class GradientDescentDiagnosisRevolt3Actuators(DiagnosisOfRevolt3Actuators):
    def __init__(
            self,
            eta:np.ndarray,
            nu:np.ndarray,
            dt:float,
            *args,
            lr:float = 1e3, # learning rate
            delta0:np.ndarray=np.ones((3,)),
            **kwargs
    ) -> None:
        super().__init__(eta=eta, nu=nu, dt=dt, *args, delta0=delta0, **kwargs)
        self.lr = lr

    def __get__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List, *args, **kwargs) -> Dict:
        eta_prev, nu_prev = self.eta, self.nu
        a1, n1 = u_actual[0]['u_actual'].tolist() # u_actual is the previous command input
        a2, n2 = u_actual[1]['u_actual'].tolist()
        a3, n3 = u_actual[2]['u_actual'].tolist()
        x = np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]])[:, None]
        error = self.f_lambda(eta_prev[0], eta_prev[1], eta_prev[5], nu_prev[0], nu_prev[1], nu_prev[5], a1, n1, a2, n2, a3, n3, self.params[0], self.params[1], self.params[2]) - x
        dfddelta = self.Jdelta_lambda(eta_prev[0], eta_prev[1], eta_prev[5], nu_prev[0], nu_prev[1], nu_prev[5], a1, n1, a2, n2, a3, n3)
        gradient = 2 * error.T @ dfddelta

        # # Coordinate descent method: Gauss-Southwell -> improves A LOT without noise but struggles in real conditions
        # keep_max = np.zeros_like(gradient)
        # keep_max[0, np.argmax(gradient, axis=1)] = 1
        # gradient = keep_max * gradient
        
        self.params = np.clip((self.params - self.lr * gradient)[0], 0, 1)
        info = {}
        # print('delta: ', self.params)
        return {'delta': self.params}, info
    
class ParticleFilterDiagnosisRevolt3Actuators(DiagnosisOfRevolt3Actuators):
    def __init__(
            self,
            eta:np.ndarray,
            nu:np.ndarray,
            dt:float,
            meas_cov:np.ndarray,
            process_cov:np.ndarray,
            *args,
            n_particles:int = 100, # Number of particles
            delta0:np.ndarray=np.ones((3,)),
            effective_size_thresh:int = None,
            **kwargs
    ) -> None:
        super().__init__(eta=eta, nu=nu, dt=dt, *args, delta0=delta0, **kwargs)
        self.meas_cov_inv:np.ndarray = np.linalg.inv(meas_cov)
        self.process_cov:np.ndarray = process_cov
        self.n_particles = n_particles
        self.effective_size_thresh = effective_size_thresh or n_particles // 10
        self.particles = np.ones((self.n_particles, 3)) * delta0
        self.weights = np.ones((self.n_particles,)) / self.n_particles

    def __get__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List, *args, **kwargs) -> Dict:
        eta_prev, nu_prev = self.eta, self.nu
        y = np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]) # Measurements
        # print("y: ", y)
        a1, n1 = u_actual[0]['u_actual'].tolist()
        a2, n2 = u_actual[1]['u_actual'].tolist()
        a3, n3 = u_actual[2]['u_actual'].tolist()
        
        x_hat = np.ndarray((6, self.n_particles))
        process_noise = np.random.multivariate_normal(np.zeros(3), self.process_cov, size=self.n_particles)
        # new_particles = np.clip(self.particles + process_noise, 0, 1)
        new_particles = self.particles + process_noise

        
        # Process model
        for i in range(self.n_particles):
            # print("particle i: ", new_particles[i])
            x_hat[:, i] = self.f_lambda(
                eta_prev[0], eta_prev[1], eta_prev[5], nu_prev[0], nu_prev[1], nu_prev[5],
                a1, n1, a2, n2, a3, n3,
                new_particles[i, 0], new_particles[i, 1], new_particles[i, 2]
            )[:, 0]
            
        # Update weights
        diff = x_hat - y[:, None]                       # shape (6, N)
        quad_form = np.einsum('ij,jk,ik->i', diff.T, self.meas_cov_inv, diff.T)  # length N
        likelihood = self.weights * np.exp(-0.5 * quad_form)
        # likelihood = self.weights * np.exp(-0.5*np.diag((x_hat-y[:, None]).T @ self.meas_cov @ (x_hat-y[:, None]))) # Check likelihood of this w.r.t measurement noise

        if np.sum(likelihood) > 1e-9:
            self.weights = likelihood / np.sum(likelihood)
        else:
            self.weights = np.ones((self.n_particles,)) / self.n_particles


        # Find best
        # best_particle = np.sum(self.weights[:, None] * new_particles, axis=0)
        # best_particle = new_particles[np.argmax(self.weights)]

        K = self.n_particles // 1  # or any number <= n_particles
        # 1) Get indices of particles with largest weights
        topK_idx = np.argsort(self.weights)[-K:]  # last K indices have largest weights
        # 2) Take weighted average among these K particles
        topK_weights = self.weights[topK_idx]
        topK_weights /= np.sum(topK_weights)     # normalize weights
        best_particle = np.sum(new_particles[topK_idx] * topK_weights[:, None], axis=0)

        # Resample with probability self.weights pick new_particles, if effective sample size < threshold
        effective_sample_size = 1 / np.sum(np.square(self.weights))
        if effective_sample_size < self.effective_size_thresh:
            # print("Resample!")
            # Draw N indices according to weights
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
            # Resample particles and reset weights
            new_particles = new_particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

        self.particles = new_particles.copy()
        # print("std: ", np.std(self.particles, axis=0))

        info = {'particles': new_particles.copy().T, 'top-k-particles': new_particles[topK_idx].copy().T}
        return {'delta': best_particle}, info
    
class LinearParametersEstimationDiagnosisRevolt3Actuators(DiagnosisOfRevolt3Actuators):
    def __init__(
            self,
            eta:np.ndarray,
            nu:np.ndarray,
            dt:float,
            *args,
            n_particles:float = 100, # Number of particles
            delta0:np.ndarray=np.ones((3,)),
            **kwargs
    ) -> None:
        super().__init__(eta=eta, nu=nu, dt=dt, *args, delta0=delta0, **kwargs)
        self.n_particles = n_particles

    def __get__(self, eta:np.ndarray, nu:np.ndarray, u_actual:List, *args, **kwargs) -> Dict:
        info = {}
        return {}, info
    
def test() -> None:
    diag = GradientDescentDiagnosisRevolt3Actuators(
        np.array(6*[0.0]),
        np.array(6*[0.0]),
        0.1
    )

if __name__ == "__main__":
    test()