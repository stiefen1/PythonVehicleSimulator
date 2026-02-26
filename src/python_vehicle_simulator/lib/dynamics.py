import numpy as np, numpy.typing as npt, casadi as cs
from abc import ABC, abstractmethod
from typing import Literal, Tuple, Optional, get_args

# Available discretization methods
DiscretizationMethod = Literal['rk4', 'euler']

class IDynamics(ABC):
    _f: cs.Function  # Continuous-time dynamics
    _fd: cs.Function # Discrete-time dynamics

    def __init__(
            self,
            nx: int,    # States
            nu: int,    # Control inputs
            nt: int,    # Number of parameters theta
            nd: int,    # Number of disturbances
            dt: float,  # Sampling time
            *args,
            **kwargs
    ):
        self.nx = nx # States
        self.nu = nu # Control inputs
        self.nt = nt # Number of parameters
        self.nd = nd # Number of disturbances
        self.dt = dt # Sampling time

        self._init_dynamics()

    @abstractmethod
    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: Optional[cs.SX], *args, **kwargs) -> cs.SX:
        """
        x:              states                      (nx,)
        u:              control commands            (nu,)
        theta:          parameters (e.g. faults)    (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        """
        pass

    def _get_continuous_time_dynamics(self) -> cs.Function:
        """
        Creates CasADi function for continuous-time dynamics.
        
        Returns:
            cs.Function: Continuous dynamics function f(x, u, theta)
        """
        x = cs.SX.sym('x', self.nx)                     # type: ignore
        u = cs.SX.sym('u', self.nu)                     # type: ignore
        theta = cs.SX.sym('theta', self.nt)             # type: ignore
        disturbance = cs.SX.sym('disturbance', self.nd) # type: ignore

        return cs.Function('continuous_time_dynamics', [x, u, theta, disturbance], [self.continuous_time_dynamics(x, u, theta, disturbance)])

    def _discretize_dynamics(self, continuous_time_dynamics: cs.Function, method: DiscretizationMethod='rk4') -> cs.Function:
        """
        Discretizes continuous-time dynamics using specified numerical method.
        
        continuous_time_dynamics:    CasADi continuous dynamics function
        method:                 Discretization method ('rk4', 'euler')
        
        Returns:
            cs.Function: Discrete-time dynamics function
        """
        available_methods = get_args(DiscretizationMethod)
        assert method in available_methods, f"Discretization method '{method}' is not implemented. Available methods: {available_methods}"
        
        # Call the method dynamically using getattr
        discretization_method = getattr(self, '_' + method)
        return discretization_method(continuous_time_dynamics)
    
    def _get_linearized_models(self) -> Tuple[cs.Function, cs.Function, cs.Function, cs.Function]:
        """
        Creates linearized models (Jacobians) for continuous and discrete dynamics.
        
        Returns:
            Tuple: (A_continuous, B_continuous, A_discrete, B_discrete) functions
        """
        x = cs.SX.sym('x', self.nx)                     # type: ignore
        u = cs.SX.sym('u', self.nu)                     # type: ignore
        theta = cs.SX.sym('theta', self.nt)             # type: ignore
        disturbance = cs.SX.sym('disturbance', self.nd) # type: ignore

        return (
            cs.Function("A_continuous", [x, u, theta, disturbance], [cs.jacobian(self._f(x, u, theta, disturbance), x)]),
            cs.Function("B_continuous", [x, u, theta, disturbance], [cs.jacobian(self._f(x, u, theta, disturbance), u)]),
            cs.Function("A_discrete", [x, u, theta, disturbance], [cs.jacobian(self._fd(x, u, theta, disturbance), x)]),
            cs.Function("B_discrete", [x, u, theta, disturbance], [cs.jacobian(self._fd(x, u, theta, disturbance), u)])
        )
        
    def _rk4(self, continuous_time_dynamics: cs.Function) -> cs.Function:
        """
        Fourth-order Runge-Kutta discretization method.
        
        continuous_time_dynamics:    CasADi continuous dynamics function
        
        Returns:
            cs.Function: Discrete-time dynamics using RK4
        """
        x = cs.SX.sym('x', self.nx)                     # type: ignore
        u = cs.SX.sym('u', self.nu)                     # type: ignore
        theta = cs.SX.sym('theta', self.nt)             # type: ignore
        disturbance = cs.SX.sym('disturbance', self.nd) # type: ignore

        # RK4 integration
        k1 = continuous_time_dynamics(x, u, theta, disturbance)
        k2 = continuous_time_dynamics(x + 0.5 * self.dt * k1, u, theta, disturbance)           # type: ignore
        k3 = continuous_time_dynamics(x + 0.5 * self.dt * k2, u, theta, disturbance)           # type: ignore
        k4 = continuous_time_dynamics(x + self.dt * k3, u, theta, disturbance)                 # type: ignore
        x_next = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)            # type: ignore

        return cs.Function('discrete_dynamics', [x, u, theta, disturbance], [x_next])

    def _euler(self, continuous_time_dynamics: cs.Function) -> cs.Function:
        """
        Forward Euler discretization method.
        
        continuous_time_dynamics:    CasADi continuous dynamics function
        
        Returns:
            cs.Function: Discrete-time dynamics using Euler method
        """
        x = cs.SX.sym('x', self.nx)                     # type: ignore
        u = cs.SX.sym('u', self.nu)                     # type: ignore
        theta = cs.SX.sym('theta', self.nt)             # type: ignore
        disturbance = cs.SX.sym('disturbance', self.nd) # type: ignore
        
        # Euler integration: x_next = x + dt * f(x, u, theta)
        x_next = x + self.dt * continuous_time_dynamics(x, u, theta, disturbance)  # type: ignore
        
        return cs.Function('discrete_dynamics', [x, u, theta, disturbance], [x_next])

    def _init_dynamics(self) -> None:
        """
        Initializes continuous and discrete dynamics functions.
        
        Sets:
            self._f:  Continuous-time dynamics function
            self._fd: Discrete-time dynamics function
        """
        self._f = self._get_continuous_time_dynamics()
        self._fd = self._discretize_dynamics(self._f)
        self._A_function, self._B_function, self._Ad_function, self._Bd_function = self._get_linearized_models()

    def A(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Continuous-time state matrix (∂f/∂x).
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: State matrix A (nx, nx)
        """
        return np.array(self._A_function(x, u, theta, disturbance))
    
    def B(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Continuous-time input matrix (∂f/∂u).
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: Input matrix B (nx, nu)
        """
        return np.array(self._B_function(x, u, theta, disturbance))
    
    def Ad(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Discrete-time state matrix (∂fd/∂x).
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: Discrete state matrix Ad (nx, nx)
        """
        return np.array(self._Ad_function(x, u, theta, disturbance))
    
    def Bd(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Discrete-time input matrix (∂fd/∂u).
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: Discrete input matrix Bd (nx, nu)
        """
        return np.array(self._Bd_function(x, u, theta, disturbance))
    
    def f(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Continuous-time dynamics function.
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: State derivatives dx/dt (nx,)
        """
        return np.array(self._f(x, u, theta, disturbance))
    
    def fd(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Discrete-time dynamics function.
        
        x:              current states              (nx,)
        u:              control commands            (nu,)  
        theta:          parameters                  (nt,)
        disturbance:    disturbance (e.g. wind)     (nd,)
        
        Returns:
            npt.NDArray: Next states x[k+1] (nx,)
        """
        return np.array(self._fd(x, u, theta, disturbance))

    
if __name__ == "__main__":
    pass