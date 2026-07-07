import numpy as np, numpy.typing as npt, casadi as cs
from abc import ABC, abstractmethod
from typing import Literal, Tuple, Optional, get_args

# Available discretization methods
DiscretizationMethod = Literal['rk4', 'euler']

class IDynamics(ABC):
    _f: cs.Function  # Continuous-time dynamics
    _fd: cs.Function # Discrete-time dynamics
    _fd_batch_cache: dict[int, cs.Function]
    _rollout_mapaccum_cache: dict[Tuple[int, int], cs.Function]

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
        self._fd_batch_cache = {}
        self._rollout_mapaccum_cache = {}

    def _get_fd_batch_function(self, batch_size: int) -> cs.Function:
        """
        Returns a cached CasADi mapped discrete-time dynamics function.
        """
        if batch_size not in self._fd_batch_cache:
            self._fd_batch_cache[batch_size] = self._fd.map(batch_size)
        return self._fd_batch_cache[batch_size]

    def _get_rollout_mapaccum_function(self, batch_size: int, horizon: int) -> cs.Function:
        """
        Returns a cached CasADi mapaccum rollout function for a fixed batch size and horizon.
        """
        key = (batch_size, horizon)
        if key in self._rollout_mapaccum_cache:
            return self._rollout_mapaccum_cache[key]

        fd_batch_function = self._get_fd_batch_function(batch_size)
        x_dim = self.nx * batch_size
        w_dim = (self.nu + self.nt + self.nd) * batch_size

        xk_flat = cs.SX.sym('xk_flat', x_dim)  # type: ignore
        wk_flat = cs.SX.sym('wk_flat', w_dim)  # type: ignore

        offset_u = self.nu * batch_size
        offset_t = offset_u + self.nt * batch_size
        u_flat = wk_flat[0:offset_u]
        theta_flat = wk_flat[offset_u:offset_t]
        disturbance_flat = wk_flat[offset_t:]

        xk = cs.reshape(xk_flat, self.nx, batch_size)
        uk = cs.reshape(u_flat, self.nu, batch_size)
        thetak = cs.reshape(theta_flat, self.nt, batch_size)
        disturbancek = cs.reshape(disturbance_flat, self.nd, batch_size)

        xk_next = fd_batch_function(xk, uk, thetak, disturbancek)
        step_function = cs.Function('rollout_step', [xk_flat, wk_flat], [cs.reshape(xk_next, x_dim, 1)])

        self._rollout_mapaccum_cache[key] = step_function.mapaccum(horizon)
        return self._rollout_mapaccum_cache[key]

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

    def fd_batch(self, x: npt.NDArray, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Batched discrete-time dynamics with public shape (N, *).

        x:              batch of current states        (N, nx)
        u:              batch of control commands      (N, nu)
        theta:          batch of parameters            (N, nt)
        disturbance:    batch of disturbances          (N, nd)

        Returns:
            npt.NDArray: Batch of next states          (N, nx)
        """
        x = np.asarray(x)
        u = np.asarray(u)
        theta = np.asarray(theta)
        disturbance = np.asarray(disturbance)

        assert x.ndim == 2 and x.shape[1] == self.nx, f"x must have shape (N, {self.nx})"
        assert u.ndim == 2 and u.shape[1] == self.nu, f"u must have shape (N, {self.nu})"
        assert theta.ndim == 2 and theta.shape[1] == self.nt, f"theta must have shape (N, {self.nt})"
        assert disturbance.ndim == 2 and disturbance.shape[1] == self.nd, f"disturbance must have shape (N, {self.nd})"

        N = x.shape[0]
        assert N > 0, "batch size N must be > 0"
        assert u.shape[0] == N and theta.shape[0] == N and disturbance.shape[0] == N, "all inputs must share the same batch size N"

        fd_batch_function = self._get_fd_batch_function(N)

        # CasADi mapped functions operate on column-wise batches: (dim, N)
        x_next = np.array(fd_batch_function(x.T, u.T, theta.T, disturbance.T))
        return x_next.T

    def _rollout_batch_loop_fast(self, x0: npt.NDArray, U: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Fast rollout path using a Python time loop but minimizing per-step overhead.
        """
        N = x0.shape[0]
        T = U.shape[2]

        fd_batch_function = self._get_fd_batch_function(N)

        # Convert once to CasADi column-wise batch layout: (dim, N)
        xk = x0.T
        U_col = np.transpose(U, (1, 0, 2))
        theta_col = theta.T
        disturbance_col = disturbance.T

        X_col = np.empty((self.nx, N, T + 1), dtype=x0.dtype)
        X_col[:, :, 0] = xk

        for k in range(T):
            xk = np.array(fd_batch_function(xk, U_col[:, :, k], theta_col, disturbance_col))
            X_col[:, :, k + 1] = xk

        return np.transpose(X_col, (1, 0, 2))

    def _rollout_batch_mapaccum_fast(self, x0: npt.NDArray, U: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Fast rollout path using CasADi mapaccum over the horizon.
        """
        N = x0.shape[0]
        T = U.shape[2]

        rollout_function = self._get_rollout_mapaccum_function(N, T)

        # Build horizon inputs wk = [u_k, theta, disturbance] for each k.
        # CasADi reshape is column-major, so we must pack using Fortran order.
        U_col = np.transpose(U, (1, 0, 2)).reshape(self.nu * N, T, order='F')

        theta_col = np.tile(theta.T.reshape(self.nt * N, 1, order='F'), (1, T))
        disturbance_col = np.tile(disturbance.T.reshape(self.nd * N, 1, order='F'), (1, T))
        W = np.vstack((U_col, theta_col, disturbance_col))

        x0_flat = x0.T.reshape(self.nx * N, 1, order='F')
        Xk = np.array(rollout_function(x0_flat, W))

        # mapaccum returns x1..xT. Prepend x0 for a full (T+1) trajectory.
        X_col = np.empty((self.nx, N, T + 1), dtype=x0.dtype)
        X_col[:, :, 0] = x0.T
        X_col[:, :, 1:] = Xk.reshape(self.nx, N, T, order='F')
        return np.transpose(X_col, (1, 0, 2))

    def rollout_batch(
            self,
            x0: npt.NDArray,
            U: npt.NDArray,
            theta: npt.NDArray,
            disturbance: npt.NDArray,
            method: Literal['loop', 'mapaccum'] = 'loop'
    ) -> npt.NDArray:
        """
        Batched rollout with public shape (N, *, T).

        x0:             initial states               (N, nx)
        U:              control sequence             (N, nu, T)
        theta:          parameters                   (N, nt)
        disturbance:    disturbances                 (N, nd)

        Returns:
            npt.NDArray: state trajectory           (N, nx, T+1)
        """
        x0 = np.asarray(x0)
        U = np.asarray(U)
        theta = np.asarray(theta)
        disturbance = np.asarray(disturbance)

        assert x0.ndim == 2 and x0.shape[1] == self.nx, f"x0 must have shape (N, {self.nx})"
        assert U.ndim == 3 and U.shape[1] == self.nu, f"U must have shape (N, {self.nu}, T)"
        assert theta.ndim == 2 and theta.shape[1] == self.nt, f"theta must have shape (N, {self.nt})"
        assert disturbance.ndim == 2 and disturbance.shape[1] == self.nd, f"disturbance must have shape (N, {self.nd})"

        N = x0.shape[0]
        T = U.shape[2]
        assert N > 0, "batch size N must be > 0"
        assert U.shape[0] == N and theta.shape[0] == N and disturbance.shape[0] == N, "all inputs must share the same batch size N"
        assert method in get_args(Literal['loop', 'mapaccum']), "method must be 'loop' or 'mapaccum'"

        if method == 'mapaccum':
            return self._rollout_batch_mapaccum_fast(x0, U, theta, disturbance)
        return self._rollout_batch_loop_fast(x0, U, theta, disturbance)

    
if __name__ == "__main__":
    pass