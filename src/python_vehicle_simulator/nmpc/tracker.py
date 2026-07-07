from typing import Optional, Literal, Dict, Any, cast

from python_vehicle_simulator.nmpc.base import NMPCBase
from python_vehicle_simulator.lib.weather import Current, Wind

import casadi as cs, numpy as np, numpy.typing as npt

class NMPCTracker(NMPCBase):
    """
    Nonlinear MPC for tracking states.
    """
    STATES_TO_TRACK_IDX = []

    def __init__(
            self,
            horizon: int, # discrete steps
            dynamics: cs.Function, # discrete dynamics mapping f(x, u, theta, disturbance) -> x_next
            nx: int,
            nu: int,
            ntheta: int,
            ndisturbance: int,
            u_lb: npt.NDArray,
            u_ub: npt.NDArray,
            *args,
            x_lb: Optional[npt.NDArray] = None,
            x_ub: Optional[npt.NDArray] = None,
            solver: Literal["ipopt"] = "ipopt",
            solver_opts: Optional[Dict] = None,
            u_0: Optional[npt.NDArray] = None, # previous control command
            seed: Optional[int] = None,
            Q: Optional[npt.NDArray] = None,
            R: Optional[npt.NDArray] = None,
            QN: Optional[npt.NDArray] = None,
            **kwargs
    ):
        if len(self.STATES_TO_TRACK_IDX) == 0: # set default states to track to all 
            self.STATES_TO_TRACK_IDX = np.arange(0, nx).tolist()

        self.Q = np.asarray(Q if Q is not None else np.eye(nx), dtype=float)
        self.R = np.asarray(R if R is not None else 1e-2 * np.eye(nu), dtype=float)
        self.QN = np.asarray(QN if QN is not None else 10*self.Q, dtype=float)
        if self.Q.shape != (len(self.STATES_TO_TRACK_IDX), len(self.STATES_TO_TRACK_IDX)):
            raise ValueError(f"Q must be ({len(self.STATES_TO_TRACK_IDX)}x{len(self.STATES_TO_TRACK_IDX)}), got {Q.shape}") # type: ignore
        if self.QN.shape != (len(self.STATES_TO_TRACK_IDX), len(self.STATES_TO_TRACK_IDX)):
            raise ValueError(f"QN must be ({len(self.STATES_TO_TRACK_IDX)}x{len(self.STATES_TO_TRACK_IDX)}), got {QN.shape}") # type: ignore
        if self.R.shape != (nu, nu):
            raise ValueError(f"R must be {nu}x{nu}, got {R.shape}") # type: ignore

        super().__init__(
            horizon,
            dynamics,
            nx,
            nu,
            ntheta,
            ndisturbance,
            u_lb,
            u_ub,
            *args,
            x_lb=x_lb,
            x_ub=x_ub,
            solver=solver,
            solver_opts=solver_opts,
            u_0=u_0,
            seed=seed,
            **kwargs,
        )

    def _tracking_error(self, x: cs.SX, x_ref: cs.SX) -> Any:
        x_to_track = x[self.STATES_TO_TRACK_IDX]
        x_ref_to_track = x_ref[self.STATES_TO_TRACK_IDX]
        error = (x_to_track - x_ref_to_track)
        error_norm = cs.sqrt(error * error)
        return error_norm

    def lagrange(self, xk: cs.SX, uk: cs.SX, x_ref_k: cs.SX, k: int) -> cs.SX:
        e = self._tracking_error(xk, x_ref_k)
        return cs.mtimes([e.T, self.Q, e]) + cs.mtimes([uk.T, self.R, uk])

    def mayer(self, xN: cs.SX, x_ref_N: cs.SX) -> cs.SX:
        eN = self._tracking_error(xN, x_ref_N)
        return cs.mtimes([eN.T, self.QN, eN])

if __name__ == "__main__":
    # Fake example: tiny linearized dynamics to validate solve pipeline end-to-end.
    dt = 0.2
    nx, nu = 8, 2

    x = cs.SX.sym("x", nx)  # type: ignore[arg-type]
    u = cs.SX.sym("u", nu)  # type: ignore[arg-type]
    theta = cs.SX.sym("theta", 0)  # type: ignore[arg-type]
    disturbance = cs.SX.sym("disturbance", 0)  # type: ignore[arg-type]

    xdot = cs.SX.zeros(nx, 1)  # type: ignore[arg-type]
    # Heading index = 5, surge index = 6, sway index = 7
    xdot[5] = u[0]
    xdot[6] = u[0]
    xdot[7] = u[1]
    x_next = x + dt * xdot
    f_dyn = cs.Function("f_dyn", [x, u, theta, disturbance], [x_next])

    controller = NMPCTracker(
        horizon=15,
        dynamics=f_dyn,
        nx=nx,
        nu=nu,
        ntheta=0,
        ndisturbance=0,
        u_lb=np.array([-1.0, -1.0]),
        u_ub=np.array([1.0, 1.0]),
        R=1e-2 * np.eye(nu),
    )

    states = np.zeros(nx)
    states_des = np.zeros(nx)
    states_des[5] = 0.7  # desired heading [rad]
    states_des[6] = 1.2  # desired surge [m/s]
    states_des[7] = 0.0  # desired sway [m/s]

    u0, info = controller(
        states_des,
        states,
        Current(0, 0),
        Wind(0, 0),
        [],
        [],
    )
    print("Solved NMPC demo")
    print("u0:", u0)
    print("cost:", info["cost"])

