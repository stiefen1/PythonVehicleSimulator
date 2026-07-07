from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle

import casadi as cs
import numpy as np
import numpy.typing as npt

DEFAULT_IPOPT_SOLVER_OPTS = {
    "error_on_fail": False,
    "expand": True,
    "print_time": False,
    "record_time": True,
    "ipopt.print_level": 0,
    "ipopt.max_iter": 200,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-4,
    "ipopt.mu_init": 1e-3,
    "ipopt.warm_start_init_point": "yes",
}

class NMPCBase(IControl):
    """Simple NMPC template: build solver once, only update parameters online."""

    vessel_params: Optional[Any] = None

    def __init__(
        self,
        horizon: int,
        dynamics: cs.Function,
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
        solver_opts: Optional[Dict[str, Any]] = None,
        u_0: Optional[npt.NDArray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.horizon = int(horizon)
        self.dynamics = dynamics
        self.nx = int(nx)
        self.nu = int(nu)
        self.ntheta = int(ntheta)
        self.ndisturbance = int(ndisturbance)

        self.u_lb = np.asarray(u_lb, dtype=float).reshape(-1)
        self.u_ub = np.asarray(u_ub, dtype=float).reshape(-1)
        if self.u_lb.shape[0] != self.nu or self.u_ub.shape[0] != self.nu:
            raise ValueError("u_lb/u_ub must have size nu")

        self.x_lb = np.asarray(x_lb, dtype=float).reshape(-1) if x_lb is not None else np.full(self.nx, -np.inf)
        self.x_ub = np.asarray(x_ub, dtype=float).reshape(-1) if x_ub is not None else np.full(self.nx, np.inf)
        if self.x_lb.shape[0] != self.nx or self.x_ub.shape[0] != self.nx:
            raise ValueError("x_lb/x_ub must have size nx")

        self._solver_opts = dict(DEFAULT_IPOPT_SOLVER_OPTS)
        if solver_opts is not None:
            self._solver_opts.update(solver_opts)

        self.prev_sol: Optional[Dict[str, np.ndarray]] = None

        u_init = np.asarray(u_0, dtype=float).reshape(-1) if u_0 is not None else np.zeros(self.nu)
        super().__init__(u_init, *args, seed=seed, **kwargs)
        self.init_nlp(solver, self._solver_opts)

    @abstractmethod
    def lagrange(self, xk: cs.SX, uk: cs.SX, x_ref_k: cs.SX, k: int) -> cs.SX:
        pass

    @abstractmethod
    def mayer(self, xN: cs.SX, x_ref_N: cs.SX) -> cs.SX:
        pass

    def init_nlp(self, solver: Literal["ipopt"], solver_opts: Dict[str, Any]) -> None:
        if solver != "ipopt":
            raise ValueError(f"Unsupported solver '{solver}'")

        n_ref = self.nx * (self.horizon + 1)
        n_runtime = self.ntheta + self.ndisturbance

        self.X = cs.SX.sym("X", self.nx, self.horizon + 1)  # type: ignore[arg-type]
        self.U = cs.SX.sym("U", self.nu, self.horizon)  # type: ignore[arg-type]
        self.P = cs.SX.sym("P", self.nx + n_ref + n_runtime)  # type: ignore[arg-type]

        x0 = self.P[0:self.nx]
        x_ref_all = cs.reshape(self.P[self.nx:self.nx + n_ref], self.nx, self.horizon + 1)
        theta_start = self.nx + n_ref
        theta = self.P[theta_start:theta_start + self.ntheta]
        disturbance = self.P[theta_start + self.ntheta:theta_start + self.ntheta + self.ndisturbance]

        g = [self.X[:, 0] - x0]
        for k in range(self.horizon):
            x_next = self.dynamics(self.X[:, k], self.U[:, k], theta, disturbance)
            g.append(self.X[:, k + 1] - x_next)
        self.g = cs.vertcat(*g)

        obj = cs.SX(0)
        for k in range(self.horizon):
            obj += self.lagrange(self.X[:, k], self.U[:, k], x_ref_all[:, k], k)
        obj += self.mayer(self.X[:, self.horizon], x_ref_all[:, self.horizon])
        self.obj = obj

        self.decision = cs.vertcat(cs.reshape(self.X, -1, 1), cs.reshape(self.U, -1, 1))
        n_xvars = self.nx * (self.horizon + 1)
        n_uvars = self.nu * self.horizon
        n_g = self.nx * (self.horizon + 1)

        self.lbx = np.concatenate([
            np.tile(self.x_lb, self.horizon + 1),
            np.tile(self.u_lb, self.horizon),
        ])
        self.ubx = np.concatenate([
            np.tile(self.x_ub, self.horizon + 1),
            np.tile(self.u_ub, self.horizon),
        ])
        self.lbg = np.zeros(n_g)
        self.ubg = np.zeros(n_g)

        n_dec = n_xvars + n_uvars
        self._x_guess_template = np.zeros(n_dec)

        nlp = {"x": self.decision, "f": self.obj, "g": self.g, "p": self.P}
        self.solver = cs.nlpsol("nmpc_solver", solver, nlp, solver_opts)

    def _build_reference_matrix(self, states_des: npt.NDArray) -> np.ndarray:
        ref = np.asarray(states_des, dtype=float)
        if ref.ndim == 1:
            if ref.shape[0] != self.nx:
                raise ValueError(f"states_des must have size {self.nx}")
            return np.repeat(ref.reshape(self.nx, 1), self.horizon + 1, axis=1)

        if ref.ndim == 2 and ref.shape == (self.nx, self.horizon + 1):
            return ref

        if ref.ndim == 2 and ref.shape == (self.horizon + 1, self.nx):
            return ref.T

        raise ValueError(
            f"states_des shape must be ({self.nx},) or ({self.nx}, {self.horizon + 1}) or ({self.horizon + 1}, {self.nx}), got {states_des.shape}"
        )

    def _build_initial_guess(self, x0: np.ndarray) -> np.ndarray:
        if self.prev_sol is None:
            x_guess = np.tile(x0, self.horizon + 1)
            u_guess = np.zeros(self.nu * self.horizon)
            return np.concatenate([x_guess, u_guess])

        x_prev = self.prev_sol["x"]
        u_prev = self.prev_sol["u"]

        x_shift = np.vstack([x_prev[1:, :], x_prev[-1:, :]])
        u_shift = np.vstack([u_prev[1:, :], u_prev[-1:, :]])
        return np.concatenate([x_shift.reshape(-1), u_shift.reshape(-1)])

    def __get__(
        self,
        states_des: npt.NDArray,
        states: npt.NDArray,
        current: Current,
        wind: Wind,
        obstacles: List[Obstacle],
        target_vessels: List,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        x0 = np.asarray(states, dtype=float).reshape(-1)
        if x0.shape[0] != self.nx:
            raise ValueError(f"states must have size {self.nx}")

        # compute disturbance, either based on vessel parameters or take it from **kwargs
        if self.vessel_params is not None: 
            # Wind perturbations
            uw = wind.u(states[5])
            vw = wind.v(states[5])

            u_rw = uw - states[6]
            v_rw = vw - states[7]

            gamma_w = wind.gamma_w(states[5])
            wind_rw2 = u_rw**2 + v_rw**2
            c_x = -self.vessel_params.cx * np.cos(gamma_w)
            c_y = self.vessel_params.cy * np.sin(gamma_w)
            c_n = self.vessel_params.cn * np.sin(2 * gamma_w)

            tau_coeff = 0.5 * wind.get_air_density() * wind_rw2
            tau_w = np.array([
                tau_coeff * c_x * self.vessel_params.proj_area_f,
                tau_coeff * c_y * self.vessel_params.proj_area_l,
                tau_coeff * c_n * self.vessel_params.proj_area_l * self.vessel_params.loa
            ]) 

            # Current perturbations
            uvr = np.take(states, [6, 7, 11])
            v_c = np.array([current.u(states[5]), current.v(states[5]), 0]) # current speed in ship frame
            tau_c_coriolis = self.vessel_params.CA(uvr) @ uvr - self.vessel_params.CA(uvr - v_c) @ (uvr - v_c) # cancel CA(nu) @ nu and add CA(nu_r) @ nu_r
            tau_c_damping = self.vessel_params.D @ v_c
            tau_c = tau_c_coriolis + tau_c_damping

            disturbance = tau_w + tau_c # Define it as a function of current, wind
        else:
            disturbance = np.asarray(kwargs.pop("disturbance", np.zeros(self.ndisturbance)), dtype=float).reshape(-1)

        theta = np.asarray(kwargs.pop("theta", np.ones(self.ntheta)), dtype=float).reshape(-1)
        if theta.shape[0] != self.ntheta:
            raise ValueError(f"theta must have size {self.ntheta}")
        if disturbance.shape[0] != self.ndisturbance:
            raise ValueError(f"disturbance must have size {self.ndisturbance}")

        ref = self._build_reference_matrix(states_des)
        p = np.concatenate([x0, ref.reshape(-1, order="F"), theta, disturbance])
        x_guess = self._build_initial_guess(x0)

        solver_in = {
            "x0": x_guess,
            "lbx": self.lbx,
            "ubx": self.ubx,
            "lbg": self.lbg,
            "ubg": self.ubg,
            "p": p,
        }

        if self.prev_sol is not None:
            if "lam_x" in self.prev_sol and "lam_g" in self.prev_sol:
                solver_in["lam_x0"] = self.prev_sol["lam_x"]
                solver_in["lam_g0"] = self.prev_sol["lam_g"]

        out = self.solver(**solver_in)

        decision = np.array(out["x"]).reshape(-1)
        n_xvars = self.nx * (self.horizon + 1)
        x_opt = decision[:n_xvars].reshape(self.horizon + 1, self.nx)
        u_opt = decision[n_xvars:].reshape(self.horizon, self.nu)

        self.prev_sol = {
            "x": x_opt,
            "u": u_opt,
            "lam_x": np.array(out["lam_x"]).reshape(-1),
            "lam_g": np.array(out["lam_g"]).reshape(-1),
        }

        u0 = u_opt[0].copy()
        info = {
            "cost": float(out["f"]),
            "x_pred": x_opt,
            "u_pred": u_opt,
            "stats": self.solver.stats(),
        }
        return u0, info

    def reset(self, initial_commands: npt.NDArray, seed: Optional[int] = None):
        self.prev_sol = None
        self.prev = {"u": np.asarray(initial_commands).reshape(-1), "info": None}

