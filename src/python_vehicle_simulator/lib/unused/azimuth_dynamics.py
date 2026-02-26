import casadi as cs, numpy, numpy.typing as npt
from typing import Optional, Tuple
from matplotlib.axes import Axes
from python_vehicle_simulator.lib.dynamics import IDynamics
from python_vehicle_simulator.lib.actuator import IActuator
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.thruster import THRUSTER_LENGTH_DEFAULT, THRUSTER_WIDTH_DEFAULT, THRUSTER_GEOMETRY, ROTATION_MATRIX

class AzimuthThruster(IDynamics):
    def __init__(
            self,
            speed_min: float,
            speed_max: float,
            azimuth_min: float,
            azimuth_max: float,
            dt: float,
            *args,
            speed_time_constant: Optional[float] = None,
            azimuth_time_constant: Optional[float] = None,
            **kwargs
    ):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.azimuth_min = azimuth_min
        self.azimuth_max = azimuth_max

        # speed_time_constant = max(dt, speed_time_constant)
        if speed_time_constant is not None:
            assert speed_time_constant >= dt, f"Sampling time must be smaller or equal to speed time constant for numerical stability. Got dt = {dt:.2f} > {speed_time_constant:.2f}"
            self.speed_time_constant = speed_time_constant
        else:
            self.speed_time_constant = dt
       
        # azimuth_time_constant = max(dt, azimuth_time_constant)
        if azimuth_time_constant is not None:
            assert azimuth_time_constant >= dt, f"Sampling time must be smaller or equal to azimuth time constant for numerical stability. Got dt = {dt:.2f} > {azimuth_time_constant:.2f}"
            self.azimuth_time_constant = azimuth_time_constant
        else:
            self.azimuth_time_constant = dt

        super().__init__(2, 2, 2, 0, dt, *args, **kwargs)

    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: Optional[cs.SX], *args, **kwargs) -> cs.SX:
        """
        x:              actual speed, thrust        (nx,)
        u:              feasible speed setpoint     (nu,)
        theta:          effectiveness               (np,)
        disturbance:    None                        (nd,)
        """
        return (u - x) / self.time_constant