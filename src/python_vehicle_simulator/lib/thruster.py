import casadi as cs, numpy, numpy.typing as npt
from typing import Optional, Tuple
from matplotlib.axes import Axes
from python_vehicle_simulator.lib.dynamics import IDynamics
from python_vehicle_simulator.lib.actuator import IActuator
from python_vehicle_simulator.utils.math_fn import Rzyx


THRUSTER_LENGTH_DEFAULT = 0.2
THRUSTER_WIDTH_DEFAULT = 0.1
THRUSTER_GEOMETRY = lambda L, W: numpy.array([
    (L/2, W/2),
    (-L/2, W/2),
    (-L/2, -W/2),
    (L/2, -W/2),
    (L/2, W/2)
])

ROTATION_MATRIX = lambda a: numpy.array([
    [numpy.cos(a), -numpy.sin(a)],
    [numpy.sin(a), numpy.cos(a)]
])

class ThrusterDynamics(IDynamics):
    def __init__(
            self,
            speed_min: float,
            speed_max: float,
            dt: float,
            *args,
            time_constant: Optional[float] = None,
            **kwargs
    ):
        """
        First-order thruster dynamics model.
        
        speed_min:      Minimum thruster speed
        speed_max:      Maximum thruster speed
        dt:             Sampling time
        time_constant:  Thruster time constant (optional)
        """
        self.speed_min = speed_min
        self.speed_max = speed_max

        # time_constant = max(dt, time_constant)
        if time_constant is not None:
            assert time_constant >= dt, f"Sampling time must be smaller or equal to time constant for numerical stability. Got dt = {dt:.2f} > {time_constant:.2f}"
            self.time_constant = time_constant
        else:
            self.time_constant = dt

        super().__init__(1, 1, 1, 0, dt, *args, **kwargs)

    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: Optional[cs.SX], *args, **kwargs) -> cs.SX:
        """
        x:              actual speed, thrust        (nx,)
        u:              feasible speed setpoint     (nu,)
        theta:          effectiveness               (np,)
        disturbance:    None                        (nd,)
        """
        return (u - x) / self.time_constant


class Thruster(IActuator):
    def __init__(
            self,
            xy: Tuple,
            orientation_deg: float,
            thrust_coeff: float,
            speed_min: float,
            speed_max: float,
            dt: float,
            initial_speed: float = 0,
            time_constant: Optional[float] = None,
            effectiveness: float = 1.0,
            length: float = THRUSTER_LENGTH_DEFAULT,
            width: float = THRUSTER_WIDTH_DEFAULT
    ):
        """
        Thruster actuator with position, orientation, and dynamics.
        
        xy:                 Position in body frame           (2,)
        orientation_deg:    Thruster orientation in degrees
        thrust_coeff:       Thrust coefficient
        speed_min:          Minimum thruster speed
        speed_max:          Maximum thruster speed
        dt:                 Sampling time
        initial_speed:      Initial thruster speed
        time_constant:      Thruster time constant (optional)
        effectiveness:      Fault parameter (0-1)
        length:             Thruster length for visualization
        width:              Thruster width for visualization
        """
        super().__init__(
            ThrusterDynamics(speed_min, speed_max, dt, time_constant=time_constant),
            (initial_speed,),
            (speed_min,),
            (speed_max,)
        )
        self.xy = numpy.array(xy)
        self.orientation_deg = orientation_deg
        self.thrust_coeff = thrust_coeff
        self.effectiveness = effectiveness
        self.envelope = THRUSTER_GEOMETRY(length, width)

        self._orientation_rad = numpy.deg2rad(orientation_deg)
        self._cos_orientation = numpy.cos(self._orientation_rad)
        self._sin_orientation = numpy.sin(self._orientation_rad)
        self._torque_lever_arm = self.xy[0] * self._sin_orientation - self.xy[1] * self._cos_orientation

    def __dynamics__(self, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Thruster dynamics: updates speed and computes force/torque.
        
        u:          Speed setpoint                  (1,)
        theta:      Effectiveness parameter         (1,)
        disturbance: Not used                       (0,)
        
        Returns:
            npt.NDArray: Force and torque [fx, fy, tau] (3,)
        """
        self.x = self.dynamics.fd(self.x, u, theta, disturbance).flatten()
        thrust = self.thrust_coeff * self.effectiveness * self.x * self.x
        return thrust * numpy.array([
            self._cos_orientation,
            self._sin_orientation,
            self._torque_lever_arm]).flatten()
    
    def __plot__(self, ax:Axes, eta:npt.NDArray, *args, verbose:int=0, **kwargs) -> Axes:
        """
        Plot thruster on given axes in NED frame.
        
        ax:         Matplotlib axes object
        eta:        Vehicle pose [x, y, z, phi, theta, psi]  (6,)
        verbose:    Verbosity level
        
        Returns:
            Axes: Updated axes object
        """
        envelope = (ROTATION_MATRIX(self._orientation_rad + self.x[0]) @ self.envelope.T) + self.xy[:, None]
        envelope_in_ned_frame = Rzyx(*eta[3:6].tolist())[0:2, 0:2] @ envelope + eta[0:2, None]

        if self.effectiveness < 1.0:
            if 'c' in kwargs.keys():
                kwargs['c'] = 'red'
            else:
                kwargs.update({'c': 'red'})
        ax.plot(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :], *args, **kwargs)
        return ax
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dt = 0.01
    a1 = Thruster(
        xy=(10, 0),
        orientation_deg=10,
        thrust_coeff=10,
        speed_min=0,
        speed_max=80,
        dt=dt,
        initial_speed=0,
        time_constant=1,
        effectiveness=1
    )

    fig, ax = plt.subplots()
    fs = []
    ts = numpy.linspace(0, 10, int(10//dt) + 1)
    xs = []
    for t in ts:
        xs.append(a1.x)
        f = a1.step((120,))
        if t > 5:
            a1.effectiveness = 0.5 
        fs.append(f)

    fs = numpy.array(fs)
    xy = numpy.array(xs)
    plt.plot(ts, fs)
    plt.legend(['fx', 'fy', 'tau'])
    plt.show()