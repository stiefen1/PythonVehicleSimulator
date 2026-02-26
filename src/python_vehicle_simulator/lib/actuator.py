import numpy as np
from abc import abstractmethod
from typing import Tuple, Dict, List, Optional
from python_vehicle_simulator.lib.weather import Current
from python_vehicle_simulator.visualizer.drawable import IDrawable
from matplotlib.axes import Axes
from python_vehicle_simulator.utils.math_fn import Rzyx

"""
^ surge (x)
|
|
O -----> sway (y)

"""

THRUSTER_LENGTH = 0.2
THRUSTER_WIDTH = 0.1
THRUSTER_GEOMETRY = lambda L, W: np.array([
    (L/2, W/2),
    (-L/2, W/2),
    (-L/2, -W/2),
    (L/2, -W/2),
    (L/2, W/2)
])

ROTATION_MATRIX = lambda a: np.array([
    [np.cos(a), -np.sin(a)],
    [np.sin(a), np.cos(a)]
])


class IActuator(IDrawable):
    def __init__(
            self,
            xy:Tuple,               # surge, sway in body frame
            orientation:float,      # clockwise angle (rad) w.r.t surge 
            u_0:Tuple,              # initial input value
            u_min:Tuple,            # min input value
            u_max:Tuple,            # max input value
            *args,
            time_const:Optional[Tuple]=None,  # Time constant
            f_min:Optional[Tuple]=None,       # min force
            f_max:Optional[Tuple]=None,       # max force
            faults:Optional[List[Dict]]=None,
            **kwargs

    ):
        super().__init__(verbose_level=1)
        self.xy = np.array(xy)
        self.orientation = orientation
        self.u_0 = np.array(u_0) # initial command
        self.u_min = np.array(u_min)
        self.u_max = np.array(u_max)
        self.dim = len(u_0)
        self.time_const = np.array(time_const) if time_const is not None else np.array(self.dim * [float('inf')])
        self.u_actual_prev = np.array(u_0)
        self.u_prev = np.array(u_0)
        self.f_min = np.array(f_min)
        self.f_max = np.array(f_max)
        self.faults = faults or []
        self.prev = {'tau': None, 'info':{'u_actual': None, 'u': None}}
        self.t = 0

    def dynamics(self, u:np.ndarray, nu:np.ndarray, current:Current, dt:float, *args, **kwargs) -> np.ndarray:
        """
            Wrapper for __dynamics__. Add saturation to actuator input commands
        """
        self.u_prev = u.copy()
        u_dot = (u-self.u_actual_prev) / self.time_const
        u = np.clip(self.u_actual_prev + u_dot * dt, self.u_min, self.u_max) # clip input within min/max bounds
        self.prev = {
            'info':{
                'u_actual': self.u_actual_prev,
                'u': self.u_prev
            }
        } # self.prev must be available when calling __dynamics__
        self.tau_prev = self.__dynamics__(u, nu, current, *args, **kwargs)
        self.prev.update({'tau': self.tau_prev})
        self.u_actual_prev = u.copy()
        self.t += dt
        return self.tau_prev

    @abstractmethod
    def __dynamics__(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        """
            Input:      u (np.ndarray) - For example propeller speed
            Output:     f (np.ndarray) - Generalized force (fx, fy, fz, Mx, My, Mz)
        """
        return np.zeros((6,))  

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax 
    
    def reset(self, random:bool=False, seed=None):
        if random:
            np.random.seed(seed=seed)
            self.u_actual_prev = np.random.uniform(self.u_min, self.u_max)
        else:
            self.u_actual_prev = self.u_0.copy()
        
        # Enforce initial steady state 
        self.u_prev = self.u_actual_prev.copy()

    @property
    def info(self) -> Dict:
        return {
            'u_actual': self.u_actual_prev
        }

class Thruster(IActuator):
    def __init__(
            self,
            xy:Tuple,
            k_pos:float,    # speed-thrust quad. mapping > 0 (n>=0)
            k_neg:float,    # speed-thrust quad. mapping > 0 (n<0)
            n_min:float,    # Min propeller speed (can be negative)
            n_max:float,    # Max propeller speed
            *args,
            n_0:float=0.0,  # initial propeller speed (rad/s)
            T_n:float=0.0,  # Time constant
            f_max:float=float('inf'),
            f_min:float=-float('inf'),
            orientation:float=0.0,
            **kwargs
    ):
        super().__init__(xy=xy, orientation=orientation, u_0=(n_0,), u_min=(n_min,), u_max=(n_max,), *args, time_const=(T_n,), f_min=(f_min,), f_max=(f_max,), **kwargs)
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.envelope = THRUSTER_GEOMETRY(THRUSTER_LENGTH, THRUSTER_WIDTH)

    def __dynamics__(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        self.thrust = np.clip(self.k_pos * u[0]**2 if u[0]>=0 else -self.k_neg * u[0]**2, self.f_min, self.f_max)
        return np.array([1, 0, 0, 0, 0, self.xy[1]]) * self.thrust
    
    def __plot__(self, ax:Axes, eta:np.ndarray, *args, verbose:int=0, **kwargs) -> Axes:
        envelope = (ROTATION_MATRIX(self.orientation) @ self.envelope.T) + self.xy[:, None]
        envelope_in_ned_frame = Rzyx(*eta[3:6].tolist())[0:2, 0:2] @ envelope + eta[0:2, None]
        ax.plot(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :], *args, **kwargs)
        return ax

class AzimuthThruster(IActuator):
    def __init__(
            self,
            xy:Tuple,
            k_pos:float,    # speed-thrust quad. mapping > 0 (n>=0)
            k_neg:float,    # speed-thrust quad. mapping > 0 (n<0)
            n_min:float,    # Min propeller speed (can be negative)
            n_max:float,    # Max propeller speed
            a_min:float,
            a_max:float,
            *args,
            alpha_0:float=0.0,
            n_0:float=0.0,  # initial propeller speed (rad/s)
            T_a:float=1.0,  # Time constant
            T_n:float=1.0,
            f_max:float=float('inf'),
            f_min:float=-float('inf'),
            orientation:float=0.0,
            faults:List[Dict]=None,
            length:float=THRUSTER_LENGTH,
            width:float=THRUSTER_WIDTH,
            **kwargs
    ):
        super().__init__(xy=xy, orientation=orientation, u_0=(alpha_0, n_0), u_min=(a_min, n_min), u_max=(a_max, n_max), *args, time_const=(T_a, T_n), f_min=(f_min,), f_max=(f_max,), faults=faults, **kwargs)
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.efficiency = 1.0
        self.prev['info'].update({'efficiency': self.efficiency})
        self.envelope = THRUSTER_GEOMETRY(length, width)

    def apply_faults(self) -> None:
        for fault in self.faults:
            match fault['type']:
                case 'loss-of-efficiency':
                    if self.t >= fault['t0']:
                        self.efficiency = fault['efficiency']
                case _:
                    print("Fault type is invalid")
        self.prev['info']['efficiency'] = self.efficiency

    def __dynamics__(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        """
        u: azimuth, speed
        """
        self.apply_faults() # Apply fault if it must occur
        self.thrust = self.efficiency * np.clip(self.k_pos * u[1]**2 if u[1]>=0 else -self.k_neg * u[1]**2, self.f_min, self.f_max)
        
        return np.array([
                np.cos(self.orientation + u[0]),
                np.sin(self.orientation + u[0]),
                0,
                0,
                0,
                self.xy[0] * np.sin(self.orientation + u[0]) - self.xy[1] * np.cos(self.orientation + u[0])
            ]) * self.thrust
            

    def __plot__(self, ax:Axes, eta:np.ndarray, *args, verbose:int=0, **kwargs) -> Axes:
        envelope = (ROTATION_MATRIX(self.orientation + self.u_actual_prev[0]) @ self.envelope.T) + self.xy[:, None]
        envelope_in_ned_frame = Rzyx(*eta[3:6].tolist())[0:2, 0:2] @ envelope + eta[0:2, None]

        if self.efficiency < 1.0:
            if 'c' in kwargs.keys():
                kwargs['c'] = 'red'
            else:
                kwargs.update({'c': 'red'})
        ax.plot(envelope_in_ned_frame[1, :], envelope_in_ned_frame[0, :], *args, **kwargs)
        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    th = Thruster((1, -0.5), 10, 10, 0, 10, orientation=0.3)
    ax = th.plot(eta=np.array([10, 0, 0, 0, 0, -0.3]), verbose=1)
    ax.set_aspect('equal')
    plt.show()
