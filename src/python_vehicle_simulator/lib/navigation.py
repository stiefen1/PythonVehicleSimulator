from abc import ABC, abstractmethod
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.sensor import ISensor, TOFArray
from python_vehicle_simulator.lib.noise import NoNoise
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
# from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
import numpy as np
from typing import List, Dict, Tuple
from copy import deepcopy
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.filter import LowPass
from python_vehicle_simulator.lib.kalman import EKFRevolt3

# According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

#         heading +- 0.2°
#         position +- 1cm
#         u, v +- 0.05 m/s
#         r not specified, assuming it is very low according to graph. let's say r +- 0.05 deg/s as well
Q_REVOLT = np.diag([0.3**2, 0.3**2, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 15*np.pi/180/3600])
R_REVOLT = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2])


class INavigation(IDrawable, ABC):
    def __init__(
            self,
            states:np.ndarray,
            sensors:Dict[str, ISensor],
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)
        self.sensors = sensors
        self.prev = {"eta": states[0:6].copy(), "nu": states[6:12].copy(), "states": states.copy(), "current": None, "wind": None, "obstacles": None, "target_vessels": None, 'info': None}
        self.last_observation = None
        self.last_info = None

    def __call__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        self.last_observation, self.last_info = self.__get__(states, current, wind, obstacles, target_vessels, *args, **kwargs)
        self.prev = self.last_observation
        self.prev.update({'info':self.last_info})
        return self.last_observation, self.last_info

    @abstractmethod
    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        return {
            "states": states,
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "target_vessels": target_vessels
            }, {}
    
    @abstractmethod
    def reset(self):
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

class Navigation(INavigation):
    def __init__(
            self,
            states:np.ndarray, 
            *args,
            **kwargs
    ):
        super().__init__(states, {}, *args, **kwargs)

    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        return super().__get__(states, current, wind, obstacles, target_vessels, *args, **kwargs)
    
    def reset(self):
        pass
    
class NavigationWithNoise(INavigation):
    def __init__(
            self, 
            states:np.ndarray,
            *args,
            offset=None,
            std=None,
            **kwargs
    ):
        super().__init__(states, {}, *args, **kwargs)
        if offset is None:
            self.offset = {
                'eta': (np.random.random(size=(6,))-0.5) * 2 * 0.0, # [-1, 1] * 0.5
                'nu': (np.random.random(size=(6,))-0.5) * 2 * 0.0 # [-1, 1] * 0.1
            }

        if std is None:
            self.std = {
                'eta' : np.array([R_REVOLT[0, 0], R_REVOLT[1, 1], 0, 0, 0, R_REVOLT[2, 2]]),
                'nu' : np.array([R_REVOLT[3, 3], R_REVOLT[4, 4], 0, 0, 0, R_REVOLT[5, 5]])
            }

        self.n_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=eta[0])
        self.e_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=eta[1])
        self.y_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=eta[5])
        self.u_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=nu[0])
        self.v_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=nu[1])
        self.r_filter = LowPass(cutoff=1e-1, sampling_frequency=10e0, order=5, init=nu[5])
        
    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        
        """
        eta, nu = states[0:6], states[6:12]
        eta = eta + np.random.normal(loc=self.offset['eta'], scale=self.std['eta'])
        nu = nu + np.random.normal(loc=self.offset['nu'], scale=self.std['nu'])
        eta = np.array([
            self.n_filter(eta[0]),
            self.e_filter(eta[1]),
            0., 0., 0.,
            self.y_filter(eta[5])
        ])
        nu = np.array([
            self.u_filter(nu[0]),
            self.v_filter(nu[1]),
            0., 0., 0.,
            self.r_filter(nu[5])
        ])

        observation = {
            "eta": eta,
            "nu": nu,
            "states": np.concatenate([eta, nu]),
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "target_vessels": target_vessels,
        }

        info = {}
        return observation, info
    
    def reset(self):
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        states = self.last_observation["states"]
        ax.scatter(states[1], states[0], c='purple')
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        states = self.last_observation["states"]
        ax.scatter(states[1], states[0], c='purple')
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

class NavigationRevolt3WithEKF(INavigation):
    """
        According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

        heading +- 0.2°
        position +- 1cm
        u, v +- 0.05 m/s
        r not specified, assuming it is very low according to graph. let's say r +- 0.05 deg/s as well
    """
    def __init__(
            self,
            states:np.ndarray,
            dt:float,
            *args,
            Q:np.ndarray = Q_REVOLT,
            R:np.ndarray = R_REVOLT,
            P0:np.ndarray = np.eye(6),
            offset=None,
            std=None,
            **kwargs
    ):
        super().__init__(states, {}, *args, **kwargs)
        if offset is None:
            self.offset = {
                'eta': (np.random.random(size=(6,))-0.5) * 2 * 0.0, # [-1, 1] * 0.5
                'nu': (np.random.random(size=(6,))-0.5) * 2 * 0.0 # [-1, 1] * 0.1
            }

        if std is None:
            self.std = {
                'eta' : np.array([R[0, 0], R[1, 1], 0, 0, 0, R[2, 2]]),
                'nu' : np.array([R[3, 3], R[4, 4], 0, 0, 0, R[5, 5]])
            }


        self.ekf = EKFRevolt3(Q=Q, R=R, x0=np.array([states[0], states[1], states[5], states[6], states[7], states[11]]), P0=P0, dt=dt)
        
    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, tau_actuators:np.ndarray=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        
        """
        eta = states[0:6].copy()
        nu = states[6:12].copy()

        if tau_actuators is None:
            tau_actuators = np.array([0., 0., 0., 0., 0., 0.])

        # Measurement
        eta = eta + np.random.normal(loc=self.offset['eta'], scale=self.std['eta'])
        nu = nu + np.random.normal(loc=self.offset['nu'], scale=self.std['nu'])
        
        # Filtering
        x = self.ekf(np.array([tau_actuators[0], tau_actuators[1], tau_actuators[5]]), np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]))

        eta_estimated = np.array([x[0], x[1], 0, 0, 0, x[2]]) 
        nu_estimated = np.array([x[3], x[4], 0, 0, 0, x[5]])
        actuator_estimated = np.array(6*[None])

        observation = {
            "eta": eta_estimated,
            "nu": nu_estimated,
            "states": np.concatenate([eta_estimated, nu_estimated, actuator_estimated]),
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "target_vessels": target_vessels,
        }

        info = {}
        return observation, info
    
    def reset(self):
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        ax.scatter(eta[1], eta[0], c='purple')
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        ax.scatter(eta[1], eta[0], c='purple')
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax