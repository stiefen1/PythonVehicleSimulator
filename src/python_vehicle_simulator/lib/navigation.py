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
            eta:np.ndarray,
            nu:np.ndarray,
            sensors:Dict[str, ISensor],
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)
        self.sensors = sensors
        self.prev = {"eta": eta.copy(), "nu": nu.copy(), "current": None, "wind": None, "obstacles": None, "target_vessels": None, 'info': None}
        self.last_observation = None
        self.last_info = None

    def __call__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        self.last_observation, self.last_info = self.__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)
        self.prev = self.last_observation
        self.prev.update({'info':self.last_info})
        return self.last_observation, self.last_info

    @abstractmethod
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        return {
            "eta": eta,
            "nu": nu,
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
            eta:np.ndarray,
            nu:np.ndarray, 
            *args,
            **kwargs
    ):
        super().__init__(eta, nu, {}, *args, **kwargs)

    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        return super().__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)
    
    def reset(self):
        pass

class NavigationTOF(INavigation):
    def __init__(
            self, 
            eta:np.ndarray,
            nu:np.ndarray, 
            tof_params:dict={"range":30.0, "angles":np.linspace(-45*DEG2RAD, 45*DEG2RAD, 6), "noise":NoNoise()},
            *args,
            **kwargs
    ):
        super().__init__(
            eta, 
            nu,
            {
                'tof': TOFArray(**tof_params)
            },
            *args,
            **kwargs
        )
        
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        actual_obstacles_in_body_frame = []
        # print(eta, type(eta))
        for obs in obstacles:
            actual_obstacles_in_body_frame.append(Obstacle(obs.get_geometry_in_frame(eta).T.tolist()))
        for tv in target_vessels:
            actual_obstacles_in_body_frame.append(Obstacle(tv.get_geometry_in_frame(eta)[0:2, :].T.to_list()))
        tof_meas, tof_info = self.sensors['tof'](actual_obstacles_in_body_frame)
        observation = {
            "eta": eta,
            "nu": nu,
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "target_vessels": target_vessels,
            "tof": tof_meas
        }

        info = tof_info
        return observation, info
    
    def reset(self):
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        tof_points_in_body = self.last_info["points"]
        tof_in_ned = (Rzyx(*eta[3:6])[0:2,0:2] @ tof_points_in_body.T + eta[0:2, None]).T
        for k in range(tof_in_ned.shape[0]):
            ax.plot([eta[1], tof_in_ned[k, 1]], [eta[0], tof_in_ned[k, 0]], c='red')
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        tof_points_in_body = self.last_info["points"]
        tof_in_ned = (Rzyx(*eta[3:6])[0:2,0:2] @ tof_points_in_body.T + eta[0:2, None]).T
        ax.scatter(tof_in_ned[:, 1], tof_in_ned[:, 0], c='red')
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax
    
class NavigationWithNoise(INavigation):
    def __init__(
            self, 
            eta:np.ndarray,
            nu:np.ndarray,
            *args,
            offset=None,
            std=None,
            **kwargs
    ):
        super().__init__(eta, nu, [], *args, **kwargs)
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
        
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        
        """
        
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
            eta:np.ndarray,
            nu:np.ndarray,
            dt:float,
            *args,
            Q:np.ndarray = Q_REVOLT,
            R:np.ndarray = R_REVOLT,
            P0:np.ndarray = np.eye(6),
            offset=None,
            std=None,
            **kwargs
    ):
        super().__init__(eta, nu, [], *args, **kwargs)
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
            # self.std = {
            #     'eta': (np.abs(np.random.random(size=(6,))-0.5)) * 2 * 0.5, # [-1, 1] * 0.5 
            #     'nu': (np.abs(np.random.random(size=(6,))-0.5)) * 2 * 0.1 # [-1, 1] * 0.1
            # }
            # self.std['eta'][5] = 0.05

        self.ekf = EKFRevolt3(Q=Q, R=R, x0=np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]), P0=P0, dt=dt)
        
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, tau_actuators:np.ndarray=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        
        """
        if tau_actuators is None:
            tau_actuators = np.array([0., 0., 0., 0., 0., 0.])

        # Measurement
        eta = eta + np.random.normal(loc=self.offset['eta'], scale=self.std['eta'])
        nu = nu + np.random.normal(loc=self.offset['nu'], scale=self.std['nu'])
        
        # Filtering
        x = self.ekf(np.array([tau_actuators[0], tau_actuators[1], tau_actuators[5]]), np.array([eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]))

        observation = {
            "eta": np.array([x[0], x[1], 0, 0, 0, x[2]]),
            "nu": np.array([x[3], x[4], 0, 0, 0, x[5]]),
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


def test_tof_navigation() -> None:
    from python_vehicle_simulator.vehicles.otter import Otter, OtterParameters
    from python_vehicle_simulator.lib.map import RandomMapGenerator
    import matplotlib.pyplot as plt
    from python_vehicle_simulator.utils.math_fn import Rzyx

    map_generator = RandomMapGenerator(
        (-30, 30),
        (-30, 30),
        (1, 10),
        min_dist=5
    )


    otter = Otter(
        OtterParameters(),
        0.02,
        eta=(5,-8,0,0,0,1),
        navigation=NavigationTOF()
    )

    obstacles, info = map_generator.get([(otter.eta.n, otter.eta.e)], min_density=0.2)


    measurements, info = otter.navigation(otter.eta.to_numpy(), otter.nu.to_numpy(), None, None, obstacles, [])
    # tof_points_in_body = info['points']
    # print("Main: ", tof_points_in_body.shape)
    # print(tof_points_in_body.shape)
    # print(otter.eta.rpy, Rzyx(*otter.eta.rpy)[0:2,0:2], otter.eta.to_numpy())
    # tof_in_ned = (Rzyx(*otter.eta.rpy)[0:2,0:2] @ tof_points_in_body.T + otter.eta.to_numpy()[0:2].reshape(2, 1)).T
    # tof_in_ned = (Rzyx(*eta[0:3])[0:2,0:2] @ tof_points_in_body.T + np.array(eta[0:2]).reshape(2, 1)).T

    # tof_in_ned = tof_points_in_body

    ax = otter.plot(c='blue', verbose=2)
    otter.navigation.scatter(ax=ax, verbose=2)
    for obs in obstacles:
        obs.plot(ax=ax, c='black')

    # ax.scatter(tof_in_ned[:, 1], tof_in_ned[:, 0], c='green')
    # otter.__scatter__(ax=ax,  c='red')
    ax.set_aspect('equal')

    plt.show()

if __name__=="__main__":
    test_tof_navigation()