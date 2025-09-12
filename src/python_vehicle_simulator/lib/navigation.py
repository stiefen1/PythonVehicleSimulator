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


class INavigation(IDrawable, ABC):
    def __init__(
            self,
            sensors:Dict[str, ISensor],
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)
        self.sensors = sensors
        self.last_observation = None
        self.last_info = None

    def __call__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        self.last_observation, self.last_info = self.__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)
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

    def __plot__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

class Navigation(INavigation):
    def __init__(
            self, 
            *args,
            **kwargs
    ):
        super().__init__({}, *args, **kwargs)

    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[Dict, Dict]:
        return super().__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs), {}
    
    def reset(self):
        pass

class NavigationTOF(INavigation):
    def __init__(
            self, 
            tof_params:dict={"range":30.0, "angles":np.linspace(-45*DEG2RAD, 45*DEG2RAD, 6), "noise":NoNoise()},
            *args,
            **kwargs
    ):
        super().__init__(
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

    def __plot__(self, ax:Axes, *args, **kwargs) -> Axes:
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