from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable, Dict, Optional
import numpy as np
from math import cos, sin
from python_vehicle_simulator.lib.noise import INoise, NoNoise
# from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
import shapely

class ISensor(ABC):
    def __init__(
            self,
            *args,
            noise: Optional[INoise] = None,
            **kwargs
    ):
        self.noise = noise or NoNoise()
        self.last_measurement = None
        self.last_info = None
        self.n_output = 0
        self.min = np.array([])
        self.max = np.array([])

    def __call__(self, *args, **kwargs) -> Tuple[Any, Any]:
        self.last_measurement, self.last_info = self.__get__(*args, **kwargs)
        return self.noise(self.last_measurement), self.last_info

    @abstractmethod
    def __get__(self, *args, **kwargs) -> Tuple[Any, Any]:
        """
        Noiseless measurement
        """
        pass

class GNSS(ISensor):
    def __init__(
            self,
            noise:INoise=None
    ):
        super().__init__(noise=noise)
        self.n_output = 2 # north, east

    def __get__(self, eta:np.ndarray) -> Tuple[np.ndarray, Dict]:
        return eta
    
class TOFArray(ISensor):
    def __init__(
        self,
        angles:Iterable,
        range:float,
        noise:INoise=None,
    ):
        super().__init__(noise=noise)
        self.angles = angles
        self.range = range
        rays = []
        for angle in angles:
            rays.append(shapely.LineString([(0, 0), (range*np.cos(angle), range*np.sin(angle))])) # X, Y coordinates in body frame
        self.rays = tuple(rays)
        self.n_output = len(self.rays)
        self.min = np.array(self.n_output*[0.0])
        self.max = np.array(self.n_output*[self.range])
        self.last_measurement = self.max.copy()

    def __get__(self, obstacles_body_frame:Iterable[Obstacle]) -> Tuple[np.ndarray, Dict]:
        """
        obstacles_body_frame: All obstacles at the current timestep (target vessels + static obstacles)
        """
        distances = self.n_output*[self.range]
        points = []
        for k in range(self.n_output):
            angle_k = self.angles[k]
            points.append((self.range*cos(angle_k), self.range*sin(angle_k)))

        for i, ray in enumerate(self.rays):
            for obstacle in obstacles_body_frame:
                obstacle_bounds = shapely.LineString(obstacle.geometry.T)
                point:shapely.LineString = shapely.intersection(obstacle_bounds, ray)
                if shapely.Polygon(obstacle_bounds).contains(shapely.Point(0, 0)):
                    distances[i] = 0
                    points[i] = (0.0, 0.0)
                    break

                # ray does intersect obstacle
                if not(point.is_empty):
                    if type(point) == shapely.MultiPoint:
                        
                        for subpoint in point.geoms:
                            dist = np.linalg.norm(subpoint.xy)
                            if dist <= distances[i]: # Keep the closest distance
                                distances[i] = dist
                                points[i] = (float(subpoint.x), float(subpoint.y))
                    else:
                        dist = np.linalg.norm(point.xy)
                        if dist <= distances[i]: # Keep the closest distance
                            distances[i] = dist
                            points[i] = (float(point.x), float(point.y))

        return np.array(distances), {"points": np.array(points)}
      
                
def test() -> None:
    import matplotlib.pyplot as plt
    from python_vehicle_simulator.vehicles.vessel import TestVessel

    # tv1 = TestVessel(eta=)

if __name__=="__main__":
    test()
    