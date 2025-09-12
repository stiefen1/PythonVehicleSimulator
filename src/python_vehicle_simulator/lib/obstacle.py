from python_vehicle_simulator.visualizer.drawable import IDrawable
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt, numpy as np
import shapely
from python_vehicle_simulator.states.states import Eta
from python_vehicle_simulator.utils.math_fn import Rzyx

class Obstacle(IDrawable):
    def __init__(
            self,
            geometry:List[Tuple],
            *args,
            verbose_level:int=0,
            **kwargs
    ):
        super().__init__(verbose_level=verbose_level)
        # Check if shape is closed (i.e. last point is equal to first point)
        if geometry[0][0] != geometry[-1][0] or geometry[0][1] != geometry[-1][1]:
            geometry.append(geometry[0]) # close the shape
        self.geometry = np.array(geometry).T

    def __plot__(self, ax, *args, **kwargs):
        """
        x = East
        y = North
        z = -depth
        """
        if isinstance(ax, Axes3D):
            ax.plot3D(self.geometry[1, :], self.geometry[0, :], -self.geometry[2, :], *args, **kwargs)
        else:
            ax.plot(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax

    def __scatter__(self, ax, *args, **kwargs):
        if isinstance(ax, Axes3D):
            ax.scatter3D(self.geometry[1, :], self.geometry[0, :], -self.geometry[2, :], *args, **kwargs)
        else:
            ax.scatter(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax

    def __fill__(self, ax, *args, **kwargs):
        if isinstance(ax, Axes3D):
            verts = [list(zip(self.geometry[1, :], self.geometry[0, :], -self.geometry[2, :]))]
            poly = Poly3DCollection(verts, *args, **kwargs)
            ax.add_collection3d(poly)
        else:
            ax.fill(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax
    
    def signed_distance(self, north:float, east:float) -> float:
        """Returns signed distance w.r.t obstacle. If point is inside the obstacle, distance is negative."""
        linestring = shapely.LineString(self.geometry.T)
        polygon = shapely.Polygon(self.geometry.T)
        point = shapely.Point(north, east)
        d = shapely.distance(linestring, point)
        if shapely.contains(polygon, point):
            return -d
        return d
    
    def distance(self, north:float, east:float) -> float:
        """Returns distance w.r.t obstacle. Returns 0 if point is inside the obstacle."""
        return shapely.distance(shapely.Polygon(self.geometry.T), shapely.Point(north, east))
    
    def get_geometry_in_frame(self, eta:np.ndarray) -> np.ndarray:
        """
        get geometry in a frame specified by nedrpy
        """
        return Rzyx(*eta[3:6])[0:2, 0:2].T @ (self.geometry - eta[0:2, None])
    
    @property
    def centroid(self) -> np.ndarray:
        return np.array(shapely.Polygon(self.geometry.T).centroid.xy)


def test() -> None:
    obs1 = Obstacle([(-2, 0), (1, 1), (2, 0)])
    print(obs1.distance(1, 0.5))
    ax = obs1.plot()
    ax.set_aspect('equal')
    plt.show()

def test_geometry_in_vessel_frame() -> None:
    from python_vehicle_simulator.vehicles.vessel import TestVessel, TestVesselParams
    from python_vehicle_simulator.states.states import Eta, Nu
    tv = TestVessel(TestVesselParams(), None, Eta(5, 5, yaw=1), Nu())
    obs1 = Obstacle([(-10, -5), (-9, -4), (-8, -5)])
    geom = obs1.get_geometry_in_frame(tv.eta)
    ax = obs1.plot()
    tv.plot(ax=ax)
    ax.set_aspect('equal')
    plt.show()

    tv.eta.zeros_inplace()
    ax = tv.plot()
    Obstacle(geom.T.tolist()).plot(ax=ax)
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    test()
    test_geometry_in_vessel_frame()

    