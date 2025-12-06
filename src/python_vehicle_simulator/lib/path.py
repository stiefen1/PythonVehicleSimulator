from python_vehicle_simulator.visualizer.drawable import IDrawable
from typing import List, Tuple, Union
from matplotlib.axes import Axes
import numpy as np, math, shapely

class PWLPath(IDrawable):
    """
    Piece-Wise Linear Path
    """

    def __init__(
            self,
            waypoints:List[Tuple],
            *args,
            verbose_level=0,
            **kwargs
    ):
        super().__init__(*args, verbose_level=verbose_level, **kwargs)
        self.waypoints = np.array(waypoints)
        self.length = shapely.LineString(self.waypoints).length
        self.init_heading()
        self.prev_target_wpts = []

    def init_heading(self) -> None:
        self.heading = []
        for k in range(1, self.waypoints.shape[0]):
            dw = self.waypoints[k] - self.waypoints[k-1]
            self.heading.append(math.atan2(dw[1], dw[0]))

    def closest_point(self, north:float, east:float) -> Tuple[float, float]:
        """
        closest point from the path that belongs to it
        """
        linestring = shapely.LineString(self.waypoints)
        point = shapely.Point(north, east)
        distance_along = linestring.project(point) # distance along the path from starting point
        closest_point = linestring.interpolate(distance_along)
        return float(closest_point.y), float(closest_point.x)
    
    @staticmethod
    def sample(d_tot:float, max_turn_deg:float, seg_len_range:Tuple[float, float], start:Tuple[float, float]=(0.0, 0.0), initial_angle:float=0.0, N:int=1, seed=None) -> Union["PWLPath", List["PWLPath"]]:
        """
        Returns a single (N=1) or list (N>1) of randomly generated piece-wise linear path, starting from start and oriented with initial_angle

        Each new segment is generated such that the turning angle is within [-max_turn_deg, max_turn_deg] and its length is in seg_len_range.

        The total length of all path generated with this function is equal to d_tot in order to compare the duration it took the vessel for reaching the end.

        Created for use in gymnasium environments.
        """
        np.random.seed(seed=seed)
        start = np.array(start)
        paths = []
        for _ in range(N):
            distance = 0
            angle = initial_angle*np.pi/180
            prev_wpt = start.copy()
            wpts = [tuple(start.tolist())]
            while distance < d_tot:
                length = float(np.random.uniform(*seg_len_range))

                # Make the total distance constant to compare Time of Arrival
                if distance + length > d_tot:
                    length = d_tot - distance

                wpt = prev_wpt + length * np.array([np.cos(angle), np.sin(angle)])

                wpts.append(tuple(wpt.tolist()))
                distance += length
                angle += float(np.pi*np.random.uniform(-max_turn_deg, max_turn_deg)/180)
                prev_wpt = wpt.copy()

            paths.append(PWLPath(wpts))
        return paths if N>1 else paths[0]
    
    def progression(self, north:float, east:float) -> float:
        return shapely.LineString(self.waypoints).project(shapely.Point(north, east)) / self.length

    def get_target_wpts_from(self, north:float, east:float, dp:float, N:int, final_heading:float=0.0) -> List[Tuple[float, float, float]]:
        """
        Returns a set of N waypoints (north, east) along the path, separated by a distance dp
        if heading is True, a third dimension is added with desired heading values.

        Projection of the current position is included to simplify MPC implementation.

        dp can be computed using the desired speed and sampling time, for instance.
        """
        linestring = shapely.LineString(self.waypoints)
        point = shapely.Point(north, east)
        initial_distance_along = linestring.project(point)
        target_wpts = []
        heading = None
        for n in range(0, N):
            p_n = linestring.interpolate(initial_distance_along + n * dp)
            p_next = linestring.interpolate(initial_distance_along + (n+1) * dp)
            if initial_distance_along + n * dp >= self.length:
                heading = final_heading
            else:
                heading = math.atan2(p_next.y-p_n.y, p_next.x-p_n.x)
            target_wpts.append((p_n.x, p_n.y, heading))
        self.prev_target_wpts = target_wpts
        return target_wpts        

    def __plot__(self, ax:Axes, *args, c='black', verbose:int=0, **kwargs) -> Axes:
        ax.plot(self.waypoints[:, 1], self.waypoints[:, 0], '--', *args, c=c, **kwargs)
        # for wpt in self.prev_target_wpts:
        #     ax.scatter(wpt[1], wpt[0], c='red')
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        ax.scatter(self.waypoints[:, 1], self.waypoints[:, 0], *args, **kwargs)
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax
    
def test() -> None:
    import matplotlib.pyplot as plt
    path = PWLPath(
        [
            (0, 0),
            (1, -0.5),
            (2.5, 1),
            (3, 2.5),
            (3, 4),
            (4, 5)
        ]
    )
    p = (3, 3)
    ax = path.plot()
    ax.scatter(p[1], p[0], c='green')
    ax.scatter(*path.closest_point(*p), c='red')
    print(path.progression(*p))
    desired_wpts = path.get_target_wpts_from(*p, 0.2, 10, final_heading=math.pi)
    
    for wpt in desired_wpts:
        ax.scatter(wpt[1], wpt[0], c='blue')
    print(len(desired_wpts), desired_wpts)
    plt.show()

if __name__ == "__main__":
    test()