from python_vehicle_simulator.visualizer.drawable import IDrawable
from typing import List, Tuple, Union, Literal
from matplotlib.axes import Axes
from shapely.ops import substring
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
            input_format:Literal['north-east', 'east-north']='north-east',
            flip: bool = False,
            **kwargs
    ):
        super().__init__(*args, verbose_level=verbose_level, **kwargs)
        if input_format == 'east-north':
            old_wpts = waypoints
            waypoints = []
            for wpt in old_wpts:
                waypoints.append((wpt[1], wpt[0]))

        self.waypoints = np.flip(waypoints, axis=0) if flip else np.array(waypoints)
        self.length = shapely.LineString(self.waypoints).length
        self.init_heading()
        self.prev_target_wpts = []

        waypoints_progression = []
        for wpt in self.waypoints:
            wpt_prog = self.progression(*wpt)
            waypoints_progression.append(wpt_prog)
        self.waypoints_progression = np.array(waypoints_progression)

    def get_current_waypoint(self, north: float, east: float) -> int:
        """
        Returns the index of the next waypoint along the path.
        """
        p = self.progression(north, east)
        return self.get_next_waypoint_index(p)
    
    def get_next_waypoint_index(self, progression: float) -> int:
        """
        Returns the index of the next waypoint given a progression value.
        """
        # Find the first waypoint that has progression greater than the current progression
        next_indices = np.where(self.waypoints_progression > progression)[0]
        
        if len(next_indices) == 0:
            # If no waypoint is ahead, return the last waypoint index
            return len(self.waypoints_progression) - 1
        else:
            return next_indices[0]

    def init_heading(self) -> None:
        self.heading = []
        for k in range(1, self.waypoints.shape[0]):
            dw = self.waypoints[k] - self.waypoints[k-1]
            self.heading.append(math.atan2(dw[1], dw[0]))

    def interpolate(self, distance: float, normalized: bool = False) -> Tuple[float, float]:
        point = shapely.LineString(self.waypoints).interpolate(distance=distance, normalized=normalized)
        return point.x, point.y

    def get_initial_pose(self, radians: bool = False) -> Tuple[float, float, float]:
        """
        Returns a 3d-tuple containing initial (north, east, heading) in NED frame (heading is clockwise positive)
        """
        linestring = shapely.LineString(self.waypoints)
        point = linestring.interpolate(0)
        next_point = linestring.interpolate(0.001)

        # compute heading
        heading = np.atan2((next_point.y - point.y), (next_point.x - point.x))

        return point.x, point.y, heading if radians else np.rad2deg(heading)

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
    
    def smooth(self, radius: float, n_arc_points: int = 20) -> "PWLPath":
        """
        Return a new PWLPath where each interior corner is replaced by a
        circular arc of (at most) the given radius, yielding a G1-continuous
        (tangent-continuous) path.

        For each interior waypoint P with predecessor A and successor B:
          1. The tangent distance d_t = radius / tan(γ/2) gives the distance
             along each segment from P to the arc tangent points T1, T2.
          2. The arc centre C lies along the angle bisector from P at
             distance radius / sin(γ/2), where γ is the interior angle.
          3. The arc sweeps from T1 to T2 in the direction determined by the
             sign of the cross product of the incoming and outgoing unit vectors.

        If adjacent segments are too short for the requested radius the arc is
        scaled down to fit within half the length of the shorter segment.

        Parameters
        ----------
        radius : float
            Desired circular-arc radius at each corner.
        n_arc_points : int
            Number of sample points used to approximate each arc (default 20).

        Returns
        -------
        PWLPath
            A new piecewise-linear path approximating the smooth trajectory.
        """
        if len(self.waypoints) < 3:
            return PWLPath(self.waypoints.tolist())

        smooth_wpts = [self.waypoints[0].tolist()]

        for i in range(1, len(self.waypoints) - 1):
            P = self.waypoints[i]
            A = self.waypoints[i - 1]
            B = self.waypoints[i + 1]

            v_in  = P - A
            v_out = B - P
            d_in  = float(np.linalg.norm(v_in))
            d_out = float(np.linalg.norm(v_out))

            if d_in < 1e-9 or d_out < 1e-9:
                smooth_wpts.append(P.tolist())
                continue

            u_in  = v_in  / d_in
            u_out = v_out / d_out

            # Interior angle γ at the corner  (0 = U-turn, π = straight)
            cos_gamma = float(np.clip(np.dot(-u_in, u_out), -1.0, 1.0))
            gamma  = math.acos(cos_gamma)
            half_g = gamma * 0.5
            sin_hg = math.sin(half_g)
            tan_hg = math.tan(half_g)

            if sin_hg < 1e-6:          # near-U-turn or degenerate
                smooth_wpts.append(P.tolist())
                continue

            # Ideal tangent distance; capped to half each adjacent segment
            # to prevent arcs from adjacent corners from overlapping
            d_t = min(radius / tan_hg, 0.5 * d_in, 0.5 * d_out)

            if d_t < 1e-9:             # near-straight, arc collapses to a point
                smooth_wpts.append(P.tolist())
                continue

            # Effective radius (smaller than requested if segments are short)
            r_eff = d_t * tan_hg

            # Tangent points on the incoming / outgoing segments
            T1 = P - d_t * u_in
            T2 = P + d_t * u_out

            # Arc centre: along the angle bisector at distance r_eff / sin(γ/2)
            bisector = -u_in + u_out
            b_len = float(np.linalg.norm(bisector))
            if b_len < 1e-9:
                smooth_wpts.append(P.tolist())
                continue
            C = P + (r_eff / sin_hg) * (bisector / b_len)

            # Angular positions of T1 and T2 relative to C
            angle1 = math.atan2(float(T1[1] - C[1]), float(T1[0] - C[0]))
            angle2 = math.atan2(float(T2[1] - C[1]), float(T2[0] - C[0]))

            # Sweep direction from the signed 2-D cross product of u_in and u_out:
            #   cross_z > 0  →  right turn (NED)  →  CCW sweep (angle increases)
            #   cross_z < 0  →  left  turn (NED)  →  CW  sweep (angle decreases)
            cross_z = float(u_in[0] * u_out[1] - u_in[1] * u_out[0])
            if cross_z >= 0:
                if angle2 <= angle1:
                    angle2 += 2.0 * math.pi
            else:
                if angle2 >= angle1:
                    angle2 -= 2.0 * math.pi

            # Sample the arc (first point ≈ T1, last point ≈ T2)
            for a in np.linspace(angle1, angle2, n_arc_points):
                smooth_wpts.append([float(C[0] + r_eff * math.cos(a)),
                                     float(C[1] + r_eff * math.sin(a))])

        smooth_wpts.append(self.waypoints[-1].tolist())
        return PWLPath(smooth_wpts)

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

    def trim(self, lim: Tuple[float, float], normalized: bool = True) -> "PWLPath":
        """
        Trim PWLPath to only keep part between lim[0] and lim[1].
        """
        sub = substring(shapely.LineString(self.waypoints), lim[0], lim[1], normalized=normalized)
        return PWLPath(list(sub.coords))

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
            (-1, 0),
            (1, -0.5),
            (2.5, 1),
            (3, 2.5),
            (3, 4),
            (4, 5)
        ]
    )
    print("Initial Pose (N, E, psi): ", path.get_initial_pose())
    p = (-2, 1) # (4, 5.5) # (3.2, 4.5) # (0, 0) # (2, 0) # (3, 2) # (4, 3)
    ax = path.plot()
    
    print(path.progression(*p))
    desired_wpts = path.get_target_wpts_from(*p, 0.2, 10, final_heading=math.pi)
    
    for wpt in desired_wpts:
        ax.scatter(wpt[1], wpt[0], c='blue')
    ax.scatter(p[1], p[0], c='green')
    ax.scatter(*path.closest_point(*p), c='red')

    print(len(desired_wpts), desired_wpts)
    print("Current waypoint: ", path.get_current_waypoint(*p))
    plt.show()

if __name__ == "__main__":
    test()