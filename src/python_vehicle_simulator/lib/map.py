import numpy as np, shapely
from typing import List, Tuple
from python_vehicle_simulator.lib.obstacle import Obstacle

class RandomMapGenerator:
    def __init__(
            self,
            north_lim:tuple,
            east_lim:tuple,
            radius_lim:tuple,
            min_dist:float
    ):
        self.north_lim = north_lim
        self.east_lim = east_lim
        self.radius_lim = radius_lim
        self.min_dist = min_dist
        self.map_area = (self.north_lim[1]-self.north_lim[0])*(self.east_lim[1]-self.east_lim[0])

    def get(self, safe_points:List[Tuple], min_density:float=0.05, max_iter:int=100, seed=None) -> List[Obstacle]:
        """
        safe_points is a list of (n, e) coordinates corresponding to point that we want to keep obstacle free
        can be for example starting/target positions.
        """
        np.random.seed(seed=seed)

        density = 0
        tot_obs_area = 0.0
        obstacles:List[shapely.Polygon] = []
        iteration = 0
        while density < min_density and iteration < max_iter:
            # Center of the candidate obstacle
            r = np.random.uniform(*self.radius_lim, size=1)[0]
            n = np.random.uniform(self.north_lim[0]+r, self.north_lim[1]-r, size=1)[0]
            e = np.random.uniform(self.east_lim[0]+r, self.east_lim[1]-r, size=1)[0]
            
            # Create candidate obstacle
            candidate:shapely.Polygon = shapely.Point(n, e).buffer(r)

            # Check acceptance criteria
            accept = True
            for obs in obstacles:
                if candidate.distance(obs) < self.min_dist:
                    accept = False
                    break

            for point in safe_points:
                dist = candidate.distance(shapely.Point(point[0], point[1]))
                # print(dist)
                if dist < self.min_dist:
                    # print(dist, "not accepted")
                    accept = False
                    break

            if accept:
                obstacles.append(candidate)
                tot_obs_area += candidate.area

                # Compute density
                density = tot_obs_area / self.map_area

            iteration += 1
        return [Obstacle(list(zip(*obs.boundary.xy))) for obs in obstacles], {'density':density, 'iter':iteration, 'obstacle area': tot_obs_area}

if __name__=="__main__":
    point = shapely.Point(2.8, 0)
    circle = shapely.Point(2, 0).buffer(3)
    # print(type(circle))
    print("dist: ", shapely.distance(point, circle))



    import matplotlib.pyplot as plt
    gen = RandomMapGenerator(
        (-30, 30),
        (-30, 30),
        (1, 10),
        4
    )
    safe_points = [
        (0, 0),
        (25, 10),
        (-20, -10)
    ]
    map1, info = gen.get(safe_points=safe_points, max_iter=10000, min_density=0.2)
    ax = None
    for obs in map1:
        ax = obs.plot(ax=ax, c='black')
    for point in safe_points:
        ax.scatter(point[1], point[0], c='red')
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_title(f"{len(map1)} Obstacles | Density: {info['density']:.2f} | Iterations: {info['iter']} | Area: {info['obstacle area']:.1f}")
    ax.set_aspect('equal')
    plt.show()





