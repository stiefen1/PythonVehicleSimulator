from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class NavEnv:
    def __init__(
            self,
            own_vessel:IVessel,
            target_vessels:List[IVessel],
            obstacles:List[Obstacle],
            dt:float,
            *args,
            wind:Wind=None,
            current:Current=None,
            **kwargs
    ):
        self.own_vessel = own_vessel
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.wind = wind or Wind(0, 0)
        self.current = current or Current(0, 0)
        self._dt = dt

        # Simulation results
        self.timestamps:List[float] = []
        self.observations:List[Any] = []    # Observation space
        self.rewards:List[float] = []       # Reward as a result of taking an action
        self.terminated:List[bool] = []     #
        self.truncated:List[bool] = []
        self.infos:List[Dict] = []
        self.dones:List[bool] = []
        self.t = 0

        # Rendering 
        self.fig = None
        self.ax = None
        self.vessel_plot = None
        

    def reset(self):
        self.timestamps:List[float] = []
        self.observations:List[Any] = []    # Observation space
        self.rewards:List[float] = []       # Reward as a result of taking an action
        self.terminated:List[bool] = []     #
        self.truncated:List[bool] = []
        self.infos:List[Dict] = []
        self.dones:List[bool] = []
        self.t = 0
        self.fig: Optional[Figure] = None
        self.ax = None
        self.vessel_plot = None

    def step(self, *args, **kwargs) -> Tuple[List, float, bool, bool, Dict, bool]:
        """
        One step forward in time. Returns a tuple containing

        - observation
        - reward
        - terminated
        - truncated
        - info
        - done

        Not useful yet, but will be for later integration with Gymnasium
        """
        # Step target vessels
        for tv in self.target_vessels:
            target_vessels = self.target_vessels.copy().remove(tv)
            tv.step(self.current, self.wind, self.obstacles, target_vessels+[self.own_vessel], *args, **kwargs)

        # Step own vessel
        obs, r, term, trunc, info, done = self.own_vessel.step(self.current, self.wind, self.obstacles, self.target_vessels, *args, **kwargs)
        
        # Rendering
        # if self.skip_frames == 0 or (self.t//self.dt) % (self.skip_frames) == 0:
        #     self.render(self.render_mode, verbose=self.verbose)

        # Time travel
        self.t += self.dt

        # Save output for replay
        self.timestamps.append(self.t)
        self.observations.append(obs)
        self.rewards.append(r)
        self.terminated.append(term)
        self.truncated.append(info)
        self.infos.append(info)
        self.dones.append(done)

        return obs, r, term, trunc, info, done

    def render(self, mode=None, window_size:Tuple=(10, 10), verbose:int=0):
        if mode not in ("human", "human3d"):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            plt.show()

        self.ax.cla()
        if mode == "human3d":
            self.own_vessel.plot(ax=self.ax, verbose=verbose)
        else:
            self.own_vessel.plot(ax=self.ax, verbose=verbose, c='black')
            for obs in self.obstacles:
                obs.plot(ax=self.ax, verbose=verbose, c='grey')
            for tv in self.target_vessels:
                tv.plot(ax=self.ax, verbose=verbose, c='red')

        self.ax.set_xlim([self.own_vessel.eta[1]-window_size[0]/2, self.own_vessel.eta[1]+window_size[0]/2])
        self.ax.set_ylim([self.own_vessel.eta[0]-window_size[1]/2, self.own_vessel.eta[0]+window_size[1]/2])
        self.ax.set_xlabel('East')
        self.ax.set_ylabel('North')
        self.ax.set_title(f"Vessel Position (t={self.t:.1f})")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        pass

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, val:float):
        self.own_vessel.dt = val
        for target_vessel in self.target_vessels:
            target_vessel.dt = val
        self._dt = val

        
