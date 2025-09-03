import gymnasium as gym, numpy as np
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
import matplotlib.pyplot as plt


class GymNavEnv(gym.Env):
    def __init__(
            self,
            own_vessel:IVessel,
            target_vessels:List[IVessel],
            obstacles:List[Obstacle],
            wind:Wind,
            current:Current,    
            render_mode:Literal['human', 'human3d']=None        
    ):
        self.own_vessel = own_vessel
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.wind = wind
        self.current = current

        self.init_action_space()
        self.init_observation_space()

        self.min_dist = 1.0 # distance to terminate episode
        self.safety_radius = 2.5
        self.target_range = {'n': (-20, 20), 'e': (-20, 20)}

        self.action_repeat = 10 # dt is 0.02 so this make the RL frequency 1/(10*0.02) = 1/0.2 = 5Hz

        # Rendering 
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.vessel_plot = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        np.random.seed(seed=seed)

        # Reset own vessel position to 0
        self.own_vessel.reset()
        for target_vessel in self.target_vessels:
            target_vessel.reset()

        # Sample a new target position in [-20, 20] x [-20, 20]
        self.target = np.random.uniform(-30, 30, size=2).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        # Reset figure
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        return observation, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        # Step vessels
        for _ in range(self.action_repeat):
            for vessel in self.target_vessels:
                vessel.step(self.current, self.wind, self.obstacles, [])
            self.own_vessel.step(self.current, self.wind, self.obstacles, self.target_vessels, control_commands=self.map_action_to_command(action))

        # Reward result of action
        reward = self.reward()

        # Get observation
        observation = self._get_obs()

        # Check if agent reached the target
        terminated = self.target_reached() or self.collision()

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False
        
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reward(self) -> float:
        # if distance <= 10 we start rewarding, otherwise penalize. If collision, add huge penalty
        return 1-self.dist_to_target()/10 if not self.collision() else -10000 # np.exp(-self.dist_to_target()/30) if not self.collision() else -1000
    
    def dist_to_target(self) -> float:
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        return np.linalg.norm(ne-self.target)

    def target_reached(self) -> bool:
        return True if self.dist_to_target() <= self.min_dist else False
    
    def collision(self) -> bool:
        for obs in self.obstacles:
            if obs.distance(*self.own_vessel.eta.neyaw[0:2]) < self.safety_radius:
                return True
        return False

    def _get_obs(self) -> Dict:
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "neyaw": np.array(self.own_vessel.eta.neyaw, dtype=np.float32),
            "uvr": np.array(self.own_vessel.nu.uvr, dtype=np.float32),
            "rel_target": self.target - self.own_vessel.eta.to_numpy()[0:2].astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {

        }
    
    def map_action_to_command(self, action) -> None:
        """
        Map the interval -1, 1 to the actuator range
        """
        command = np.zeros_like(action)
        idx = 0
        for actuator in self.own_vessel.actuators:
            N_i = actuator.u_min.shape[0] # Number of commands expected for actuator i
            command[idx:idx+N_i] = action[idx:idx+N_i] * (actuator.u_max - actuator.u_min) / 2  + (actuator.u_max + actuator.u_min) / 2
            idx += N_i
        return command
    
    def init_action_space(self) -> None:
        self.action_space = gym.spaces.Box(-np.ones(shape=(len(self.own_vessel.actuators),), dtype=np.float32), np.ones(shape=(len(self.own_vessel.actuators),), dtype=np.float32)) # action space is -1, +1

    def init_observation_space(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "neyaw": gym.spaces.Box(np.array([-1e3, -1e3, -10], dtype=np.float32), np.array([1e3, 1e3, 10], dtype=np.float32)),  # North-East-Yaw
                "uvr": gym.spaces.Box(np.array([-1e3, -1e3, -10], dtype=np.float32), np.array([1e3, 1e3, 10], dtype=np.float32)),    # Surge-Sway-YawRate
                "rel_target": gym.spaces.Box(np.array([-1e3, -1e3], dtype=np.float32), np.array([1e3, 1e3], dtype=np.float32)),
                "a": gym.spaces.Discrete(
                    
                )
            }
        )

    def render(self, mode=None):
        mode = mode or self.render_mode
        if mode not in ("human", "human3d"):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            if mode == "human3d":
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.vessel_plot, = self.ax.plot([], [], [], label='Vessel')
                self.ax.set_xlim(-30, 30)
                self.ax.set_ylim(-30, 30)
                self.ax.set_zlim(-10, 10)
                self.ax.set_xlabel('East')
                self.ax.set_ylabel('North')
                self.ax.set_zlabel('-Down')
                self.ax.scatter3D(*self.target, zs=0, c='red')
            else:
                self.ax = self.fig.add_subplot(111)
                self.vessel_plot, = self.ax.plot([], [], label='Vessel')
                self.ax.set_xlim(-30, 30)
                self.ax.set_ylim(-30, 30)
                self.ax.scatter(*self.target, c='red')
                self.ax.set_xlabel('East')
                self.ax.set_ylabel('North')
            self.ax.set_title('Vessel Position')
            self.ax.legend()
            plt.ion()
            plt.show()

        # Assuming vessel position is self.own_vessel.eta.e, .n, .d
        # x = self.own_vessel.eta.e
        # y = self.own_vessel.eta.n
        # z = getattr(self.own_vessel.eta, 'd', 0.0)
        if mode == "human3d":
            geometry = self.own_vessel.geometry_for_3D_plot
            self.vessel_plot.set_data(*geometry[0:2])
            self.vessel_plot.set_3d_properties(geometry[2])
        else:
            self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

# # Register the environment so we can create it with gym.make()
# gym.register(
#     id="gymnasium_env/GymNavEnv-v0",
#     entry_point=GymNavEnv,
#     max_episode_steps=300,  # Prevent infinite episodes
# )

def check_environment() -> None:
    from gymnasium.utils.env_checker import check_env
    from python_vehicle_simulator.vehicles.otter import Otter, OtterParameters, OtterThrusterParameters
    from python_vehicle_simulator.lib.actuator import Thruster
    from python_vehicle_simulator.lib.weather import Wind, Current
    from python_vehicle_simulator.utils.unit_conversion import DEG2RAD

    dt = 0.02

    thruster_params = OtterThrusterParameters()
    otter = Otter(
        OtterParameters(),
        dt,
        actuators=[Thruster(xy=(0, 0.395), **vars(thruster_params)), Thruster(xy=(0, -0.395), **vars(thruster_params))]
    )

    env = GymNavEnv(
        own_vessel=otter,
        target_vessels=[],
        obstacles=[],
        wind=Wind(0, 0),
        current=Current(beta=-30.0*DEG2RAD, v=0.3)
    )
    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

if __name__=="__main__":
    check_environment()