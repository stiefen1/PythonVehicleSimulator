import gymnasium as gym, numpy as np
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.map import RandomMapGenerator
from python_vehicle_simulator.utils.math_fn import ssa
import matplotlib.pyplot as plt



class GymNavEnvTOF(gym.Env):
    def __init__(
            self,
            own_vessel:IVessel,
            target_vessels:List[IVessel],
            obstacles:List[Obstacle],
            wind:Wind,
            current:Current,    
            render_mode:Literal['human', 'human3d']=None,
            verbose:int=0        
    ):
        self.own_vessel = own_vessel
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.wind = wind
        self.current = current

        self.init_action_space()
        self.init_observation_space()

        # self.min_dist = 1.0 # distance to terminate episode
        self.safety_radius = 2.5
        # self.target_range = {'n': (-20, 20), 'e': (-20, 20)}

        self.action_repeat = 10 # dt is 0.02 so this make the RL frequency 1/(10*0.02) = 1/0.2 = 5Hz

        # Rendering 
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.vessel_plot = None
        self.verbose = verbose

        # Current step (for plot purpose)
        self._step = 0
        self.max_steps = 200 # i.e. 100 seconds for dt=0.02 and action_repeat=10

        # observation space
        self.ne_range = {"min": np.array([-50, -50]), "max": np.array([50, 50])}
        self.uvr_range = {"min": np.array([-50, -50, -10]), "max": np.array([50, 50, 10])}
        self.rel_target_range = {"min":np.array([-50, -50]), "max": np.array([50, 50])}
        self.rel_yaw_range = {"min": np.array([-np.pi]), "max": np.array([np.pi])}
        self.tofs_range = {"min": self.own_vessel.navigation.sensors['tof'].min, "max": self.own_vessel.navigation.sensors['tof'].max}

        # Reward function
        self.delta = 0.1

        # Map
        self.map_generator = RandomMapGenerator(
            north_lim=(self.ne_range["min"][0], self.ne_range["max"][0]),
            east_lim=(self.ne_range["min"][1], self.ne_range["max"][1]),
            radius_lim=(5, 30),
            min_dist=5
        )
        self.map_obstacles = []

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
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Reset own vessel position to 0
        self.own_vessel.reset()
        for target_vessel in self.target_vessels:
            target_vessel.reset()

        # Sample a new target position in [-20, 20] x [-20, 20]
        self.target:np.ndarray = self.np_random.uniform(-30, 30, size=2)

        observation = self._get_obs()
        info = self._get_info()

        # Reset figure
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Reset step (for plot purpose)
        self._step = 0

        # Map
        self.map_obstacles, _ = self.map_generator.get(safe_points=[(0., 0.), tuple(self.target.tolist())], min_density=0.2, seed=seed)

        return observation, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._step += 1

        # Step vessels
        for _ in range(self.action_repeat):
            for vessel in self.target_vessels:
                vessel.step(self.current, self.wind, self.obstacles, [])
            self.own_vessel.step(self.current, self.wind, self.obstacles+self.map_obstacles, self.target_vessels, control_commands=self.map_action_to_command(action))
        # print("TOF: ", self.own_vessel.navigation.sensors['tof'].last_measurement)
        # Reward result of action
        reward = self.reward(command=self.map_action_to_command(action))

        # Get observation
        observation = self._get_obs()

        # Check if agent reached the target
        terminated = self.collision()

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = self._step >= self.max_steps
        
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reward(self, command=None) -> float:
        # I want to reward the vessel for being alive and terminate an episode if it gets outside of my map
        # -> constant reward at each timestamp.
        # -> No termination condition when close to the target
        # print(command.shape, command)
        # This one worked better:
        return (
            1 -
            (self.dist_to_target()/100) -
            (command.T @ command)/command.shape[0]/1e6
            # (command.T @ command)/command.shape[0]/1e5 # worst case is command.T@command=20'000.
            # /1e5 and /5e4 is not enough, the vessel is not moving -> 20'000 / 2 / 50'000 = 0.2
            # /1e5 -> 20'000 / 2 / 100'000 = 0.1
            
            # np.min([0, self.own_vessel.nu[0]])
        )

        # (Reinforcement learning-based NMPC for tracking control of ASVs: Theory and experiments)
        # first suggested in (NMPC based on Huber Penalty Functions to Handle Large Deviations of Quadrature States) from Gros & Diehl
        # Consider pseudo-Huber function for better numerical stability:
        # return (self.delta**2)*(np.sqrt(1+(self.dist_to_target()/self.delta)**2)-1)

        # return np.exp(-self.dist_to_target()/100) # \in [0, 1]. equal to 1 if distance is zero, otherwise decreases ----------------> Ca ne pénalise pas assez violemment le fait de ne pas être proche, le bateau fait des courbes dans tous les sens.
        # return np.exp(-self.dist_to_target()/10) # Plus grande pénalité, au début le bateau part en arrière et fait des courbes dans tous les sens -> pénalité pour qu'il aille vers l'avant ?

        # Maybe think about a more complex cost where heading error is penalized as well

        # if distance <= 10 we start rewarding, otherwise penalize. If collision, add huge penalty
        # return 1-((self.dist_to_target()/30)**2) if not self.collision() else -10000 # np.exp(-self.dist_to_target()/30) if not self.collision() else -1000
    
    def dist_to_target(self) -> float:
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        return np.linalg.norm(ne-self.target)

    # def target_reached(self) -> bool:
    #     return True if self.dist_to_target() <= self.min_dist else False
    
    def collision(self) -> bool:
        if np.any(self.own_vessel.eta.to_numpy()[0:2] > self.ne_range["max"]) or np.any(self.own_vessel.eta.to_numpy()[0:2] < self.ne_range["min"]):
            return True
        for obs in self.obstacles + self.map_obstacles:
            if obs.distance(*self.own_vessel.eta.neyaw[0:2]) < self.safety_radius:
                return True
        return False

    def _normalize(self, x, min_val, max_val):
        """Normalize x from [min_val, max_val] to [-1, 1]."""
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def _get_obs(self) -> Dict:
        """Convert internal state to normalized observation format."""
        # Get raw values
        ne = np.array([self.own_vessel.eta.n, self.own_vessel.eta.e])
        uvr = np.array(self.own_vessel.nu.uvr)
        delta = self.target - ne
        rel_yaw = np.array([ssa(self.own_vessel.eta[5] + np.atan2(-delta[1], delta[0]))])
        tofs = self.own_vessel.navigation.sensors['tof'].last_measurement

        # Normalize each and cast to float32
        ne_norm = self._normalize(ne, self.ne_range["min"], self.ne_range["max"]).astype(np.float32)
        uvr_norm = self._normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = self._normalize(delta, self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = self._normalize(rel_yaw, self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        tofs_norm = self._normalize(tofs, self.tofs_range["min"], self.tofs_range["max"]).astype(np.float32)
        return {
            "ne": ne_norm,
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw": rel_yaw_norm,
            "tof": tofs_norm
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
        self.action_space = gym.spaces.Box(-np.ones(shape=(len(self.own_vessel.actuators),)), np.ones(shape=(len(self.own_vessel.actuators),))) # action space is -1, +1

    def init_observation_space(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "ne": gym.spaces.Box(-1.0, 1.0, shape=(2,)),            # Because we want the vessel to remain within bounds
                "uvr": gym.spaces.Box(-1.0, 1.0, shape=(3,)),           # Surge-Sway-YawRate
                "rel_target": gym.spaces.Box(-1.0, 1.0, shape=(2,)),    # Easier to figure out using relative pose
                "rel_yaw": gym.spaces.Box(-1.0, 1.0, shape=(1,)), 
                "tof": gym.spaces.Box(-1.0, 1.0, shape=(self.own_vessel.navigation.sensors['tof'].n_output,))       
            }
        )

    def render(self, mode=None):
        mode = mode or self.render_mode
        if mode not in ("human", "human3d"):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            plt.show()

        self.ax.cla()
        self.own_vessel.plot(ax=self.ax, verbose=self.verbose, c='blue')
        self.ax.scatter(self.target[1], self.target[0], c='red')
        for obs in self.obstacles + self.map_obstacles:
            obs.plot(ax=self.ax, verbose=self.verbose, c='grey')
        for tv in self.target_vessels:
            tv.plot(ax=self.ax, verbose=self.verbose, c='red')
        # self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot)
        self.ax.set_xlim([self.own_vessel.eta[1]-50, self.own_vessel.eta[1]+50])
        self.ax.set_ylim([self.own_vessel.eta[0]-50, self.own_vessel.eta[0]+50])
        self.ax.set_xlabel('East')
        self.ax.set_ylabel('North')
        self.ax.set_title(f"Vessel Position (steps={self._step})")
        # self.ax.legend()
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
    from python_vehicle_simulator.lib.navigation import NavigationTOF
    import numpy as np

    dt = 0.02

    thruster_params = OtterThrusterParameters()
    otter = Otter(
        OtterParameters(),
        dt,
        actuators=[Thruster(xy=(0, 0.395), **vars(thruster_params)), Thruster(xy=(0, -0.395), **vars(thruster_params))],
        navigation=NavigationTOF(
            tof_params={
                "range": 50,
                "angles": np.linspace(-15*DEG2RAD, 15*DEG2RAD, 5)
            }
        )
    )

    env = GymNavEnvTOF(
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