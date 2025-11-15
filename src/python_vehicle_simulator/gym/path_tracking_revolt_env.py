import gymnasium as gym, numpy as np
from typing import Dict, Optional, Tuple, List, Literal
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.utils.math_fn import ssa
import matplotlib.pyplot as plt
from python_vehicle_simulator.lib.path import PWLPath



class GymPathTrackingReVoltEnv(gym.Env):
    def __init__(
            self,
            own_vessel:IVessel,
            target_vessels:List[IVessel],
            obstacles:List[Obstacle],
            wind:Wind,
            current:Current,    
            render_mode:Literal['human', 'human3d']=None,
            path_params:Dict={'d_tot':50, 'max_turn_deg':30, 'seg_len_range': (5, 10), 'start': (0, 0), 'initial_angle':0},
            desired_speed_range:Tuple=(0.2, 0.7),
            heading_max_error:Tuple=np.pi/3,
            horizon:int=20,
            controller_type:Literal["acados", None] = None,
            max_steps:int=200,
            dp:float=None # distance between desired waypoints 
    ):
        self.own_vessel = own_vessel
        self.target_vessels = target_vessels
        self.obstacles = obstacles
        self.wind = wind
        self.current = current
        self.path_params = path_params
        self.desired_speed_range = desired_speed_range
        self.heading_max_error = heading_max_error
        self.horizon = horizon
        self.controller_type = controller_type
        self.dp = dp

        self.init_action_space()
        self.init_observation_space()

        self.safety_radius = 2.5

        self.action_repeat = 1 # dt is 0.2

        # Rendering 
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Current step (for plot purpose)
        self._step = 0
        self.max_steps = max_steps # i.e. 100 seconds for dt=0.02 and action_repeat=10

        # observation space
        self.ne_range = {"min": np.array(2*[-self.path_params['d_tot']]), "max": np.array(2*[self.path_params['d_tot']])}
        self.uvr_range = {"min": np.array([-10, -10, -10]), "max": np.array([10, 10, 10])}
        self.rel_target_range = {"min":np.array(self.horizon*[0]), "max": np.array(self.horizon*[self.path_params['d_tot']])} # relative distance to a point of the horizon
        self.rel_yaw_range = {"min": np.array(self.horizon*[-np.pi]), "max": np.array(self.horizon*[np.pi])} # relative bearing angle to a point of the horizon
        self.speed_error_range = {"min": np.array([-3*self.desired_speed_range[1]]), "max": np.array([3*self.desired_speed_range[1]])}
        self.u_actual_range = {"min": np.concatenate([actuator.u_min for actuator in self.own_vessel.actuators]), "max": np.concatenate([actuator.u_max for actuator in self.own_vessel.actuators])}

        # Reward function
        self.huber_penalty_slope = 10 # delta
        self.huber_penalty_weight = 30 # q_x,y
        self.heading_penalty_weight = 100 # q_psi
        self.singular_value_penalty = 1e-3 # epsilon -> for nonsigular thruster configuration
        self.singular_value_weight = 1e-5 # rho -> for nonsigular thruster configuration

        self.Q = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ]) # Velocity weight matrix

        self.Ra = np.eye(3) * 1e-2 # Azimuth weight matrix
        self.Rf = np.eye(3) * 0 # 1e-3 # 1e-1 # Force weight matrix

    def cost_tracking(self, p:np.ndarray, p_d:np.ndarray) -> float:
        return float(self.huber_penalty_slope**2 * (np.sqrt(1 + ((p[0:2]-p_d[0:2]).T @ (p[0:2]-p_d[0:2])) / self.huber_penalty_slope**2) - 1))
    
    def cost_heading(self, psi:float, psi_d:float) -> float:
        return float((1 - np.cos(psi - psi_d)) / 2)
    
    def cost_speed(self, nu:np.ndarray, nu_d:np.ndarray) -> float:
        return float((nu-nu_d).T @ self.Q @ (nu-nu_d))
    
    def cost_alpha(self, alpha:np.ndarray) -> float:
        return alpha.T @ self.Ra @ alpha
    
    def cost_force(self, force:np.ndarray) -> float:
        return force.T @ self.Rf @ force
    
    def cost_singularity(self, alpha:np.ndarray) -> float:
        B = np.array([
            [np.cos(alpha[0]), np.sin(alpha[0]), self.own_vessel.actuators[0].xy[0]*np.sin(alpha[0]) - self.own_vessel.actuators[0].xy[1] * np.cos(alpha[0])],
            [np.cos(alpha[1]), np.sin(alpha[1]), self.own_vessel.actuators[1].xy[0]*np.sin(alpha[1]) - self.own_vessel.actuators[1].xy[1] * np.cos(alpha[1])],
            [np.cos(alpha[2]), np.sin(alpha[2]), self.own_vessel.actuators[2].xy[0]*np.sin(alpha[2]) - self.own_vessel.actuators[2].xy[1] * np.cos(alpha[2])]
        ])
        return 1/(np.linalg.det(B@B.T) + self.singular_value_penalty)
    
    def cost(self, eta:np.ndarray, nu:np.ndarray, eta_d:np.ndarray, nu_d:np.ndarray, force:np.ndarray, alpha:np.ndarray=np.array([0, 0, 0])) -> float:
        # print(
        #     self.huber_penalty_weight * self.cost_tracking(eta[0:2], eta_d[0:2]),
        #     self.heading_penalty_weight * self.cost_heading(eta[2], eta_d[2]),
        #     self.singular_value_weight * self.cost_singularity(alpha),
        #     self.cost_speed(nu, nu_d),
        #     self.cost_alpha(alpha),
        #     self.cost_force(force)
        #     )
        # print(eta, nu, eta_d, nu_d, force, alpha)
        # print(alpha, force)

        
        cost = self.huber_penalty_weight * self.cost_tracking(eta[0:2], eta_d[0:2])
        cost += self.heading_penalty_weight * self.cost_heading(eta[2], eta_d[2])
        cost += self.singular_value_weight * self.cost_singularity(alpha)
        cost += self.cost_speed(nu, nu_d)
        cost += self.cost_alpha(alpha)
        cost += self.cost_force(force)
        return float(cost[0, 0])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, random: Optional[bool] = True) -> Tuple[Dict, Dict]:
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

        # Reset own vessel position to 0 -> MAYBE CONSIDER CHOOSING A RANDOM INITIAL SPEED AT THE BEGINNING OF EACH EPISODE FOR BETTER EXPLORATION
        self.own_vessel.reset(
            random=random,
            seed=seed,
            eta_min=np.array([-1, -1, 0, 0, 0, -0.2]),
            eta_max=np.array([1, 1, 0, 0, 0, 0.2]),
            nu_min=np.array([self.desired_speed_range[0]]+5*[0]),
            nu_max=np.array([self.desired_speed_range[1]]+5*[0])
        )
        for target_vessel in self.target_vessels:
            target_vessel.reset(random=random)

        # Sample a new target position in [-20, 20] x [-20, 20]
        self.path = PWLPath.sample(**self.path_params, seed=seed)
        self.desired_speed = float(np.random.uniform(*self.desired_speed_range))

        observation = self._get_obs()
        info = self._get_info()

        # Reset figure
        self.fig = None
        self.ax = None
        self.vessel_plot = None

        # Reset step (for plot purpose)
        self._step = 0

        return observation, info

    def step(self, action, rescale:bool=True) -> Tuple[Dict, float, bool, bool, Dict]:
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
            self.own_vessel.step(self.current, self.wind, self.obstacles, self.target_vessels, control_commands=self.map_action_to_command(action, rescale=rescale))

        # Reward result of action
        reward = self.reward()

        # Get observation
        observation = self._get_obs()

        # Check for collisions with environment
        terminated = self.collision()
        terminated = terminated or self.states_out_of_range()

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = self._step >= self.max_steps
        
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def states_out_of_range(self) -> float:
        eta = np.array(self.own_vessel.eta.neyaw)
        surge_speed = self.own_vessel.nu.u
        eta_d = self.path.get_target_wpts_from(eta[0], eta[1], 0, 1)[0]
        heading_absolute_error = abs(ssa(self.own_vessel.eta.yaw - eta_d[2]))
        # print(f"heading error: {absolute_error:.1f}, h: {eta[2]:.1f}, hd: {eta_d[2]:.1f}")
        # print(f"{self.desired_speed_range[0]:.2f} <= surge speed: {surge_speed:.2f} <= {self.desired_speed_range[1]:.2f}")
        if heading_absolute_error > self.heading_max_error:
            return True
        elif (surge_speed > self.desired_speed_range[1]) or (surge_speed < self.desired_speed_range[0]):
            return True
        return False        

    def reward(self) -> float:
        eta = np.array(self.own_vessel.eta.neyaw)
        nu = np.array(self.own_vessel.nu.uvr)
        eta_d = np.array(self.path.get_target_wpts_from(eta[0], eta[1], 0, 1)[0]) # dp is wrong but we don't care because we only take the first point
        nu_d = np.array([self.desired_speed, 0, 0])
        force = np.array([actuator.thrust for actuator in self.own_vessel.actuators])
        alpha = np.array([actuator.u_actual_prev[0] for actuator in self.own_vessel.actuators])
        # print(self.cost(eta, nu, eta_d, nu_d, force, alpha)/1e2)
        return float(np.exp(-self.cost(eta, nu, eta_d, nu_d, force, alpha)/1e2)) #float(np.exp(-self.cost(eta, nu, eta_d, nu_d, force, alpha)/10))

    def dist_to_target(self) -> float:
        ne = np.array(self.own_vessel.eta.neyaw[0:2])
        target = np.array(self.path.closest_point(ne[0], ne[1]))
        return np.linalg.norm(ne-target)

    def collision(self) -> bool:
        if np.any(self.own_vessel.eta.to_numpy()[0:2] > 50) or np.any(self.own_vessel.eta.to_numpy()[0:2] < -50):
            return True
        for obs in self.obstacles:
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
        distances, rel_yaws = [], []
        for target in self.path.get_target_wpts_from(ne[0], ne[1], self.dp or self.action_repeat*self.own_vessel.dt*self.desired_speed, self.horizon):
            delta = target[0:2] - ne
            distance = float(np.linalg.norm(delta))
            rel_yaw = ssa(self.own_vessel.eta[5] + np.atan2(-delta[1], delta[0]))
            distances.append(distance)
            rel_yaws.append(rel_yaw)

        u_actual = np.concatenate([actuator.u_actual_prev for actuator in self.own_vessel.actuators])

        # Normalize each and cast to float32
        ne_norm = self._normalize(ne, self.ne_range["min"], self.ne_range["max"]).astype(np.float32)
        uvr_norm = self._normalize(uvr, self.uvr_range["min"], self.uvr_range["max"]).astype(np.float32)
        rel_target_norm = self._normalize(np.array(distances), self.rel_target_range["min"], self.rel_target_range["max"]).astype(np.float32)
        rel_yaw_norm = self._normalize(np.array(rel_yaws), self.rel_yaw_range["min"], self.rel_yaw_range["max"]).astype(np.float32)
        speed_error_norm = self._normalize(np.array(self.own_vessel.nu.u - self.desired_speed), self.speed_error_range["min"], self.speed_error_range["max"]).astype(np.float32)
        u_actual_norm = self._normalize(u_actual, self.u_actual_range["min"], self.u_actual_range["max"]).astype(np.float32)

        return {
            "ne": ne_norm,
            "uvr": uvr_norm,
            "rel_target": rel_target_norm,
            "rel_yaw": rel_yaw_norm,
            "speed_error": speed_error_norm,
            "u_actual": u_actual_norm,
        }
    
    def _get_info(self) -> Dict:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {

        }
    
    def map_action_to_command(self, action, rescale:bool=True) -> None:
        """
        Map the interval -1, 1 to the actuator range if rescale=True

        rescale=False can be useful when action is produced by an MPC controller for instance.

        action = [a1, n1, a2, n2, a3, n3] if self.controller_type=None elif self.controller_type="acados" action = [a1, a2, a3, n1, n2, n3]
        """
        # print("action: ", action)
        command = []
        idx = 0
        if self.controller_type is None:
            for actuator in self.own_vessel.actuators:
                N_i = actuator.u_min.shape[0] # Number of commands expected for actuator i
                if rescale:
                    command.append(action[idx:idx+N_i] * (actuator.u_max - actuator.u_min) / 2  + (actuator.u_max + actuator.u_min) / 2)
                else:
                    command.append(action[idx:idx+N_i])
                idx += N_i
        elif self.controller_type == "acados":
            idx_action = 0
            for actuator in self.own_vessel.actuators:
                N_i = actuator.u_min.shape[0] # Number of commands expected for actuator i
                command.append(np.array([action[idx_action], action[idx_action+3]]))
                idx += N_i
                idx_action += 1
        # print("action: ", action, "command: ", command)
        return command
    
    def init_action_space(self) -> None:
        self.action_space = gym.spaces.Box(-np.ones(shape=(6,)), np.ones(shape=(6,))) # action space is -1, +1

    def init_observation_space(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "ne": gym.spaces.Box(-1.0, 1.0, shape=(2,)),            # Because we want the vessel to remain within bounds
                "uvr": gym.spaces.Box(-1.0, 1.0, shape=(3,)),           # Surge-Sway-YawRate
                "rel_target": gym.spaces.Box(-1.0, 1.0, shape=(self.horizon,)),    # Easier to figure out using relative pose
                "rel_yaw": gym.spaces.Box(-1.0, 1.0, shape=(self.horizon,))  ,
                "speed_error": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                "u_actual": gym.spaces.Box(-1.0, 1.0, shape=(6,))      
            }
        )

    def render(self, mode=None):
        mode = mode or self.render_mode

        if mode is None or mode not in ("human"):
            return

        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.vessel_plot, = self.ax.plot([], [], label='Vessel')
            # Use plot instead of scatter for waypoints
            self.path_plot, = self.ax.plot([], [], 'go', markersize=3, label='Target Waypoints')
            self.ax.set_xlim(-30, 30)
            self.ax.set_ylim(-30, 30)
            self.path.plot(ax=self.ax)
            self.ax.set_xlabel('East')
            self.ax.set_ylabel('North')
            
            self.ax.legend()
            plt.ion()
            plt.show()

        # print(self.own_vessel.nu.uvr[0], self.desired_speed)
        self.vessel_plot.set_data(*self.own_vessel.geometry_for_2D_plot)
        
        # Update waypoints using set_data (works with Line2D objects)
        wpts = self.path.get_target_wpts_from(*self.own_vessel.eta.ned[0:2], self.dp or self.action_repeat*self.desired_speed*self.own_vessel.dt, self.horizon)
        if wpts:
            self.path_plot.set_data([wpt[1] for wpt in wpts], [wpt[0] for wpt in wpts])
        else:
            self.path_plot.set_data([], [])
        
        self.ax.set_title(f"Time: {self._step*self.action_repeat*self.own_vessel.dt:.1f} [s]")
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
    # from python_vehicle_simulator.vehicles.otter import Otter, OtterParameters, OtterThrusterParameters
    from python_vehicle_simulator.vehicles.revolt3 import Revolt3DOF, RevoltThrusterParameters, RevoltParameters3DOF, RevoltSternThrusterParams, RevoltBowThrusterParams
    from python_vehicle_simulator.lib.actuator import AzimuthThruster
    from python_vehicle_simulator.lib.weather import Wind, Current
    from python_vehicle_simulator.utils.unit_conversion import DEG2RAD

    dt = 0.2

    revolt = Revolt3DOF(
        params=RevoltParameters3DOF(),
        eta=np.array([0, 0, 0, 0, 0, 0]),
        dt=dt,
        actuators=[
            AzimuthThruster(xy=(-1.65, -0.15), **vars(RevoltSternThrusterParams())),
            AzimuthThruster(xy=(-1.65, 0.15), **vars(RevoltSternThrusterParams())), # , faults=[{'type': 'loss-of-efficiency', 't0': 20, 'efficiency': 0.5}]),
            AzimuthThruster(xy=(1.15, 0.0), **vars(RevoltBowThrusterParams()))
        ],
    )

    env = GymPathTrackingReVoltEnv(
        own_vessel=revolt,
        target_vessels=[],
        obstacles=[],
        wind=Wind(0, 0),
        current=Current(beta=-30.0*DEG2RAD, v=0.3),

    )
    # This will catch many common issues
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

if __name__=="__main__":
    # generate_random_pwl_path()
    check_environment()