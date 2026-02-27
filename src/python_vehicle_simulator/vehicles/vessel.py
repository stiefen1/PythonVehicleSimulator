from abc import abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.lib.guidance import IGuidance, Guidance
from python_vehicle_simulator.lib.navigation import INavigation, Navigation
from python_vehicle_simulator.lib.control import IControl, Control
from python_vehicle_simulator.states.states import Eta, Nu
from python_vehicle_simulator.utils.math_fn import Rzyx
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.diagnosis import IDiagnosis, Diagnosis
from python_vehicle_simulator.lib.dynamics import IDynamics
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import deepcopy
import numpy as np, numpy.typing as npt


# Coordinates in NED frame. yaw is clockwise positive
#   \eta  = [N, E, D, roll, pitch, yaw]  
#   \nu = [u, v, r]
#   x = [\eta, \nu]

VESSEL_GEOMETRY = lambda loa, beam: np.array([
    (loa/2,     0,          0),
    (loa/4,     beam/2,     0),
    (-loa/2,    beam/2,     0),
    (-loa/2,    -beam/2,    0),
    (loa/4,     -beam/2,    0),
    (loa/2,     0,          0)
]).T # North-East-Down coordinates

class IVessel(IDrawable):
    def __init__(
            self,
            loa: float,
            beam: float,
            dynamics: IDynamics,
            states: Tuple,
            *args,
            guidance:Optional[IGuidance] = None,
            navigation:Optional[INavigation] = None,
            control:Optional[IControl] = None,
            diagnosis:Optional[IDiagnosis] = None,
            name:str = 'IVessel',
            mmsi:Optional[str] = None, # Maritime Mobile Service Identity number
            verbose_level:int = 0,
            **kwargs
    ):
        super().__init__(verbose_level=verbose_level, **kwargs)
        self.dynamics = dynamics
        self.states = np.array(states)
        self._states_0 = deepcopy(self.states)
        self.guidance = guidance or Guidance()
        self.navigation = navigation or Navigation(self.states.copy())
        self.control = control or Control()
        self.diagnosis = diagnosis or Diagnosis(states=self.states.copy(), params=None, dt=self.dynamics.dt)
        self.initial_geometry = VESSEL_GEOMETRY(loa, beam)
        self.name = name
        self.mmsi = mmsi
        self.control_commands_prev = None

    @abstractmethod
    def __dynamics__(self, control_commands:npt.NDArray, current:Current, wind:Wind, *args, theta: Optional[npt.NDArray] = None, **kwargs) -> np.ndarray:
        disturbance = np.array([0, 0, 0]) # Define it as a function of current, wind
        theta = theta if theta is not None else np.array(self.dynamics.nt*[1.0])
        x = self.dynamics.fd(self.states, control_commands, theta, disturbance)
        return x

    def step(
            self,
            current:Current,
            wind:Wind,
            obstacles:List[Obstacle],
            target_vessels:List[Any],
            *args,
            control_commands: Optional[npt.NDArray] = None,
            theta: Optional[npt.NDArray] = None,
            **kwargs
        ) -> Tuple[np.ndarray, float, bool, bool, Dict, bool]:
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
        # GNC
        ## Navigation: measure environments
        measurements, navigation_info = self.navigation(self.states, current, wind, obstacles, target_vessels, *args, **kwargs)
        
        ## Fault Diagnosis
        diagnosis, diagnosis_info = self.diagnosis(**measurements, **navigation_info)
        
        ## Guidance: Get desired states
        states_des, guidance_info = self.guidance(**measurements, **navigation_info, **diagnosis, **diagnosis_info)

        # Control commands can be devised by an RL agent for instance
        if control_commands is None:
            ## Control: Generate action to track desired states
            control_commands, control_info = self.control(states_des, **measurements, **navigation_info, **diagnosis, **diagnosis_info, **guidance_info)

        ## USV Dynamics
        self.states = self.__dynamics__(control_commands, current, wind, theta=theta)

        return (self.states, 0, False, False, {}, False)
    
    def reset(
            self,
            random:bool=False,
            seed=None,
            x_min:Optional[npt.NDArray | Tuple]=None,
            x_max:Optional[npt.NDArray | Tuple]=None,
        ):
        self.guidance.reset()
        self.navigation.reset()
        self.control.reset()

        # Set to initialize value by default
        self.states = self._states_0.copy()

        if random: # if random sampling is asked, eta_min, et_max, nu_min, nu_max must be specified. Otherwise keep default values
            np.random.seed(seed=seed)
            if x_min is not None and x_max is not None:
                self.states = np.random.uniform(x_min, x_max)

    def __plot__(self, ax, *args, verbose:int=0, **kwargs):
        """
        x = East
        y = North
        z = -depth
        """
        ax.scatter(self.eta[1], self.eta[0], *args, **kwargs)
        ax.plot(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax

    def __scatter__(self, ax, *args, **kwargs):
        ax.scatter(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax

    def __fill__(self, ax, *args, **kwargs):
        ax.fill(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax
    
    def get_geometry_from_pose(self, eta:Eta) -> npt.NDArray:
        return Rzyx(*eta.rpy) @ self.initial_geometry + eta.to_numpy()[0:3].reshape(3, 1)

    def get_geometry_in_frame(self, eta:Eta) -> npt.NDArray:
        """
        get geometry in a frame specified by nedrpy
        """
        return Rzyx(*eta.rpy).T @ (Rzyx(*self.eta.rpy) @ self.initial_geometry - eta.to_numpy()[0:3].reshape(3, 1) + self.eta.to_numpy()[0:3].reshape(3, 1))

    @property
    def geometry(self) -> npt.NDArray:
        return self.get_geometry_from_pose(self.eta)
    
    @property
    def geometry_for_2D_plot(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.geometry[1, :], self.geometry[0, :]
    
    @property
    def eta(self) -> Eta:
        return Eta(*self.states[0:6])
    
    @property
    def nu(self) -> Nu:
        return Nu(*self.states[6:12])
    