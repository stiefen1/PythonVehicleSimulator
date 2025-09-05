from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.lib.guidance import IGuidance, Guidance
from python_vehicle_simulator.lib.navigation import INavigation, Navigation
from python_vehicle_simulator.lib.control import IControl, Control
from python_vehicle_simulator.lib.integrator import Euler
from python_vehicle_simulator.states.states import Eta, Nu
from python_vehicle_simulator.utils.math_fn import Rzyx, Tzyx
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.actuator import IActuator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import deepcopy




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
            params,
            dt,
            *args,
            eta:Union[Eta, Tuple]=None,
            nu:Union[Nu, Tuple]=None,
            guidance:IGuidance=None,
            navigation:INavigation=None,
            control:IControl=None,
            actuators:List[IActuator]=None,
            name:str='IVessel',
            mmsi:str=None, # Maritime Mobile Service Identity number
            verbose_level:int=0,
            **kwargs
    ):
        super().__init__(verbose_level=verbose_level)
        self.params = params
        self.dt = dt
        self.eta = eta if isinstance(eta, Eta) else Eta(*eta) if eta is not None else Eta()
        self.nu = nu if isinstance(nu, Nu) else Nu(*nu) if nu is not None else Nu()
        self._eta_0 = deepcopy(self.eta)
        self._nu_0 = deepcopy(self.nu)
        self.guidance = guidance or Guidance()
        self.navigation = navigation or Navigation()
        self.control = control or Control()
        self.actuators = actuators or []
        self.initial_geometry = VESSEL_GEOMETRY(self.params.loa, self.params.beam)
        self.name = name
        self.mmsi = mmsi

    @abstractmethod
    def __dynamics__(self, tau_actuators:np.ndarray, current:Current, wind:Wind, *args, **kwargs) -> np.ndarray:
        pass

    def step(self, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List["IVessel"], *args, control_commands=None, **kwargs) -> Tuple[List, float, bool, bool, Dict, bool]:
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
        measurements, info = self.navigation(self.eta.to_numpy(), self.nu.to_numpy(), current, wind, obstacles, target_vessels) # obs = (eta_m, nu_m, current_m, wind_m, obstacles_m, target_vessels_m)

        # control_commands can be devised by an RL agent for example.
        if control_commands is None:
            ## Guidance: Get desired states
            eta_des, nu_des = self.guidance(**measurements)
            ## Control: Generate action to track desired states
            control_commands = self.control(eta_des, nu_des, **measurements)

        # Actuators
        tau_actuators = np.zeros((6,), float)
        for actuator, control_command in zip(self.actuators, control_commands):
            tau_actuators = tau_actuators + actuator.dynamics(control_command, self.nu.to_numpy(), current, self.dt)
        # print("tau_a: ", tau_actuators)

        # USV Dynamics
        nu_dot = self.__dynamics__(tau_actuators, current, wind, *args, **kwargs)

        # Forward Euler integration of nu
        nu = Euler(self.nu.to_numpy(), nu_dot, self.dt)

        # Forward Euler integration of eta
        eta_dot = self.eta_dot(nu)
        eta = Euler(self.eta.to_numpy(), eta_dot, self.dt)
        
        self.eta, self.nu = Eta(*eta), Nu(*nu)
        return (measurements, 0, False, False, {}, False)
    
    def reset(self):
        for actuator in self.actuators:
            actuator.reset()

        self.guidance.reset()
        self.navigation.reset()
        self.control.reset()
        self.eta = deepcopy(self._eta_0)
        self.nu = deepcopy(self._nu_0)

    def eta_dot(self, nu:np.ndarray):
        p_dot   = np.matmul( Rzyx(*self.eta.to_list()[3:6]), nu[0:3] ) 
        v_dot   = np.matmul( Tzyx(*self.eta.to_list()[3:5]), nu[3:6] )
        return np.concatenate([p_dot, v_dot])

    def __plot__(self, ax, *args, **kwargs):
        """
        x = East
        y = North
        z = -depth
        """
        if isinstance(ax, Axes3D):
            ax.plot3D(*self.geometry_for_3D_plot, *args, **kwargs)
        else:
            ax.plot(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax

    def __scatter__(self, ax, *args, **kwargs):
        if isinstance(ax, Axes3D):
            ax.scatter3D(*self.geometry_for_3D_plot, *args, **kwargs)
        else:
            ax.scatter(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax

    def __fill__(self, ax, *args, **kwargs):
        if isinstance(ax, Axes3D):
            verts = [list(zip(*self.geometry_for_3D_plot))]
            poly = Poly3DCollection(verts, *args, **kwargs)
            ax.add_collection3d(poly)
        else:
            ax.fill(*self.geometry_for_2D_plot, *args, **kwargs)
        return ax
    
    def get_geometry_from_pose(self, eta:Eta) -> np.ndarray:
        return Rzyx(*eta.rpy) @ self.initial_geometry + eta.to_numpy()[0:3].reshape(3, 1)

    def get_geometry_in_frame(self, eta:Eta) -> np.ndarray:
        """
        get geometry in a frame specified by nedrpy
        """
        return Rzyx(*eta.rpy).T @ (Rzyx(*self.eta.rpy) @ self.initial_geometry - eta.to_numpy()[0:3].reshape(3, 1) + self.eta.to_numpy()[0:3].reshape(3, 1))

    @property
    def geometry(self) -> np.ndarray:
        return self.get_geometry_from_pose(self.eta)
    
    @property
    def geometry_for_3D_plot(self) -> list[np.ndarray, np.ndarray]:
        return self.geometry[1, :], self.geometry[0, :], -self.geometry[2, :]
    
    @property
    def geometry_for_2D_plot(self) -> list[np.ndarray, np.ndarray]:
        return self.geometry_for_3D_plot[0:2]
    
    @property
    def n_actuators(self) -> int:
        return len(self.actuators)
    
@dataclass
class TestVesselParams:
    loa:float=10
    beam:float=3


class TestVessel(IVessel):
    def __init__(
            self,
            params,
            dt,
            eta:Eta,
            nu:Nu,
            *args,
            **kwargs
    ):
        super().__init__(params=params, dt=None, eta=eta, nu=nu, *args, **kwargs)

    def __dynamics__(self, *args, **kwargs):
        return super().__dynamics__(*args, **kwargs)

def show_2D_vessel() -> None:
    import matplotlib.pyplot as plt
    params = TestVesselParams()
    vessel = TestVessel(params, None, Eta(n=5, e=10, roll=0.5, pitch=0.5, yaw=1.2), Nu(u=1, v=0.1, r=0.0))
    ax = vessel.fill(c='blue')
    vessel.plot(ax=ax, c='red')
    vessel.scatter(ax=ax, c='black')
    ax.set_aspect('equal')
    plt.show()

def interactive_3D_vessel() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from python_vehicle_simulator.vehicles.vessel import TestVessel
    from python_vehicle_simulator.states.states import Eta, Nu
    from math import pi

    # Initial vessel state
    eta = Eta(n=0, e=0, d=0, roll=0, pitch=0, yaw=0)
    nu = Nu(u=0, v=0, w=0, p=0, q=0, r=0)
    params = TestVesselParams()
    vessel = TestVessel(params, None, eta, nu)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Initial plot
    vessel.fill(ax=ax, facecolor='blue')
    vessel.plot(ax=ax, c='red')
    vessel.scatter(ax=ax, c='black')
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-6, 6])
    ax.set_aspect('equal')

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_roll = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_pitch = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_yaw = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    s_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=0)
    s_pitch = Slider(ax_pitch, 'Pitch', -90, 90, valinit=0)
    s_yaw = Slider(ax_yaw, 'Yaw', -180, 180, valinit=0)

    def update(val):
        vessel.eta.roll = pi*s_roll.val/180
        vessel.eta.pitch = pi*s_pitch.val/180
        vessel.eta.yaw = pi*s_yaw.val/180
        ax.cla()
        vessel.fill(ax=ax, facecolor='blue')
        vessel.plot(ax=ax, c='red')
        vessel.scatter(ax=ax, c='black')
        
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_zlim([-6, 6])
        ax.set_aspect('equal')
        plt.draw()

    s_roll.on_changed(update)
    s_pitch.on_changed(update)
    s_yaw.on_changed(update)

    plt.show()

def test_geometry_in_target_body_frame() -> None:
    import matplotlib.pyplot as plt
    params = TestVesselParams()
    vessel = TestVessel(params, None, Eta(n=5, e=10, roll=0., pitch=0., yaw=2), Nu(u=1, v=0.1, r=0.0))
    tv = TestVessel(params, None, Eta(n=5, e=20, roll=0., pitch=0., yaw=-0.5), Nu(u=1, v=0.1, r=0.0))
    tv_geom = tv.get_geometry_in_frame(vessel.eta)


    ax = vessel.fill(c='blue')
    tv.fill(ax=ax, c='green')
    ax.set_aspect('equal')
    plt.show()

    vessel.eta = Eta() # Plot in vessel frame
    ax = vessel.fill(c='blue')
    ax.plot(tv_geom[1, :], tv_geom[0, :], c='green')
    vessel.plot(ax=ax, c='red')
    vessel.scatter(ax=ax, c='black')
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # show_2D_vessel()
    # interactive_3D_vessel()
    test_geometry_in_target_body_frame()