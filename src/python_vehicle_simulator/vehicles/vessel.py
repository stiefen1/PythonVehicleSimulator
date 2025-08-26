from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.states.states import Eta, Nu
from python_vehicle_simulator.utils.transformation import R_body_to_ned
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Coordinates in NED frame. Psi is clockwise positive
#   \eta  = [N, E, \psi]  
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
            loa:float,
            beam:float,
            eta:Eta,
            nu:Nu,
            *args,
            **kwargs
    ):
        self.loa = loa,
        self.beam = beam
        self.eta = eta
        self.nu = nu
        self.initial_geometry = VESSEL_GEOMETRY(loa, beam)

    @abstractmethod
    def __dynamics__(self, *args, **kwargs) -> Tuple[Eta, Nu]:
        pass

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
    
    def get_geometry_from_pose(self, eta:Eta) -> np.ndarray:
        return R_body_to_ned(*eta.rpy) @ self.initial_geometry + eta.to_numpy()[0:3]

    @property
    def geometry(self) -> np.ndarray:
        return self.get_geometry_from_pose(self.eta)
    
class TestVessel(IVessel):
    def __init__(
            self,
            loa:float,
            beam:float,
            eta:Eta,
            nu:Nu,
            *args,
            **kwargs
    ):
        super().__init__(loa=loa, beam=beam, eta=eta, nu=nu, *args, **kwargs)

    def __dynamics__(self, *args, **kwargs):
        return super().__dynamics__(*args, **kwargs)

def show_2D_vessel() -> None:
    import matplotlib.pyplot as plt
    vessel = TestVessel(10, 3, Eta(n=5, e=10, roll=0.5, pitch=0.5, yaw=1.2), Nu(u=1, v=0.1, r=0.0))
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
    vessel = TestVessel(10, 3, eta, nu)

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


if __name__ == "__main__":
    show_2D_vessel()
    interactive_3D_vessel()