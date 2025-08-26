from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.states.states import Eta, Nu
from python_vehicle_simulator.utils.transformation import R_body_to_ned
import numpy as np


# Coordinates in NED frame. Psi is clockwise positive
#   \eta  = [N, E, \psi]  
#   \nu = [u, v, r]
#   x = [\eta, \nu]

VESSEL_GEOMETRY = lambda loa, beam: np.array([
    (loa/2,     0),
    (loa/4,     beam/2),
    (-loa/2,    beam/2),
    (-loa/2,    -beam/2),
    (loa/4,     -beam/2),
    (loa/2,     0)
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
        self.geometry = R_body_to_ned(eta.psi, dim=2) @ VESSEL_GEOMETRY(loa, beam) + eta.to_numpy()[0:2]

    @abstractmethod
    def __dynamics__(self, *args, **kwargs) -> Tuple[Eta, Nu]:
        pass

    def __plot__(self, ax, *args, **kwargs):
        ax.plot(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax

    def __scatter__(self, ax, *args, **kwargs):
        ax.scatter(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax

    def __fill__(self, ax, *args, **kwargs):
        ax.fill(self.geometry[1, :], self.geometry[0, :], *args, **kwargs)
        return ax
    
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

def test_vessel() -> None:
    import matplotlib.pyplot as plt
    vessel = TestVessel(10, 3, Eta(5, 10, 1.2), Nu(1, 0.1, 0.0))
    ax = vessel.fill(c='blue')
    vessel.plot(ax=ax, c='red')
    vessel.scatter(ax=ax, c='black')
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    test_vessel()