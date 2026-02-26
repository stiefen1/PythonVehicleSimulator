import numpy as np, numpy.typing as npt
from typing import List, Tuple, Literal
from python_vehicle_simulator.utils.math_fn import Rzyx
class Eta:
    def __init__(self, n:float=0.0, e:float=0.0, d:float=0.0, roll:float=0.0, pitch:float=0.0, yaw:float=0.0):
        """
            See "Handbook of Marine Craft Hydrodynamics and Motion Control - 2011 - Fossen" page 19
            psi stored in radians
        """
        self.n = n          # x
        self.e = e          # y
        self.d = d          # z (depth)
        self.roll = roll    # phi
        self.pitch = pitch  # theta
        self.yaw = yaw      # psi

    def to_list(self) -> List:
        return [self.n, self.e, self.d, self.roll, self.pitch, self.yaw]
    
    def __getitem__(self, idx:int) -> float:
        return self.to_list()[idx]

    def to_numpy(self, dofs: Literal[3, 6] = 6) -> np.ndarray:
        match dofs:
            case 3:
                return np.array(self.neyaw, float)
            case 6:
                return np.array(self.to_list(), float)
            case _:
                raise ValueError(f"dofs must be 3 or 6, got dofs={dofs}")
    
    def zeros_inplace(self) -> None:
        self.n, self.e, self.d, self.roll, self.pitch, self.yaw = 0., 0., 0., 0., 0., 0.

    @staticmethod
    def from_ndofs_states(states: Tuple | npt.NDArray | List, ndofs: Literal[3, 6]) -> "Eta":
        return Eta(n=states[0], e=states[1], yaw=states[2]) if ndofs==3 else Eta(*states[0:6])
    
    @property
    def rpy(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)
    
    @property
    def ned(self) -> Tuple[float, float, float]:
        return (self.n, self.e, self.d)
    
    @property
    def neyaw(self) -> Tuple[float, float, float]:
        return (self.n, self.e, self.yaw)


class Nu:
    def __init__(self, u:float=0.0, v:float=0.0, w:float=0.0, p:float=0.0, q:float=0.0, r:float=0.0):
        """
            See "Handbook of Marine Craft Hydrodynamics and Motion Control - 2011 - Fossen" page 19
            pqr stored in radians, uvr in body frame
        """
        self.u = u  # surge speed
        self.v = v  # sway speed
        self.w = w  # heave speed
        self.p = p  # roll rate
        self.q = q  # pitch rate
        self.r = r  # yaw rate

    def to_list(self) -> List:
        return [self.u, self.v, self.w, self.p, self.q, self.r]

    def __getitem__(self, idx:int) -> float:
        return self.to_list()[idx]

    def to_numpy(self, dofs: Literal[3, 6] = 6) -> np.ndarray:
        match dofs:
            case 3:
                return np.array(self.uvr, float)
            case 6:
                return np.array(self.to_list(), float)
            case _:
                raise ValueError(f"dofs must be 3 or 6, got dofs={dofs}")
    
    def to_ned(self, roll:float, pitch:float, yaw:float) -> np.ndarray:
        return (Rzyx(roll, pitch, yaw) @ self.to_numpy()[0:3])
    
    def zeros_inplace(self) -> None:
        self = Nu()

    @staticmethod
    def from_ndofs_states(states: Tuple | npt.NDArray | List, ndofs: Literal[3, 6]) -> "Nu":
        return Nu(u=states[3], v=states[4], r=states[5]) if ndofs==3 else Nu(*states[6:12])

    @property
    def uvw(self) -> Tuple[float, float, float]:
        return (self.u, self.v, self.w)
    
    @property
    def pqr(self) -> Tuple[float, float, float]:
        return (self.p, self.q, self.r)

    @property
    def uvr(self) -> Tuple[float, float, float]:
        return (self.u, self.v, self.r)

class States:
    def __init__(self, eta:Eta, nu:Nu):
        self.eta = eta
        self.nu = nu

    def to_list(self) -> List[float]:
        return self.eta.to_list() + self.nu.to_list()
    
    def __getitem__(self, idx:int) -> float:
        return self.to_list()[idx]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), float)
    
def test():
    eta = Eta(n=30, e=10, yaw=1.5)
    nu = Nu(u=1, v=0.1, r=0.1)
    x = States(eta, nu)
    print(x.to_numpy())
    print(nu.to_ned(0, 0, 1.2))

if __name__ == "__main__":
    test()