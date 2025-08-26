import numpy as np
from typing import List, Tuple
from python_vehicle_simulator.lib.gnc import Rzyx
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

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), float).reshape(6, 1)
    
    @property
    def rpy(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)
    
    @property
    def ned(self) -> Tuple[float, float, float]:
        return (self.n, self.e, self.d)

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
        return [self.u, self.v, self.r, self.p, self.q, self.r]

    def __getitem__(self, idx:int) -> float:
        return self.to_list()[idx]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), float).T
    
    def to_ned(self, roll:float, pitch:float, yaw:float) -> np.ndarray:
        return (Rzyx(roll, pitch, yaw) @ self.to_numpy()[0:3]).reshape(3, 1)

    @property
    def uvw(self) -> Tuple[float, float, float]:
        return (self.u, self.v, self.w)
    
    @property
    def pqr(self) -> Tuple[float, float, float]:
        return (self.p, self.q, self.r)

class States:
    def __init__(self, eta:Eta, nu:Nu):
        self.eta = eta
        self.nu = nu

    def to_list(self) -> List[float]:
        return self.eta.to_list() + self.nu.to_list()
    
    def __getitem__(self, idx:int) -> float:
        return self.to_list()[idx]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), float).reshape(12, 1)
    
def test():
    eta = Eta(n=30, e=10, yaw=1.5)
    nu = Nu(u=1, v=0.1, r=0.1)
    x = States(eta, nu)
    print(x.to_numpy())
    print(nu.to_ned(0, 0, 1.2))

if __name__ == "__main__":
    test()