import numpy as np
from typing import List
from python_vehicle_simulator.utils.transformation import R_body_to_ned

class Eta:
    def __init__(self, n:float=0.0, e:float=0.0, psi:float=0.0):
        """psi stored in radians"""
        self.n = n
        self.e = e
        self.psi = psi

    def to_list(self) -> List:
        return [self.n, self.e, self.psi]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list()).reshape(3, 1)

class Nu:
    def __init__(self, u:float=0.0, v:float=0.0, r:float=0.0):
        """r stored in radians, uvr in body frame"""
        self.u = u
        self.v = v
        self.r = r

    def to_list(self) -> List:
        return [self.u, self.v, self.r]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list()).T
    
    def to_ned(self, psi:float) -> np.ndarray:
        return (R_body_to_ned(psi, dim=3) @ self.to_numpy()).reshape(3, 1)


class States:
    def __init__(self, eta:Eta, nu:Nu):
        self.eta = eta
        self.nu = nu

    def to_list(self) -> List[float]:
        return self.eta.to_list() + self.nu.to_list()

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list()).reshape(6, 1)
    
def test():
    eta = Eta(30, 10, 1.5)
    nu = Nu(1, 0.1, 0.1)
    x = States(eta, nu)
    print(x.to_numpy())
    print(nu.to_ned(1.2))

if __name__ == "__main__":
    test()