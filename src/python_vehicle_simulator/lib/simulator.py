from tqdm import tqdm
import numpy as np, time
from python_vehicle_simulator.lib.env import NavEnv

class Simulator:

    def __init__(
            self,
            env:NavEnv,
            *args,
            dt:float=None,
            **kwargs
    ):
        self.dt = dt or env.dt
        self.env = env
        self.env.dt = self.dt

    def run(self, tf:float, *args, **kwargs) -> None:
        """
        Run simulation from 0 to tf with sampling time self.dt
        """
        self.env.reset()
        print("Running simulation..")
        N = int(tf//self.dt) + 1
        for _ in tqdm(np.linspace(0, tf, N)):
            obs, r, term, trunc, info, done = self.env.step(*args, **kwargs)

    def play(self, *args, **kwargs) -> None:
        pass

def test() -> None:
    sim = Simulator(1, None)
    sim.run(100)

if __name__ == "__main__":
    test()
