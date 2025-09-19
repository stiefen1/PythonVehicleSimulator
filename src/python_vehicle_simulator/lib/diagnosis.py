from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List

"""

Diagnosis object takes as input:
    state measurements, commands and parameters of the object to diagnose

stores:
    previous states, commands and diagnosis

returns
    new diagnosis as a dictionnary

"""

class IDiagnosis(ABC):
    def __init__(
            self,
            params,
            dt:float,
            *args,
            **kwargs
    ):
        self.initial_params = params # PARAMETERS DEPEND ON WHAT WE WANT TO DIAGNOSE
        self.eta_prev = None
        self.nu_prev = None
        self.commands_prev = None

    def get(self, eta:np.ndarray, nu:np.ndarray, commands:List, *args, **kwargs) -> Dict:
        self.diagnose_prev = self.__get__(eta, nu, commands, *args, **kwargs)
        self.eta_prev = eta.copy()
        self.nu_prev = nu.copy()
        self.commands_prev = commands
        return self.diagnose_prev

    @abstractmethod
    def __get__(self, eta:np.ndarray, nu:np.ndarray, commands:List, *args, **kwargs) -> Dict:
        return {}