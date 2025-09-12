import numpy as np
from abc import ABC, abstractmethod

class INoise(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def __get__(self, value:np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    def __call__(self, value:np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.__get__(value, *args, **kwargs)
    
class NoNoise(INoise):
    def __init__(
            self
    ):
        super().__init__()

    def __get__(self, value:np.ndarray, *args, **kwargs) -> np.ndarray:
        return value