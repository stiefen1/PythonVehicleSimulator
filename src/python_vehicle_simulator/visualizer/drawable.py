from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class IDrawable(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __plot__(self, ax:Axes, *args, **kwargs) -> Axes:
        pass

    @abstractmethod
    def __scatter__(self, *args, **kwargs) -> Axes:
        pass

    @abstractmethod
    def __fill__(self, *args, **kwargs) -> Axes:
        pass

    def plot(self, *args, ax:Axes=None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        return self.__plot__(ax, *args, **kwargs)
    
    def scatter(self, *args, ax:Axes=None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        return self.__scatter__(ax, *args, **kwargs)
    
    def fill(self, *args, ax:Axes=None, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        return self.__fill__(ax, *args, **kwargs)