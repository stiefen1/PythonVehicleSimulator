from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class IDrawable(ABC):
    def __init__(self, *args, verbose_level:int=0, **kwargs):
        self.verbose_level = verbose_level

    @abstractmethod
    def __plot__(self, ax:Axes, *args, **kwargs) -> Axes:
        pass

    @abstractmethod
    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        pass

    @abstractmethod
    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        pass

    def plot(self, *args, ax:Axes=None, verbose:int=0, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if verbose >= self.verbose_level:
            return self.__plot__(ax, *args, **kwargs)
        else:
            return ax
    
    def scatter(self, *args, ax:Axes=None, verbose:int=0, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if verbose >= self.verbose_level:
            return self.__scatter__(ax, *args, **kwargs)
        else:
            return ax
    
    def fill(self, *args, ax:Axes=None, verbose:int=0, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if verbose >= self.verbose_level:
            return self.__fill__(ax, *args, **kwargs)
        else:
            return ax