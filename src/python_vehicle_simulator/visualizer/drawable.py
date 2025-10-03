from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple

class IDrawable(ABC):
    def __init__(self, *args, verbose_level:int=0, **kwargs):
        self.verbose_level = verbose_level

    @abstractmethod
    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
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
            for item in self.__dict__.values():
                if isinstance(item, IDrawable):
                    # If object is a drawable, plot all sub objects
                    item.plot(*args, ax=ax, verbose=verbose, **kwargs)
            return self.__plot__(*args, ax=ax, verbose=verbose, **kwargs)
        else:
            return ax
    
    def scatter(self, *args, ax:Axes=None, verbose:int=0, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if verbose >= self.verbose_level:
            for item in self.__dict__.values():
                if isinstance(item, IDrawable):
                    # If object is a drawable, plot all sub objects
                    item.scatter(*args, ax=ax, verbose=verbose, **kwargs)
            return self.__scatter__(ax, *args, **kwargs)
        else:
            return ax
    
    def fill(self, *args, ax:Axes=None, verbose:int=0, **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if verbose >= self.verbose_level:
            for item in self.__dict__.values():
                if isinstance(item, IDrawable):
                    # If object is a drawable, plot all sub objects
                    item.fill(*args, ax=ax, verbose=verbose, **kwargs)
            return self.__fill__(ax, *args, **kwargs)
        else:
            return ax