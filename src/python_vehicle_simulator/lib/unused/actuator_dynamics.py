from python_vehicle_simulator.visualizer.drawable import IDrawable
from python_vehicle_simulator.lib.dynamics import IDynamics
from typing import Tuple, Optional
from matplotlib.axes import Axes
import numpy.typing as npt, numpy
from abc import abstractmethod



class IActuator(IDrawable):
    def __init__(
            self,
            dynamics: IDynamics,
            x: Tuple,
            u_min: Tuple,
            u_max: Tuple,
            *args,
            **kwargs
    ):
        """
        dynamics:   Dynamics model for the actuator
        x:          Initial state                  (nx,)
        u_min:      Minimum control limits         (nu,)
        u_max:      Maximum control limits         (nu,)
        """
        super().__init__(verbose_level=1, *args, **kwargs)
        self.dynamics = dynamics
        self.x = numpy.array(x)
        self.u_min = numpy.array(u_min)
        self.u_max = numpy.array(u_max)

    def step(self, u: Tuple, theta: Optional[Tuple] = None, disturbance: Optional[Tuple] = None) -> Tuple:
        """
        Execute one actuator step with control input and constraints.
        
        u:          Control commands               (nu,)
        theta:      Parameters (optional)          (np,)
        disturbance: Disturbance inputs (optional) (nd,)
        
        Returns:
            Tuple: Next actuator state
        """
        theta = theta if theta is not None else ()
        disturbance = disturbance if disturbance is not None else ()
        u_feasible = numpy.clip(numpy.array(u), self.u_min, self.u_max)
        return tuple(self.__dynamics__(u_feasible, numpy.array(theta), numpy.array(disturbance)).flatten().tolist())

    @abstractmethod
    def __dynamics__(self, u: npt.NDArray, theta: npt.NDArray, disturbance: npt.NDArray) -> npt.NDArray:
        """
        Actuator dynamics implementation.
        
        u:          Feasible control commands     (nu,)
        theta:      Parameters                    (np,)
        disturbance: Disturbance inputs           (nd,)
        
        Returns:
            npt.NDArray: Next actuator state      (nx,)
        """
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        """
        Plot actuator on given axes.
        
        ax:         Matplotlib axes object
        verbose:    Verbosity level
        
        Returns:
            Axes: Updated axes object
        """
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        """
        Create scatter plot of actuator on given axes.
        
        ax:         Matplotlib axes object
        
        Returns:
            Axes: Updated axes object
        """
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        """
        Create filled plot of actuator on given axes.
        
        ax:         Matplotlib axes object
        
        Returns:
            Axes: Updated axes object
        """
        return ax 



    
