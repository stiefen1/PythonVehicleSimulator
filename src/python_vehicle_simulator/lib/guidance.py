#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guidance algorithms.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
from abc import ABC, abstractmethod
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.visualizer.drawable import IDrawable
from typing import Tuple, List
from matplotlib.axes import Axes
class IGuidance(IDrawable, ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)

    def __call__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)

    @abstractmethod
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return eta, nu, {}
    
    @abstractmethod
    def reset(self) :
        pass

    def __plot__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax

class Guidance(IGuidance):
    def __init__(
            self,
            *args,
            desired_heading:float=0.0,
            desired_speed:float=1.0,
            **kwargs
    ):
        self.desired_heading = desired_heading
        self.desired_speed = desired_speed
        super().__init__(*args, **kwargs)

    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0, 0, 0, 0, 0, self.desired_heading], float), np.array([self.desired_speed, 0, 0, 0, 0, 0], float), {}

    def reset(self):
        pass

class PathFollowingGuidance(IGuidance):
    def __init__(
            self,
            path:PWLPath,
            horizon:int,
            dt:float,
            *args,
            desired_speed:float = 0.5,
            final_heading:float = 0.0,
            **kwargs
    ):
        self.desired_speed = desired_speed
        self.final_heading = final_heading
        self.path = path
        self.horizon = horizon
        self.dt = dt
        super().__init__(*args, **kwargs)

    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return None, None, {
            'path': self.path.get_target_wpts_from(eta[0], eta[1], self.desired_speed*self.dt, self.horizon+1, final_heading=self.final_heading)
        }

    def reset(self):
        pass
    
