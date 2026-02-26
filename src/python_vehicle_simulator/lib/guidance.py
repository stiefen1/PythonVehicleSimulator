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
from typing import Tuple, List, Dict
from matplotlib.axes import Axes
class IGuidance(IDrawable, ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)
        self.prev = {'eta_des': None, 'nu_des': None, 'info': None}

    def __call__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        states_des, info = self.__get__(states, current, wind, obstacles, target_vessels, *args, **kwargs)
        self.prev = {'eta_des': states_des[0:6], 'nu_des': states_des[6:12], 'info': info}
        return states_des, info

    @abstractmethod
    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        return states, {}
    
    @abstractmethod
    def reset(self) :
        pass

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
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

    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        return np.array([0, 0, 0, 0, 0, self.desired_heading, self.desired_speed, 0, 0, 0, 0, 0], float), {}

    def reset(self):
        pass


