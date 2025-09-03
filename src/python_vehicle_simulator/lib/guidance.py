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
from typing import Tuple, List

class IGuidance(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    def __call__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.__get__(eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)

    @abstractmethod
    def __get__(self, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return eta, nu
    
    @abstractmethod
    def reset(self) :
        pass

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
        return np.array([0, 0, 0, 0, 0, self.desired_heading], float), np.array([self.desired_speed, 0, 0, 0, 0, 0], float)

    def reset(self):
        pass
