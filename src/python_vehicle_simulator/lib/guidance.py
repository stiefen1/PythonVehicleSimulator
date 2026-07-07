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
import shapely
class IGuidance(IDrawable, ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        IDrawable.__init__(self, *args, verbose_level=2, **kwargs)
        self.prev = {'eta_des': None, 'nu_des': None, 'states_des': None, 'info': {'term': False}}

    def __call__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        states_des, info = self.__get__(states, current, wind, obstacles, target_vessels, *args, **kwargs)
        self.prev = {'eta_des': states_des[0:6], 'nu_des': states_des[6:12], 'states_des': states_des, 'info': info}
        return states_des, info

    @abstractmethod
    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        return states, {'term': False}
    
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
        return np.array([0, 0, 0, 0, 0, self.desired_heading, self.desired_speed, 0, 0, 0, 0, 0], float), {'term': False}

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

    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        return np.array([self.desired_speed, 0, 0, 0, 0, 0]), {
            'path': self.path.get_target_wpts_from(states[0], states[1], self.desired_speed*self.dt, self.horizon+1, final_heading=self.final_heading)
        }

    def reset(self):
        pass

class GeneralizedLOSGuidance(IGuidance):
    def __init__(
            self,
            path: PWLPath,
            desired_speed: float,
            *args,
            kp: float = 1e-5,
            **kwargs
    ):
        self.path = path
        self.desired_speed = desired_speed
        self.kp = kp
        super().__init__(*args, **kwargs)

    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        line = shapely.LineString(self.path.waypoints)
        p = shapely.Point(float(states[0]), float(states[1]))

        # Closest point on path and local tangent direction.
        s = float(line.project(p))
        p_closest = line.interpolate(s)
        ds = max(1e-3, 0.5 * self.desired_speed)
        s_fwd = min(s + ds, float(line.length))
        p_fwd = line.interpolate(s_fwd)
        if s_fwd == s:
            s_back = max(0.0, s - ds)
            p_back = line.interpolate(s_back)
            t_n = p_closest.x - p_back.x
            t_e = p_closest.y - p_back.y
        else:
            t_n = p_fwd.x - p_closest.x
            t_e = p_fwd.y - p_closest.y

        psi_path = np.arctan2(t_e, t_n)

        # Signed cross-track error and bounded LOS correction in [-pi/2, pi/2].
        e_n = p.x - p_closest.x
        e_e = p.y - p_closest.y
        t_norm = np.hypot(t_n, t_e) + 1e-12
        e_ct = (t_n * e_e - t_e * e_n) / t_norm
        psi_corr = np.arctan(max(0.0, self.kp) * e_ct)
        psi_ref = psi_path - psi_corr

        states_des = np.array(5*[0.0] + [psi_ref, self.desired_speed] + 11*[0.0])
        info = {'term': bool(s >= float(line.length) - 1e-6)}
        return states_des, info

    def reset(self):
        pass
