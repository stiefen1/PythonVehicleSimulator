#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control methods.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
from python_vehicle_simulator.utils.math_fn import ssa, Rzyx
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.actuator import Thruster
from typing import List, Tuple
from abc import ABC, abstractmethod
from math import sqrt
import keyboard

class IControl(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.prev = {'u': None, 'info': None}

    def __call__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> List[np.ndarray]:
        u, info = self.__get__(eta_des, nu_des, eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)
        self.prev = {'u': u, 'info': info}
        return u, info

    @abstractmethod
    def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> List[np.ndarray]:
        return [], {}
    
    @abstractmethod
    def reset(self):
        pass

class Control(IControl):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> List[np.ndarray]:
        return super().__get__(eta_des, nu_des, eta, nu, current, wind, obstacles, target_vessels, *args, **kwargs)

    def reset(self):
        pass

class UserInputControlTwoTrusters(Control):
    """

    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        speeds = {'zero': 0, 'low': 33, 'medium': 66, 'high': 100}
        self.hashmap = {
            '1': (speeds['zero'], speeds['zero']),
            '2':(speeds['low'], speeds['zero']),
            '3': (speeds['medium'], speeds['zero']),
            '4': (speeds['high'], speeds['zero']),
            'q': (speeds['zero'], speeds['low']),
            'w': (speeds['low'], speeds['low']),
            'e': (speeds['medium'], speeds['low']),
            'r': (speeds['high'], speeds['low']),
            'a': (speeds['zero'], speeds['medium']),
            's': (speeds['low'], speeds['medium']),
            'd': (speeds['medium'], speeds['medium']),
            'f': (speeds['high'], speeds['medium']),
            'y': (speeds['zero'], speeds['high']),
            'x': (speeds['low'], speeds['high']),
            'c': (speeds['medium'], speeds['high']),
            'v': (speeds['high'], speeds['high'])
        }
        self.u = [np.array([0.0]), np.array([0.0])]

    def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> List[np.ndarray]:
        for key in self.hashmap.keys():
            if keyboard.is_pressed(key):
                self.u = [np.array(ui) for ui in self.hashmap[key]]
        return self.u, {}

class HeadingAutopilotTwoThrusters(IControl):
    def __init__(
            self,
            actuators:Tuple[Thruster, Thruster],
            dt:float,
            *args,
            tau_X:float = 120,
            wn:float=2.5, 
            zeta:float=1.0, 
            wn_d:float=0.5, 
            zeta_d:float=1.0, 
            r_max:float=10*DEG2RAD,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.actuators = actuators
        self.tau_X = tau_X
        self.e_int = 0.0 # Integral error
        self.wn = wn  # PID natural frequency
        self.zeta = zeta  # PID natural relative damping factor
        self.wn_d = wn_d  # reference model natural frequency
        self.zeta_d = zeta_d  # reference model relative damping factor
        self.r_max = r_max # Maximum yaw rate
        self.psi_d = 0.0 # angle, angular rate and angular acc. states
        self.r_d = 0.0
        self.a_d = 0.0
        self.dt = dt
        B = (0.02216 / 2) * np.array([[1, 1], [0.395, -0.395]])
        self.Binv = np.linalg.inv(B)

    def __get__(self, eta_des:np.ndarray, nu_des:np.ndarray, eta:np.ndarray, nu:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> List[np.ndarray]:
        tau_x, tau_n = self.headingAutopilot(eta, nu, eta_des[5])
        # print("tau_d: ", tau_x, tau_n)
        # u = self.two_thrusters_control_allocation(tau_x, tau_n)
        u = self.controlAllocation(tau_x, tau_n)
        # print(u)
        return u, {}
    
    def reset(self):
        self.psi_d = 0.0
        self.r_d = 0.0
        self.a_d = 0.0
        self.e_int = 0.0
        for actuator in self.actuators:
            actuator.reset()

    def headingAutopilot(self, eta:np.ndarray, nu:np.ndarray, psi_setpoint:float):
        """
        u = headingAutopilot(eta,nu,dt) is a PID controller
        for automatic heading control based on pole placement.

        psi_setpoint is in radians.

        tau_N = (T/K) * a_d + (1/K) * rd
                - Kp * ( ssa( psi-psi_d ) + Td * (r - r_d) + (1/Ti) * z )

                
        Setpoint psi_ref ----> Reference Model ----> desired yaw psi_d (that keep reasonable jerk value) ----> PID Controller designed based on Pole Placement

        """
        psi = eta[5]  # yaw angle
        r = nu[5]  # yaw rate
        e_psi = psi - self.psi_d  # yaw angle tracking error --> PSI_D IS THE DESIRED VALUE ACCORDING TO THE REFERENCE MODEL
        e_r = r - self.r_d  # yaw rate tracking error
        psi_ref = psi_setpoint  # yaw angle setpoint --> THE ACTUAL VALUE WE WANT TO REACH

        

        m = 41.4  # moment of inertia in yaw including added mass
        T = 1
        K = T / m
        d = 1 / K
        k = 0

        # PID feedback controller with 3rd-order reference model
        tau_X = self.tau_X

        [tau_N, self.e_int, self.psi_d, self.r_d, self.a_d] = PIDpolePlacement(
            self.e_int,
            e_psi,
            e_r,
            self.psi_d,
            self.r_d,
            self.a_d,
            m,
            d,
            k,
            self.wn_d,
            self.zeta_d,
            self.wn,
            self.zeta,
            psi_ref,
            self.r_max,
            self.dt,
        )

        return tau_X, tau_N

    def two_thrusters_control_allocation(self, tau_x:float, tau_n:float) -> np.ndarray: #, y_1:float, y_2:float, k_pos_1:float, k_neg_1:float, k_pos_2:float, k_neg_2:float) 
        """
        y_1, y_2: signed position of actuators in sway
        """
        # self.actuators[0].xy[1], self.actuators[1].xy[1], self.actuators[0].k_pos, self.actuators[0].k_neg, self.actuators[1].k_pos, self.actuators[1].k_neg
        ## First compute f1 and f2, i.e. force to be generated by each thruster
        f_2 = (tau_x*self.actuators[0].xy[1]-tau_n)/(self.actuators[0].xy[1]-self.actuators[1].xy[1])
        f_1 = tau_x - f_2

        ## Then map these forces to actual propeller speed, taking rotation direction into account
        n_1 = sqrt(f_1/self.actuators[0].k_pos) if f_1>=0 else -sqrt(-f_1/self.actuators[0].k_neg)
        n_2 = sqrt(f_2/self.actuators[1].k_pos) if f_2>=0 else -sqrt(-f_2/self.actuators[1].k_neg)
        return np.array([n_1, n_2])

    def controlAllocation(self, tau_X, tau_N):
        """
        [n1, n2] = controlAllocation(tau_X, tau_N)

        This formulation does not allow for negative force from the thruster, or at least is not consistent because Binv is computed 
        using k_pos in cases
        """
        tau = np.array([tau_X, tau_N])  # tau = B * u_alloc
        u_alloc = np.matmul(self.Binv, tau)  # u_alloc = inv(B) * tau

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n1 = np.sign(u_alloc[0]) * sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * sqrt(abs(u_alloc[1]))

        return n1, n2

# SISO PID pole placement
def PIDpolePlacement(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    m,
    d,
    k,
    wn_d,
    zeta_d,
    wn,
    zeta,
    r,
    v_max,
    dt,
):
    """
    PID gains are placed for the system to match a second-order system parametrized by (wn, zeta) with a feedforward term gain k
    The reference model is designed to smooth the desired value
    """

    # PID gains based on pole placement 
    Kp = m * wn ** 2.0 - k          # Equation 12.97 (p.373) with zero acceleration feedback (Km=0)
    Kd = m * 2.0 * zeta * wn - d    # Equation 12.98 (p.373) with zero acceleration feedback (Km=0)
    Ki = (wn / 10.0) * Kp           # Equation 12.103 (p.373)

    # PID control law
    u = -Kp * e_x - Kd * e_v - Ki * e_int

    # Integral error, Euler's method
    e_int += dt * e_x

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, dt)

    return u, e_int, x_d, v_d, a_d


# MIMO nonlinear PID pole placement
def DPpolePlacement(
    e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, eta_ref, sampleTime
):

    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp

    # DP control law - setpoint regulation
    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (
        - np.matmul((R.T @ Kp), e)
        - np.matmul(Kd, nu3)
        - np.matmul((R.T @ Ki), e_int)
    )

    # Low-pass filters, Euler's method
    T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
    x_d += sampleTime * (eta_ref[0] - x_d) / T[0]
    y_d += sampleTime * (eta_ref[1] - y_d) / T[1]
    psi_d += sampleTime * (eta_ref[2] - psi_d) / T[2]

    # Integral error, Euler's method
    e_int += sampleTime * e

    return tau, e_int, x_d, y_d, psi_d

# Heading autopilot - Intergral SMC (Equation 16.479 in Fossen 2021)
def integralSMC(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    T_nomoto,
    K_nomoto,
    wn_d,
    zeta_d,
    K_d,
    K_sigma,
    lam,
    phi_b,
    r,
    v_max,
    sampleTime,
):

    # Sliding surface
    v_r_dot = a_d - 2 * lam * e_v - lam ** 2 * ssa(e_x)
    v_r     = v_d - 2 * lam * ssa(e_x) - lam ** 2 * e_int
    sigma   = e_v + 2 * lam * ssa(e_x) + lam ** 2 * e_int

    #  Control law
    if abs(sigma / phi_b) > 1.0:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * np.sign(sigma) ) / K_nomoto
    else:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * (sigma / phi_b) ) / K_nomoto

    # Integral error, Euler's method
    e_int += sampleTime * ssa(e_x)

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return delta, e_int, x_d, v_d, a_d

# [x_d,v_d,a_d] = refModel3(x_d,v_d,a_d,r,wn_d,zeta_d,v_max,sampleTime) is a 3-order 
# reference  model for generation of a smooth desired position x_d, velocity |v_d| < v_max, 
# and acceleration a_d. Inputs are natural frequency wn_d and relative damping zeta_d.
def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, dt):
    
    # print("InRefModel: ", x_d, v_d, a_d, r, wn_d, zeta_d, v_max, dt)

    # desired "jerk"
    j_d = wn_d**3 * (r -x_d) - (2*zeta_d+1) * wn_d**2 * v_d - (2*zeta_d+1) * wn_d * a_d # Equation 10.27 (p.250)

   # Forward Euler integration
    x_d += dt * v_d             # desired position
    v_d += dt * a_d             # desired velocity
    a_d += dt * j_d             # desired acceleration 
    
    # Velocity saturation
    if (v_d > v_max):
        v_d = v_max
    elif (v_d < -v_max): 
        v_d = -v_max    
    
    return x_d, v_d, a_d




