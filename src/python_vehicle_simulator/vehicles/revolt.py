#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Based on https://dev.azure.com/GTR-DigitalAssurance/ReVolt/_git/FMUs?path=/ReVoltShipSimulatorOSP/src/revolt.m

"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Union
from python_vehicle_simulator.lib.control import PIDpolePlacement
from python_vehicle_simulator.lib.physics import m2c, crossFlowDrag, GRAVITY, RHO, addedMassSurge
from python_vehicle_simulator.utils.math_fn import Smtrx, Hmtrx, Rzyx, sat, Tzyx
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec, DEG2RAD
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.states.states import Nu, Eta
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.actuator import IActuator

@dataclass
class RevoltThrusterParameters:
    ## Propellers       
    T_n: float = 0.1                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: np.ndarray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: np.ndarray = field(init=False)                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: np.ndarray = field(init=False)                       # Max positive force, one propeller
    f_min: np.ndarray = field(init=False)                       # Max negative force, one propeller
    n_max: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    n_min: np.ndarray = field(init=False)                       # Min (negative) propeller speed
    n_dot_max: float = field(init=False)
    xy: np.ndarray = field(init=False) # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    max_radians_per_step: np.ndarray = field(init=False)
    max_newton_per_step: np.ndarray = field(init=False)

    def __post_init__(self):
        self.k_pos: np.ndarray = np.array([2.7e-3, 2.7e-3, 1.518e-3])    # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: np.ndarray = np.array([2.7e-3, 2.7e-3, 6.172e-4])     # Negative Bollard, one propeller
        self.f_max: np.ndarray = np.array([25, 25, 14])                  # Max positive force, one propeller
        self.f_min: np.ndarray = np.array([-25, -25, -6.1]) 
        self.n_max = np.sqrt(self.f_max / self.k_pos)
        self.n_min = -np.sqrt(-self.f_min / self.k_neg)
        self.xy = np.array([[-1.65, -0.15], [-1.65, 0.15], [1.15, 0.0]])
        self.max_radians_per_step = np.array([np.pi/6, np.pi/6, np.pi/36])
        self.max_newton_per_step = np.array([10.0, 10.0, 4.0])


@dataclass
class RevoltParameters:
    loa: float = 3.0                                    # Length Over All (m)
    beam: float = 0.72                                  # Beam (m)
    initial_draft:float = 0.25                          # Initial draft 

    R44: float = 0.36 * beam                             # radii of gyration (m)
    R55: float = 0.26 * loa
    R66: float = 0.26 * loa

    ## Time constants
    T_yaw: float = 3.0                                  # Time constant in yaw (s)

    ## Mass & Payload
    m: float = 157.0                                    # Mass (kg) from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.63)
    mp: float = 100.0                                   # Payload mass (kg)
    m_tot: float = m+mp
    rp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))          # Location of payload (m)
    rg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))      # CG for hull and corrected for payload in __post_init__(m)
    S_rg: np.ndarray = field(init=False)                # Skew matrix for rg
    H_rg: np.ndarray = field(init=False)                # Homogeneous transformation for rg
    S_rp: np.ndarray = field(init=False)                # Skew matrix for rp

    ## Max speed
    u_max: float = knot_to_m_per_sec(3)                 # Max forward speed (m/s)

    ## Pontoon
    num_pontoons: int = 1
    B_pont: float = beam                                # Beam of one pontoon (m)
    y_pont: float = 0                                   # Distance from centerline to waterline centroid (m)
    Cw_pont: float = 2.0/(loa*beam)                     # Waterline area coefficient (-)
    Cb_pont: float = (m/RHO) / (loa*beam*initial_draft) # Block coefficient

    ## Inertia dyadic, volume displacement and draft
    nabla: float = (m+mp)/RHO                    # volume of volume displaced
    Ig_cg: np.ndarray = field(init=False)               # Inertia dyadic at CG
    Ig: np.ndarray = field(init=False)                  # Inertia dyadic
    draft: float = nabla / (num_pontoons * Cb_pont * B_pont * loa)


    ## MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
    ##               O3       Ig ]
    MRB_CG: np.ndarray = field(init=False)              # Rigid-body mass matrix at CG
    MRB: np.ndarray = field(init=False)                 # Rigid-body mass matrix

    def __post_init__(self):
        # Basic calculations
        self.m_tot = self.m + self.mp
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp)
        self.S_rg = Smtrx(self.rg)
        self.H_rg = Hmtrx(self.rg)
        self.S_rp = Smtrx(self.rp)
        self.Ig_cg = self.m * np.diag([self.R44 ** 2, self.R55 ** 2, self.R66 ** 2])
        self.Ig = self.Ig_cg - self.m * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp
        
        # Mass matrices
        self.MRB_CG = np.zeros((6, 6))
        self.MRB_CG[0:3, 0:3] = (self.m + self.mp) * np.identity(3)
        self.MRB_CG[3:6, 3:6] = self.Ig
        self.MRB = self.H_rg.T @ self.MRB_CG @ self.H_rg
        
        # Hydrodynamics added mass
        self.Xudot = -addedMassSurge(self.m, self.loa)
        self.Yvdot = -4.0 * self.m
        self.Zwdot = -1.0 * self.m
        self.Kpdot = -0.2 * self.Ig[0, 0]
        self.Mqdot = -0.8 * self.Ig[1, 1]
        self.Nrdot = -1.7 * self.Ig[2, 2]
        
        # System mass matrix
        self.MA = -np.diag([self.Xudot, self.Yvdot, self.Zwdot, self.Kpdot, self.Mqdot, self.Nrdot])
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)
        
        # Hydrostatic quantities
        self.Aw_pont = self.Cw_pont * self.loa * self.B_pont
        self.I_T = (
            self.num_pontoons
            * (1 / 12)
            * self.loa
            * self.B_pont ** 3
            * (6 * self.Cw_pont ** 3 / ((1 + self.Cw_pont) * (1 + 2 * self.Cw_pont)))
            + self.num_pontoons * self.Aw_pont * self.y_pont ** 2
        )
        self.I_L = 0.8 * self.num_pontoons * (1 / 12) * self.B_pont * self.loa ** 3
        self.KB = (1 / 3) * (5 * self.draft / 2 - 0.5 * self.nabla / (self.loa * self.B_pont))
        self.BM_T = self.I_T / self.nabla
        self.BM_L = self.I_L / self.nabla
        self.KM_T = self.KB + self.BM_T
        self.KM_L = self.KB + self.BM_L
        self.KG = self.draft - self.rg[2]
        self.GM_T = self.KM_T - self.KG
        self.GM_L = self.KM_L - self.KG

        self.G33 = RHO * GRAVITY * (self.num_pontoons * self.Aw_pont)
        self.G44 = RHO * GRAVITY * self.nabla * self.GM_T
        self.G55 = RHO * GRAVITY * self.nabla * self.GM_L
        self.G_CF = np.diag([0, 0, self.G33, self.G44, self.G55, 0])           # I HAVE CHANGED SIGN OF G33 TO (-) AND NOW IT WORKS FINE -> WTFFFF ? CHANGING THE SIGN IN ETA ALSO WORKS
        self.LCF = -0.2
        self.H = Hmtrx(np.array([self.LCF, 0.0, 0.0])) # Transform G_CF from CF to CO
        self.Gmtrx = self.H.T @ self.G_CF @ self.H

        # Natural frequencies
        self.w3 = math.sqrt(self.G33 / self.M[2, 2])
        self.w4 = math.sqrt(self.G44 / self.M[3, 3])
        self.w5 = math.sqrt(self.G55 / self.M[4, 4])

        # Linear damping terms
        self.Xu = -24.4 * GRAVITY / self.u_max
        self.Yv = 0.0
        self.Zw = -2 * 0.3 * self.w3 * self.M[2, 2]
        self.Kp = -2 * 0.2 * self.w4 * self.M[3, 3]
        self.Mq = -2 * 0.4 * self.w5 * self.M[4, 4]
        self.Nr = -self.M[5, 5] / self.T_yaw

        self.D = -np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])


# Class Vehicle
class Revolt(IVessel):
    def __init__(
        self, 
        params:RevoltParameters, # Could be moved in the construct since it will never change
        dt:float,
        tau_X = 120, # Surge force, maintained constant, generated by the pilot. No control on it
        eta:Union[Eta, Tuple]=None,
        nu:Union[Nu, Tuple]=None,
        guidance:IGuidance=None,
        navigation:INavigation=None,
        control:IControl=None,
        actuators:List[IActuator]=None,
        name:str="Otter USV (see 'otter.py' for more details)",
        mmsi:str=None
    ):
        super().__init__(params, dt=dt, eta=eta, nu=nu, guidance=guidance, navigation=navigation, control=control, actuators=actuators, name=name, mmsi=mmsi)

        self.tauX = tau_X  # surge force (N)

        # Heading autopilot
        self.e_int = 0  # integral state
        self.wn = 2.5   # PID pole placement
        self.zeta = 1

        # Reference model
        self.r_max = 10 * DEG2RAD  # maximum yaw rate
        self.psi_d = 0   # angle, angular rate and angular acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = 0.5  # desired natural frequency in yaw
        self.zeta_d = 1  # desired relative damping ratio

        # self.t = 0


    def __dynamics__(self, tau_actuators:np.ndarray, current:Current, wind:Wind) -> np.ndarray:
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """
        eta, nu = self.eta.to_numpy(), self.nu.to_numpy()

        u_c = current.u(eta[5])
        v_c = current.v(eta[5])

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.params.m_tot * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.params.Ig, nu[3:6]))
        CRB = self.params.H_rg.T @ CRB_CG @ self.params.H_rg  # transform CRB from CG to CO

        CA = m2c(self.params.MA, nu_r)
        # Cancel the Munk moment in yaw
        CA[5, 0] = 0  
        CA[5, 1] = 0 
        CA[0, 5] = 0
        CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(*eta[3:6])
        f_payload = np.matmul(R.T, np.array([0, 0, self.params.mp * GRAVITY], float))              
        m_payload = np.matmul(self.params.S_rp, f_payload)
        g_0 = np.array([ f_payload[0],f_payload[1],f_payload[2], 
                         m_payload[0],m_payload[1],m_payload[2] ])

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.params.D, nu_r)
        tau_damp[5] = self.params.D[5] * (1 + 10 * abs(nu_r[5]) * nu_r[5]) # tau_damp[5] - 10 * self.params.D[5, 5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.params.loa, self.params.B_pont, self.params.draft, nu_r)
        sum_tau = (
            tau_actuators
            + tau_damp
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.params.Gmtrx, eta)
            + g_0
        )

        # USV dynamics
        nu_dot = np.matmul(self.params.Minv, sum_tau)

        return nu_dot
    


    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs.
        """
        n1 = 100  # rad/s
        n2 = 80

        if t > 30 and t < 100:
            n1 = 80
            n2 = 120
        else:
            n1 = 0
            n2 = 0

        u_control = np.array([n1, n2], float)

        return u_control


if __name__ == "__main__":
    data = RevoltParameters()
    print(data)