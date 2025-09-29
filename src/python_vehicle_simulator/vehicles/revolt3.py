#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Based on https://dev.azure.com/GTR-DigitalAssurance/ReVolt/_git/FMUs?path=/ReVoltShipSimulatorOSP/src/revolt.m -> NOT ANYMORE

Based on experimental data from "Development of a Dynamic Positioning System for the ReVolt Model Ship" (p.61) - https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115

"""

import numpy as np
from math import pi
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
from python_vehicle_simulator.lib.diagnosis import IDiagnosis, Diagnosis
import casadi as ca, math

INF = float('inf')

# xy:Tuple,
# k_pos:float,    # speed-thrust quad. mapping > 0 (n>=0)
# k_neg:float,    # speed-thrust quad. mapping > 0 (n<0)
# n_min:float,    # Min propeller speed (can be negative)
# n_max:float,    # Max propeller speed
# alpha_min:float,
# alpha_max:float,
# *args,
# n_0:float=0.0,  # initial propeller speed (rad/s)
# alpha_0:float=0.0,
# T_a:float=1.0,  # Time constant
# T_n:float=1.0,
# f_max:float=float('inf'),
# f_min:float=-float('inf'),
# orientation:float=0.0,
# **kwargs

@dataclass
class RevoltBowThrusterParams: # front
    ## Propellers       
    T_n: float = 0.3                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 2.7e-3 # 1.518e-3                                     # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 6.172e-4                                     # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 14                                           # Max positive force, one propeller
    f_min: float = 0 # -6.1                                         # Max negative force, one propeller
    n_max: float = math.sqrt(f_max/k_pos)                       # Max (positive) propeller speed
    n_min: float = -math.sqrt(-f_min/k_neg)
    a_max: float = math.pi                                     # Max (positive) propeller speed
    a_min: float = -math.pi                                      # Min (negative) propeller speed
    max_radians_per_step: float = math.pi/6
    max_newton_per_step: float = 10

@dataclass
class RevoltSternThrusterParams: # back
    ## Propellers       
    T_n: float = 0.3                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 2.7e-3                                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 2.7e-3                                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 25                                           # Max positive force, one propeller
    f_min: float = 0 # -25                                      # Max negative force, one propeller
    n_max: float = math.sqrt(f_max/k_pos)                       # Max (positive) propeller speed
    n_min: float = -math.sqrt(-f_min/k_neg)
    a_max: float = math.pi                                      # Max (positive) propeller speed
    a_min: float = -math.pi                                     # Min (negative) propeller speed
    max_radians_per_step: float = math.pi/6
    max_newton_per_step: float = 10

    

@dataclass
class RevoltThrusterParameters:
    ## Propellers       
    # T_n: float = 0.3                                            # Propeller time constant (s)
    # T_a: float = 3.0 
    thrusters: List = field(init=False)                         # Azimuth angle time constant (s) -> Chosen by me
    k_pos: np.ndarray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: np.ndarray = field(init=False)                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: np.ndarray = field(init=False)                       # Max positive force, one propeller
    f_min: np.ndarray = field(init=False)                       # Max negative force, one propeller
    n_max: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    n_min: np.ndarray = field(init=False)
    lba: np.ndarray = field(init=False)                       # Max (positive) propeller speed
    uba: np.ndarray = field(init=False)                         # Min (negative) propeller speed
    xy: np.ndarray = field(init=False) # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    max_radians_per_step: np.ndarray = field(init=False)
    max_newton_per_step: np.ndarray = field(init=False)
    time_constant: np.ndarray = field(init=False)

    def __post_init__(self):
        self.thrusters = [RevoltSternThrusterParams(), RevoltSternThrusterParams(), RevoltBowThrusterParams()]  
        self.k_pos: np.ndarray = np.array([thruster.k_pos for thruster in self.thrusters])    # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: np.ndarray = np.array([thruster.k_neg for thruster in self.thrusters])     # Negative Bollard, one propeller
        self.f_max: np.ndarray = np.array([thruster.f_max for thruster in self.thrusters])                  # Max positive force, one propeller
        self.f_min: np.ndarray = np.array([thruster.f_min for thruster in self.thrusters]) # np.array([-25, -25, -6.1]) 
        self.n_max = np.array([thruster.n_max for thruster in self.thrusters])
        self.n_min = np.array([thruster.n_min for thruster in self.thrusters])
        self.lba = np.array([thruster.a_min for thruster in self.thrusters])                       # Azimuth angles constraints
        self.uba = np.array([thruster.a_max for thruster in self.thrusters])
        self.xy = np.array([[-1.65, -0.15], [-1.65, 0.15], [1.15, 0.0]])
        self.max_radians_per_step = np.array([np.pi/6, np.pi/6, np.pi/36])
        self.max_newton_per_step = np.array([10.0, 10.0, 4.0])
        self.time_constant = np.array([thruster.T_n for thruster in self.thrusters] + [thruster.T_a for thruster in self.thrusters])

        self.Ti = lambda alpha, lx, ly : np.array([
            ca.cos(alpha),
            ca.sin(alpha),
            lx*ca.sin(alpha) - ly * ca.cos(alpha)
        ])

        self.T = lambda a1, a2, a3 : ca.vertcat(
            ca.horzcat(ca.cos(a1), ca.sin(a1), self.xy[0, 0]*ca.sin(a1) - self.xy[0, 1] * ca.cos(a1)),
            ca.horzcat(ca.cos(a2), ca.sin(a2), self.xy[1, 0]*ca.sin(a2) - self.xy[1, 1] * ca.cos(a2)),
            ca.horzcat(ca.cos(a3), ca.sin(a3), self.xy[2, 0]*ca.sin(a3) - self.xy[2, 1] * ca.cos(a3))
        )


@dataclass
class RevoltParameters3DOF:
    Nx: int = 3
    loa: float = 3.0                                    # Length Over All (m) assumed equal to LPP
    beam: float = 0.72                                  # Beam (m)
    initial_draft:float = 0.25                          # Initial draft
    volume:float = 0.268                                # m^3 volume
    volume_iz:float = 0.310                             # m^5 volument moment of inertia

    R44: float = 0.36 * beam                             # radii of gyration (m)
    R55: float = 0.26 * loa
    R66: float = 0.26 * loa

    ## Time constants
    T_yaw: float = 3.0                                  # Time constant in yaw (s)

    ## Mass & Payload
    m: float = 157.0                                    # Mass (kg) from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.63)
    mp: float = 100.0                                   # Payload mass (kg)
    rg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))  
    rp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, 0.0], float))          # Location of payload (m)

    ## State constraints
    lbx: np.ndarray = field(default_factory=lambda: np.array([-INF, -INF, -pi, -1, -1, -pi/6]))
    ubx: np.ndarray = field(default_factory=lambda: np.array([INF, INF, pi, knot_to_m_per_sec(3), 1, pi/6]))


    def __post_init__(self):
        self.m_tot = self.m + self.mp
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp) # corrected center of gravity with payload

        # Basic calculations
        self.Xudot, self.Yvdot, self.Yrdot, self.Nvdot = (-self.volume*RHO*np.array([0.0253, 0.1802, 0.0085 * self.loa, 0.0099 * self.loa**2])).tolist()
        self.Xu, self.Yv, self.Yr, self.Nv, self.Nr = (-np.array([
            0.102 * RHO * self.volume * GRAVITY / self.loa,
            1.212 * GRAVITY / self.loa,
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.056 * RHO * self.volume * np.sqrt(GRAVITY * self.loa),
            0.0601 * RHO * self.volume * np.sqrt(GRAVITY * self.loa)
            ])).tolist()
        self.Iz = self.m_tot / self.volume * self.volume_iz

        ## Inertia Matrix
        MRB = np.array([
            [self.m_tot, 0, 0],
            [0, self.m_tot, self.m_tot*self.rg[0]],
            [0, self.m_tot*self.rg[0], self.Iz]
        ])
        MA = np.array([
            [-self.Xudot, 0, 0],
            [0, -self.Yvdot, -self.Yrdot],
            [0, -self.Yrdot, -self.Nvdot]
        ])
        self.Minv = np.linalg.inv(MA + MRB)

        ## Coriolis-Centripetal Matrix
        self.CRB = lambda nu : np.array([
            [0, 0, -self.m_tot * (self.rg[0]*nu[2] + nu[1])],
            [0, 0, self.m_tot * nu[0]],
            [self.m_tot * (self.rg[0]*nu[2] + nu[1]), -self.m_tot * nu[0], 0]
        ])
        self.CA = lambda nu_r : np.array([
            [0, 0, self.Yvdot * nu_r[1] + self.Yrdot * nu_r[2]],
            [0, 0, -self.Xudot * nu_r[0]],
            [-self.Yvdot * nu_r[1]-self.Yrdot * nu_r[2], self.Xudot * nu_r[0], 0]
        ])
        # self.C = lambda nu : CA(nu) + CRB(nu)

        ## Damping Matrix
        self.D = np.array([
            [-self.Xu, 0, 0],
            [0, -self.Yv, -self.Yr],
            [0, -self.Nv, -self.Nr]
        ])

        # self.dynamics = lambda x, tau : self.Minv @ (
        #     tau -
        #     self.C(x[3:6]) @ x[3:6] +
        #     self.D @ x[3:6]
        # )


# Class Vehicle
class Revolt3DOF(IVessel):
    def __init__(
        self, 
        params:RevoltParameters3DOF, # Could be moved in the construct since it will never change
        dt:float,
        tau_X = 120, # Surge force, maintained constant, generated by the pilot. No control on it
        eta:Union[Eta, Tuple]=None,
        nu:Union[Nu, Tuple]=None,
        guidance:IGuidance=None,
        navigation:INavigation=None,
        control:IControl=None,
        diagnosis:IDiagnosis=None,
        actuators:List[IActuator]=None,
        name:str="Otter USV (see 'otter.py' for more details)",
        mmsi:str=None,
    ):
        super().__init__(params, dt=dt, eta=eta, nu=nu, guidance=guidance, navigation=navigation, control=control, diagnosis=diagnosis, actuators=actuators, name=name, mmsi=mmsi)

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


    def __dynamics__(self, tau_actuators:np.ndarray, current:Current, wind:Wind) -> np.ndarray:
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """
        nu = np.array(self.nu.uvr)
        tau_actuators_3 = np.array([tau_actuators[0], tau_actuators[1], tau_actuators[5]])

        CRB = self.params.CRB(nu)
        CA = self.params.CA(nu)
        C = CRB + CA

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = np.matmul(self.params.D, nu)

        # State derivatives (with dimension)
        sum_tau = (
            tau_actuators_3
            - tau_damp
            - np.matmul(C, nu)
        )

        # USV dynamics
        nu_dot = np.matmul(self.params.Minv, sum_tau)

        return np.array([nu_dot[0], nu_dot[1], 0, 0, 0, nu_dot[2]])

def test() -> None:
    from python_vehicle_simulator.lib.actuator import AzimuthThruster
    vessel = Revolt3DOF(
        params=RevoltParameters3DOF(),
        dt=0.02,
        actuators=[
            AzimuthThruster(xy=(-1.65, -0.15), **vars(RevoltSternThrusterParams())),
            AzimuthThruster(xy=(-1.65, 0.15), **vars(RevoltSternThrusterParams())),
            AzimuthThruster(xy=(1.15, 0.0), **vars(RevoltBowThrusterParams()))
        ]
    )
    params = RevoltThrusterParameters()
    print(params.n_max, params.n_min)

    # [-1.65, -0.15], [-1.65, 0.15], [1.15, 0.0]

def jacobians() -> None:
    params = RevoltParameters3DOF()
    import sympy as sp
    dt = 0.1
    n, e, psi, u, v, r = sp.symbols('n e psi u v r')
    tau_u, tau_v, tau_r = sp.symbols('tau_u tau_v tau_r')
    eta = sp.Matrix([n, e, psi])
    nu = sp.Matrix([u, v, r])
    x = sp.Matrix([n, e, psi, u, v, r])  # Full state vector
    tau = sp.Matrix([tau_u, tau_v, tau_r])

    ######### I must add impact of input to get complete model!!! ########

    # Kinematic equations: eta_dot = J(psi) * nu
    J_psi = sp.Matrix([
        [sp.cos(psi), -sp.sin(psi), 0],
        [sp.sin(psi), sp.cos(psi), 0],
        [0, 0, 1]
    ])
    eta_dot = J_psi @ nu
    eta_kp1 = eta + eta_dot * dt 

    # Dynamic equations: nu_dot = M^(-1) * (tau - C*nu - D*nu)
    MRB = sp.Matrix([
        [params.m_tot, 0, 0],
        [0, params.m_tot, params.m_tot*params.rg[0]],
        [0, params.m_tot*params.rg[0], params.Iz]
    ])
    MA = sp.Matrix([
        [-params.Xudot, 0, 0],
        [0, -params.Yvdot, -params.Yrdot],
        [0, -params.Yrdot, -params.Nvdot]
    ])
    CRB = sp.Matrix([
        [0, 0, -params.m_tot * (params.rg[0]*r + v)],
        [0, 0, params.m_tot * u],
        [params.m_tot * (params.rg[0]*r + v), -params.m_tot * u, 0]
    ])
    CA = sp.Matrix([
        [0, 0, params.Yvdot * v + params.Yrdot * r],
        [0, 0, -params.Xudot * u],
        [-params.Yvdot * v - params.Yrdot * r, params.Xudot * u, 0]
    ])
    D = sp.Matrix([
        [-params.Xu, 0, 0],
        [0, -params.Yv, -params.Yr],
        [0, -params.Nv, -params.Nr]
    ])

    nu_dot = (MA + MRB).inv() @ (tau - (CA + CRB) @ nu - D @ nu)
    nu_kp1 = nu + nu_dot * dt

    # Complete dynamics: f = [eta_dot; nu_dot]
    f = sp.Matrix([
        eta_kp1[0, 0],  # n_k+1
        eta_kp1[1, 0],  # e_k+1
        eta_kp1[2, 0],  # psi_k+1
        nu_kp1[0, 0],   # u_k+1
        nu_kp1[1, 0],   # v_k+1
        nu_kp1[2, 0]    # r_k+1
    ])

    # Now compute the Jacobian
    J_x = f.jacobian(x)

    # Convert to lambdified function
    Jx_lambda = sp.lambdify([n, e, psi, u, v, r], J_x, 'numpy')
    f_lambda = sp.lambdify([n, e, psi, u, v, r] + [tau_u, tau_v, tau_r], f, 'numpy')
    
    print("Jacobian shape:", J_x.shape)
    print("Jacobian:\n", J_x)
    print("Model:\n", f)
    print(Jx_lambda(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

if __name__ == "__main__":
    # jacobians()
    test()