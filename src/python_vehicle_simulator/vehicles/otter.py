#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otter.py: 
    Class for the Maritime Robotics Otter USV, www.maritimerobotics.com. 
    The length of the USV is L = 2.0 m. The constructors are:

    otter()                                          
        Step inputs for propeller revolutions n1 and n2
        
    otter('headingAutopilot',psi_d,V_current,beta_current,tau_X)  
       Heading autopilot with options:
          psi_d: desired yaw angle (deg)
          V_current: current speed (m/s)
          beta_c: current direction (deg)
          tau_X: surge force, pilot input (N)
        
Methods:
    
[nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) returns 
    nu[k+1] and u_actual[k+1] using Euler's method. The control inputs are:

    u_control = [ n1 n2 ]' where 
        n1: propeller shaft speed, left (rad/s)
        n2: propeller shaft speed, right (rad/s)

u = headingAutopilot(eta,nu,sampleTime) 
    PID controller for automatic heading control based on pole placement.

u = stepInput(t) generates propeller step inputs.

[n1, n2] = controlAllocation(tau_X, tau_N)     
    Control allocation algorithm.
    
References: 
  T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion 
     Control. 2nd. Edition, Wiley. 
     URL: www.fossen.biz/wiley            

Author:     Thor I. Fossen
"""
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Union
from python_vehicle_simulator.lib.control import PIDpolePlacement
from python_vehicle_simulator.lib.physics import m2c, crossFlowDrag, GRAVITY, RHO
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
class OtterThrusterParameters:
    ## Propellers       
    T_n: float = 0.1                    # Propeller time constant (s)
    k_neg: float = 0.01289 / 2 / 2       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    k_pos: float = 0.02216 / 2 / 2 # 0.02216 / 2          # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    f_max: float = 24.4 * GRAVITY / 2 / 1        # Max positive force, one propeller
    f_min: float = -13.6 * GRAVITY / 2 / 1       # Max negative force, one propeller
    n_max: float = field(init=False)    # Max (positive) propeller speed
    n_min: float = field(init=False)    # Min (negative) propeller speed
    n_dot_max: float = field(init=False)

    def __post_init__(self):
        self.n_max = math.sqrt(self.f_max / self.k_pos) 
        self.n_min = -math.sqrt(-self.f_min / self.k_neg)


@dataclass
class OtterParameters:
    loa: float = 2.0                                    # Length Over All (m)
    beam: float = 1.08                                  # Beam (m)
    R44: float = 0.4 * beam                             # radii of gyration (m)
    R55: float = 0.25 * loa
    R66: float = 0.25 * loa

    ## Time constants
    T_sway: float = 1.0                                 # Time constant in sway (s)
    T_yaw: float = 1.0                                  # Time constant in yaw (s)

    ## Mass & Payload
    m: float = 55.0                                     # Mass (kg)
    mp: float = 25.0                                    # Payload mass (kg)
    m_tot: float = field(init=False)                    # Total mass (kg)
    rp: np.ndarray = field(default_factory=lambda: np.array([0.05, 0, -0.35], float))         # Location of payload (m)
    rg: np.ndarray = field(default_factory=lambda: np.array([0.2, 0, -0.2], float))           # CG for hull and corrected for payload in __post_init__(m)
    S_rg: np.ndarray = field(init=False)                # Skew matrix for rg
    H_rg: np.ndarray = field(init=False)                # Homogeneous transformation for rg
    S_rp: np.ndarray = field(init=False)                # Skew matrix for rp

    ## Max speed
    u_max: float = knot_to_m_per_sec(6)                 # Max forward speed (m/s)

    ## Pontoon
    B_pont: float = 0.25                                # Beam of one pontoon (m)
    y_pont: float = 0.395                               # Distance from centerline to waterline centroid (m)
    Cw_pont: float = 0.75                               # Waterline area coefficient (-)
    Cb_pont: float = 0.4                                # Block coefficient, computed from m = 55 kg

    ## Inertia dyadic, volume displacement and draft
    nabla: float = field(init=False)                    # volume of volume displaced
    draft: float = field(init=False)                    # draft
    Ig_cg: np.ndarray = field(init=False)               # Inertia dyadic at CG
    Ig: np.ndarray = field(init=False)                  # Inertia dyadic


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
        self.nabla = (self.m + self.mp) / RHO
        self.draft = self.nabla / (2 * self.Cb_pont * self.B_pont * self.loa)
        self.Ig_cg = self.m * np.diag(np.array([self.R44 ** 2, self.R55 ** 2, self.R66 ** 2]))
        self.Ig = self.Ig_cg - self.m * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp
        
        # Mass matrices
        self.MRB_CG = np.zeros((6, 6))
        self.MRB_CG[0:3, 0:3] = (self.m + self.mp) * np.identity(3)
        self.MRB_CG[3:6, 3:6] = self.Ig
        self.MRB = self.H_rg.T @ self.MRB_CG @ self.H_rg
        
        # Hydrodynamics added mass
        self.Xudot = -0.1 * self.m
        self.Yvdot = -1.5 * self.m
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
            2
            * (1 / 12)
            * self.loa
            * self.B_pont ** 3
            * (6 * self.Cw_pont ** 3 / ((1 + self.Cw_pont) * (1 + 2 * self.Cw_pont)))
            + 2 * self.Aw_pont * self.y_pont ** 2
        )
        self.I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.loa ** 3
        self.KB = (1 / 3) * (5 * self.draft / 2 - 0.5 * self.nabla / (self.loa * self.B_pont))
        self.BM_T = self.I_T / self.nabla
        self.BM_L = self.I_L / self.nabla
        self.KM_T = self.KB + self.BM_T
        self.KM_L = self.KB + self.BM_L
        self.KG = self.draft - self.rg[2]
        self.GM_T = self.KM_T - self.KG
        self.GM_L = self.KM_L - self.KG

        self.G33 = RHO * GRAVITY * (2 * self.Aw_pont)
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
        self.Yv = -self.M[1, 1] / self.T_sway
        self.Zw = -2 * 0.3 * self.w3 * self.M[2, 2]
        self.Kp = -2 * 0.2 * self.w4 * self.M[3, 3]
        self.Mq = -2 * 0.4 * self.w5 * self.M[4, 4]
        self.Nr = -self.M[5, 5] / self.T_yaw

        self.D = -np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])


# Class Vehicle
class Otter(IVessel):
    """
    otter()                                           Propeller step inputs
    otter('headingAutopilot',psi_d,V_c,beta_c,tau_X)  Heading autopilot
    
    Inputs:
        psi_d: desired heading angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)        
    """

    def __init__(
        self, 
        params:OtterParameters, # Could be moved in the construct since it will never change
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
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0], float) # derivative
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.params.m_tot * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.params.Ig, nu[3:6]))
        CRB = self.params.H_rg.T @ CRB_CG @ self.params.H_rg  # transform CRB from CG to CO

        CA = m2c(self.params.MA, nu_r)
        # Uncomment to cancel the Munk moment in yaw, if stability problems
        # CA[5, 0] = 0  
        # CA[5, 1] = 0 
        # CA[0, 5] = 0
        # CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(*eta[3:6])
        f_payload = np.matmul(R.T, np.array([0, 0, self.params.mp * GRAVITY], float))              
        m_payload = np.matmul(self.params.S_rp, f_payload)
        g_0 = np.array([ f_payload[0],f_payload[1],f_payload[2], 
                         m_payload[0],m_payload[1],m_payload[2] ])

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.params.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.params.D[5, 5] * abs(nu_r[5]) * nu_r[5]

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

        # print("actuators: ", tau_actuators)
        # print("nu: ", nu)
        # print("damping: ", tau_damp)
        # print("crossflow: ", tau_crossflow)
        # print("C@nu_r: ", np.matmul(C, nu_r))
        # print("Gmtrx @ eta: ", np.matmul(self.params.Gmtrx, eta))
        # print("g_0: ", g_0)

        # USV dynamics
        nu_dot = Dnu_c + np.matmul(self.params.Minv, sum_tau)

        return nu_dot


@dataclass
class OtterParameters3DOF:
    """3DOF Otter parameters - only horizontal plane motion (surge, sway, yaw)"""
    loa: float = 2.0                                    # Length Over All (m)
    beam: float = 1.08                                  # Beam (m)
    R66: float = 0.4 * beam                             # Radius of gyration in yaw (m)
    
    # Mass and inertia (3DOF: [m, m, Izz])
    m: float = 55.0                                     # Mass (kg)
    Izz: float = m * R66**2                             # Yaw moment of inertia (kg*m^2)
    
    # Added mass matrix (3DOF) - directly from 6DOF model for consistency
    X_du: float = 5.5                                # Surge added mass: 0.1 * m
    Y_dv: float = 82.5                               # Sway added mass: 1.5 * m  
    N_dr: float = 24.6                               # Yaw added mass: 1.7 * Izz (approx)
    
    # Damping coefficients (3DOF) - CORRECTED for realistic dynamics
    X_u: float = 2.75                                  # Linear surge damping (was -0.4)
    Y_v: float = 5.50                                  # Linear sway damping (was -0.8)
    N_r: float = 4.69                                  # Linear yaw damping (was -1.0)
    
    # Cross-flow drag parameters
    B_pont: float = 0.25                                # Pontoon beam (m)
    draft: float = 0.075                                # Draft (m)
    
    def __post_init__(self):
        # Mass matrix (3x3)
        self.M = np.array([
            [self.m + self.X_du, 0, 0],
            [0, self.m + self.Y_dv, 0],
            [0, 0, self.Izz + self.N_dr]
        ])
        
        self.CA = lambda ur, vr, r : np.array([
            [0, 0, self.Y_dv * vr],
            [0, 0, -self.X_du * ur],
            [-self.Y_dv * vr, self.X_du * ur, 0]])

        self.CRB = lambda u, v, r : np.array([
            [0,           0,         -self.m * v],
            [0,           0,          self.m * u],
            [self.m * v,    -self.m * u,    0      ]
        ])

        # Damping matrix (3x3) - linear terms only
        self.D =  np.array([
            [self.m / self.X_u, 0, 0],
            [0, self.m / self.Y_v, 0],
            [0, 0, self.Izz / self.N_r]
        ])

        # self.DNL = lambda u, v, r : np.array([
        #     [self.ku * self.forward_speed, 0, 0],
        #     [0, self.kv * self.sideways_speed, 0],
        #     [0, 0, self.kr * self.yaw_rate]
        # ])
        
        # Inverse mass matrix
        self.Minv = np.linalg.inv(self.M)


class Otter3DOF(IVessel):
    """3DOF Otter USV model - horizontal plane motion only"""
    
    def __init__(self, params: OtterParameters3DOF, dt: float, actuators: List = None, nu: Tuple[float, float, float] = (0, 0, 0), eta: Tuple[float, float, float] = (0, 0, 0), **kwargs):
        """
        Initialize 3DOF Otter USV
        
        Args:
            params: Otter3DOF parameters
            dt: Sample time (s)
            actuators: List of actuators (thrusters)
            nu: Initial velocity [u, v, r] (m/s, m/s, rad/s)
            eta: Initial position [x, y, psi] (m, m, rad)
        """
        # Convert 3DOF states to 6DOF for base class compatibility
        eta_6dof = (eta[0], eta[1], 0.0, 0.0, 0.0, eta[2])  # [N, E, D, roll, pitch, yaw]
        nu_6dof = (nu[0], nu[1], 0.0, 0.0, 0.0, nu[2])      # [u, v, w, p, q, r]
        
        super().__init__(params, dt, actuators=actuators, eta=eta_6dof, nu=nu_6dof, **kwargs)
        self.params = params
        
    def get_eta_3dof(self) -> np.ndarray:
        """Get 3DOF position/orientation [x, y, psi]"""
        return np.array([self.eta.n, self.eta.e, self.eta.yaw])
        
    def get_nu_3dof(self) -> np.ndarray:
        """Get 3DOF velocity [u, v, r]"""
        return np.array([self.nu.u, self.nu.v, self.nu.r])
    
    def get_thruster_actual_speeds(self) -> np.ndarray:
        u_actual = []
        for actuator in self.actuators:
            u_actual.append(actuator.u_actual_prev[0])
        return np.array(u_actual)
        
    def set_eta_3dof(self, eta_3dof: np.ndarray):
        """Set 3DOF position/orientation [x, y, psi]"""
        self.eta.n = eta_3dof[0]
        self.eta.e = eta_3dof[1]
        self.eta.yaw = eta_3dof[2]
        # Keep other DOF at zero
        self.eta.d = 0.0
        self.eta.roll = 0.0
        self.eta.pitch = 0.0
        
    def set_nu_3dof(self, nu_3dof: np.ndarray):
        """Set 3DOF velocity [u, v, r]"""
        self.nu.u = nu_3dof[0] 
        self.nu.v = nu_3dof[1]
        self.nu.r = nu_3dof[2]
        # Keep other DOF at zero
        self.nu.w = 0.0
        self.nu.p = 0.0
        self.nu.q = 0.0
        
    def __dynamics__(self, tau_actuators: np.ndarray, current: Current, wind: Wind, *args, **kwargs) -> np.ndarray:
        """
        Required implementation of abstract method from IVessel base class.
        3DOF USV dynamics: M * nu_dot + C(nu) * nu + D * nu = tau
        """
        # Get current 3DOF states
        nu_3dof = self.get_nu_3dof()
        
        # Extract 3DOF forces from 6DOF actuator input
        tau_3dof = np.array([tau_actuators[0], tau_actuators[1], tau_actuators[5]])  # [X, Y, N]
        
        # 3DOF dynamics computation
        u, v, r = nu_3dof
        
        # Coriolis-centripetal matrix (3DOF)        
        CRB = self.params.CRB(u, v, r)
        CA = self.params.CA(u, v, r)
        C = CA + CRB


        
        # Cross-flow drag (simplified for 3DOF)
        # tau_crossflow = self._crossflow_drag_3dof(nu_3dof)
        
        # Quadratic damping (simplified)
        # tau_quad_damp = np.array([
        #     -0.1 * abs(u) * u,  # Surge quadratic damping
        #     -0.2 * abs(v) * v,  # Sway quadratic damping  
        #     -0.1 * abs(r) * r   # Yaw quadratic damping
        # ])
        
        # Total forces and moments
        tau_coriolis = - C @ nu_3dof
        tau_damping = - self.params.D @ nu_3dof
        tau_total = tau_3dof + tau_coriolis + tau_damping
        # print(tau_3dof, tau_coriolis, tau_damping * np.array([u, v, r]))
        
        # Acceleration
        nu_dot_3dof = self.params.Minv @ tau_total
        
        # Convert back to 6DOF for base class compatibility
        nu_dot_6dof = np.zeros(6)
        nu_dot_6dof[0] = nu_dot_3dof[0]  # u_dot
        nu_dot_6dof[1] = nu_dot_3dof[1]  # v_dot  
        nu_dot_6dof[5] = nu_dot_3dof[2]  # r_dot
        
        return nu_dot_6dof
        
    # def _crossflow_drag_3dof(self, nu: np.ndarray) -> np.ndarray:
    #     """Simplified cross-flow drag for 3DOF model"""
    #     v = nu[1]  # Sway velocity
    #     r = nu[2]  # Yaw rate
        
    #     # Simplified cross-flow - only significant for large sway velocities
    #     rho = RHO
    #     L = self.params.loa
    #     T = self.params.draft
    #     Cd_2D = 1.2  # Simplified drag coefficient
        
    #     # Cross-flow force and moment (very simplified)
    #     Y_crossflow = -0.5 * rho * T * Cd_2D * abs(v) * v
    #     N_crossflow = -0.5 * rho * T * Cd_2D * (L/4) * abs(v) * v  # Simplified moment arm
        
    #     return np.array([0, Y_crossflow, N_crossflow])


if __name__ == "__main__":
    # Test both classes
    print("6DOF Otter Parameters:")
    data_6dof = OtterParameters()
    print(data_6dof)
    
    print("\n3DOF Otter Parameters:")
    data_3dof = OtterParameters3DOF()
    print(data_3dof)
    
    print(f"\n3DOF Mass matrix:\n{data_3dof.M}")
    print(f"3DOF Damping matrix:\n{data_3dof.D}")