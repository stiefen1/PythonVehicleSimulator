from math import pi
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from python_vehicle_simulator.lib.physics import GRAVITY, RHO
from python_vehicle_simulator.utils.math_fn import R_casadi
from python_vehicle_simulator.utils.unit_conversion import knot_to_m_per_sec
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.diagnosis import IDiagnosis
from python_vehicle_simulator.lib.dynamics import IDynamics
import casadi as cs, math, numpy, numpy.typing as npt

INF = float('inf')

@dataclass
class RevoltBowThrusterParams: # front
    ## Propellers       
    T_n: float = 0.3                                            # Propeller time constant (s)
    T_a: float = 3.0                                            # Azimuth angle time constant (s) -> Chosen by me
    k_pos: float = 1.518e-3                                     # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: float = 6.172e-4                                     # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: float = 14                                           # Max positive force, one propeller
    f_min: float = 0 # -6.1                                     # Max negative force, one propeller
    speed_max: float = math.sqrt(f_max/k_pos)                   # Max (positive) propeller speed
    speed_min: float = -math.sqrt(-f_min/k_neg)
    alpha_max: float = math.pi                                  # Max (positive) propeller speed
    alpha_min: float = -math.pi                                 # Min (negative) propeller speed
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
    speed_max: float = math.sqrt(f_max/k_pos)                   # Max (positive) propeller speed
    speed_min: float = -math.sqrt(-f_min/k_neg)
    alpha_max: float = math.pi                                      # Max (positive) propeller speed
    alpha_min: float = -math.pi                                     # Min (negative) propeller speed
    max_radians_per_step: float = math.pi/6
    max_newton_per_step: float = 10

@dataclass
class RevoltThrusterParameters:
    ## Propellers       
    thrusters: List = field(init=False)                         
    k_pos: npt.NDArray = field(init=False)                       # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
    k_neg: npt.NDArray = field(init=False)                       # Negative Bollard, one propeller (Division by two because there are two propellers, values are obtained with a Bollard pull)
    f_max: npt.NDArray = field(init=False)                       # Max positive force, one propeller
    f_min: npt.NDArray = field(init=False)                       # Max negative force, one propeller
    speed_max: npt.NDArray = field(init=False)                   # Max (positive) propeller speed
    speed_min: npt.NDArray = field(init=False)
    alpha_min: npt.NDArray = field(init=False)                   # Max (positive) propeller speed
    alpha_max: npt.NDArray = field(init=False)                   # Min (negative) propeller speed
    xy: npt.NDArray = field(init=False)                          # azimuth, azimuth, thruster from https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2452115/16486_FULLTEXT.pdf (p.56)
    time_constant: npt.NDArray = field(init=False)

    def __post_init__(self):
        self.thrusters = [RevoltSternThrusterParams(), RevoltSternThrusterParams(), RevoltBowThrusterParams()]  
        self.k_pos: npt.NDArray = numpy.array([thruster.k_pos for thruster in self.thrusters])      # Positive Bollard, one propeller -> f_i = k_pos * n_i * |n_i| if n_i>0 else k_neg * n_i * |n_i|
        self.k_neg: npt.NDArray = numpy.array([thruster.k_neg for thruster in self.thrusters])      # Negative Bollard, one propeller
        self.f_max: npt.NDArray = numpy.array([thruster.f_max for thruster in self.thrusters])      # Max positive force, one propeller
        self.f_min: npt.NDArray = numpy.array([thruster.f_min for thruster in self.thrusters])      # numpy.array([-25, -25, -6.1]) 
        
        self.speed_max = numpy.array([thruster.speed_max for thruster in self.thrusters])          # Thruster speed constraints
        self.speed_min = numpy.array([thruster.speed_min for thruster in self.thrusters])
        self.alpha_min = numpy.array([thruster.alpha_min for thruster in self.thrusters])          # Azimuth angles constraints
        self.alpha_max = numpy.array([thruster.alpha_max for thruster in self.thrusters])
        
        self.xy = numpy.array([[-1.65, -0.15], [-1.65, 0.15], [1.15, 0.0]])                        # Azimuth thruster positions
        self.T_n = numpy.array([thruster.T_n for thruster in self.thrusters])
        self.T_a = numpy.array([thruster.T_a for thruster in self.thrusters])

        self.Ti = lambda alpha, lx, ly : numpy.array([
            cs.cos(alpha),
            cs.sin(alpha),
            lx * cs.sin(alpha) - ly * cs.cos(alpha)
        ])

        ####### WARNING : IF YOU CHANGE T YOU HAVE TO DO IT AS WELL IN RL ENVIRONMENTS, IT IS NOT LINKED ######
        self.T = lambda a1, a2, a3 : cs.vertcat(
            cs.horzcat(cs.cos(a1), cs.sin(a1), self.xy[0, 0]*cs.sin(a1) - self.xy[0, 1] * cs.cos(a1)),
            cs.horzcat(cs.cos(a2), cs.sin(a2), self.xy[1, 0]*cs.sin(a2) - self.xy[1, 1] * cs.cos(a2)),
            cs.horzcat(cs.cos(a3), cs.sin(a3), self.xy[2, 0]*cs.sin(a3) - self.xy[2, 1] * cs.cos(a3))
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
    rg: npt.NDArray = field(default_factory=lambda: numpy.array([0.0, 0, 0.0], float))  
    rp: npt.NDArray = field(default_factory=lambda: numpy.array([0.0, 0, 0.0], float))          # Location of payload (m)

    ## State constraints
    lbx: npt.NDArray = field(default_factory=lambda: numpy.array([-INF, -INF, -pi, -1, -1, -pi/6]))
    ubx: npt.NDArray = field(default_factory=lambda: numpy.array([INF, INF, pi, knot_to_m_per_sec(3), 1, pi/6]))


    def __post_init__(self):
        self.m_tot = self.m + self.mp
        self.rg = (self.m * self.rg + self.mp * self.rp) / (self.m + self.mp) # corrected center of gravity with payload

        # Basic calculations
        self.Xudot, self.Yvdot, self.Yrdot, self.Nvdot = (-self.volume*RHO*numpy.array([0.0253, 0.1802, 0.0085 * self.loa, 0.0099 * self.loa**2])).tolist()
        self.Xu, self.Yv, self.Yr, self.Nv, self.Nr = (-numpy.array([
            0.102 * RHO * self.volume * GRAVITY / self.loa,
            1.212 * GRAVITY / self.loa,
            0.056 * RHO * self.volume * numpy.sqrt(GRAVITY * self.loa),
            0.056 * RHO * self.volume * numpy.sqrt(GRAVITY * self.loa),
            0.0601 * RHO * self.volume * numpy.sqrt(GRAVITY * self.loa)
            ])).tolist()
        self.Iz = self.m_tot / self.volume * self.volume_iz

        ## Inertia Matrix
        MRB = numpy.array([
            [self.m_tot, 0, 0],
            [0, self.m_tot, self.m_tot*self.rg[0]],
            [0, self.m_tot*self.rg[0], self.Iz]
        ])
        MA = numpy.array([
            [-self.Xudot, 0, 0],
            [0, -self.Yvdot, -self.Yrdot],
            [0, -self.Yrdot, -self.Nvdot]
        ])
        self.Minv = numpy.linalg.inv(MA + MRB)

        ## Coriolis-Centripetal Matrix
        self.CRB = lambda nu : numpy.array([
            [0, 0, -self.m_tot * (self.rg[0]*nu[2] + nu[1])],
            [0, 0, self.m_tot * nu[0]],
            [self.m_tot * (self.rg[0]*nu[2] + nu[1]), -self.m_tot * nu[0], 0]
        ])
        self.CA = lambda nu_r : numpy.array([
            [0, 0, self.Yvdot * nu_r[1] + self.Yrdot * nu_r[2]],
            [0, 0, -self.Xudot * nu_r[0]],
            [-self.Yvdot * nu_r[1]-self.Yrdot * nu_r[2], self.Xudot * nu_r[0], 0]
        ])

        ## Damping Matrix
        self.D = numpy.array([
            [-self.Xu, 0, 0],
            [0, -self.Yv, -self.Yr],
            [0, -self.Nv, -self.Nr]
        ])

class ReVolt3Dynamics(IDynamics):
    actuators_params = RevoltThrusterParameters()
    vessel_params = RevoltParameters3DOF()

    def __init__(
            self,
            dt: float,
            *args, 
            **kwargs
    ):
        nx, nu, np, nd = 12, 6, 6, 3 # states, commands, parameters, disturbances
        super().__init__(nx, nu, np, nd, dt, *args, **kwargs)

    def continuous_time_dynamics(self, x: cs.SX, u: cs.SX, theta: cs.SX, disturbance: cs.SX, *args, **kwargs) -> cs.SX:
        """
        x:              states: eta, nu, thruster speeds, azimuth angles    (nx,)
        u:              control commands                                    (nu,)
        theta:          parameters (e.g. faults)                            (np,)
        disturbance:    disturbance (e.g. wind)                             (nd,)
        """
        f = cs.SX.zeros(self.nx) # type: ignore

        eta, nu, azimuth, speed = x[0:3], x[3:6], x[6:9], x[9:12]
        azimuth_setpoint, speed_setpoint = u[0:3], u[3:6]
        azimuth_stucked, propeller_effectiveness = theta[0:3], theta[3:6]

        # Generalized force generated by actuators
        thrust = propeller_effectiveness * self.actuators_params.k_pos * speed * speed
        tau_actuators = self.actuators_params.T(azimuth[0], azimuth[1], azimuth[2]) @ thrust

        # Hull dynamics (body frame)
        Minv, C, D = self.vessel_params.Minv, self.vessel_params.CA(nu) + self.vessel_params.CRB(nu), self.vessel_params.D
        f[3:6] = Minv @ (tau_actuators - C @ nu - D @ nu)

        # Ship's kinematics (body -> NED)
        f[0:3] = cs.mtimes(R_casadi(eta[2]), nu)

        # Actuator's dynamics: Low-pass
        f[6:9] = azimuth_stucked * (azimuth_setpoint - azimuth) / self.actuators_params.T_a
        f[9:12] = (speed_setpoint - speed) / self.actuators_params.T_n

        return f


class ReVolt3(IVessel):
    vessel_params = RevoltParameters3DOF()

    def __init__(
            self,
            dt: float,
            eta: Tuple = (0, 0, 0),
            nu: Tuple = (0, 0, 0),
            guidance: Optional[IGuidance] = None,
            navigation: Optional[INavigation] = None,
            control: Optional[IControl] = None,
            diagnosis: Optional[IDiagnosis] = None,
            mmsi: Optional[str] = None,
            verbose_level: int = 0
    ):
        super().__init__(
            self.vessel_params.loa,
            self.vessel_params.beam,
            ReVolt3Dynamics(dt),
            eta=eta,
            nu=nu,
            guidance=guidance,
            navigation=navigation,
            control=control,
            diagnosis=diagnosis,
            name='ReVolt3',
            mmsi=mmsi,
            verbose_level=verbose_level
        )
    
    def __dynamics__(self, control_commands:npt.NDArray, current:Current, wind:Wind, *args, theta: Optional[npt.NDArray] = None, **kwargs) -> Tuple[List, List]:
        disturbance = numpy.array([0, 0, 0]) # Define it as a function of current, wind
        theta = theta if theta is not None else numpy.array([])
        x = self.dynamics.fd(self.get_state(dofs=3), control_commands, theta, disturbance)
        return x[0:3].tolist(), x[3:6].tolist()

if __name__ == "__main__":
    revolt = ReVolt3(0.1)
    print(revolt.dynamics._f)
    print(revolt.dynamics.fd(
        numpy.array([0, 0, 0, 4, 0, 0, 0, 0, 0, 30, 30, 30]), 
        numpy.array([0, 0, 0, 40, 40, 40]),
        numpy.array([1, 1, 1, 1, 1, 1]),
        numpy.array([0, 0, 0])
        )
    )