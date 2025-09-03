import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple
from python_vehicle_simulator.lib.weather import Current

"""
^ surge (x)
|
|
O -----> sway (y)

"""


class IActuator(ABC):
    def __init__(
            self,
            xy:Tuple,               # surge, sway in body frame
            orientation:float,      # clockwise angle (rad) w.r.t surge 
            u_0:Tuple,              # initial input value
            u_min:Tuple,            # min input value
            u_max:Tuple,            # max input value
            *args,
            time_const:Tuple=None,  # Time constant
            f_min:Tuple=None,       # min force
            f_max:Tuple=None,       # max force
            **kwargs

    ):
        self.xy = np.array(xy)
        self.u_0 = np.array(u_0) # initial command
        self.u_min = np.array(u_min)
        self.u_max = np.array(u_max)
        self.dim = len(u_0)
        self.time_const = np.array(time_const) or np.array(self.dim * [float('inf')])
        self.u_prev = np.array(u_0)
        self.f_min = np.array(f_min)
        self.f_max = np.array(f_max)


    def dynamics(self, u:np.ndarray, nu:np.ndarray, current:Current, dt:float, *args, **kwargs) -> np.ndarray:
        """
            Wrapper for __dynamics__. Add saturation to actuator input commands
        """
        u_dot = (u-self.u_prev) / self.time_const
        u = np.clip(self.u_prev + u_dot * dt, self.u_min, self.u_max) # clip input within min/max bounds
        tau = self.__dynamics__(u, nu, current, *args, **kwargs)
        self.u_prev = u
        return tau

    @abstractmethod
    def __dynamics__(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        """
            Input:      u (np.ndarray) - For example propeller speed
            Output:     f (np.ndarray) - Generalized force (fx, fy, fz, Mx, My, Mz)
        """
        return np.zeros((6,))   
    
    def reset(self):
        self.u_prev = self.u_0.copy()

class Thruster(IActuator):
    def __init__(
            self,
            xy:Tuple,
            k_pos:float,    # speed-thrust quad. mapping > 0 (n>=0)
            k_neg:float,    # speed-thrust quad. mapping > 0 (n<0)
            n_min:float,    # Min propeller speed (can be negative)
            n_max:float,    # Max propeller speed
            *args,
            n_0:float=0.0,  # initial propeller speed (rad/s)
            T_n:float=0.0,  # Time constant
            f_max:float=float('inf'),
            f_min:float=-float('inf'),
            orientation:float=0.0,
            **kwargs
    ):
        super().__init__(xy=xy, orientation=orientation, u_0=(n_0,), u_min=(n_min,), u_max=(n_max,), *args, time_const=(T_n,), f_min=(f_min,), f_max=(f_max,), **kwargs)
        self.k_pos = k_pos
        self.k_neg = k_neg

    def __dynamics__(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        f = np.clip(self.k_pos * u[0]**2 if u[0]>=0 else -self.k_neg * u[0]**2, self.f_min, self.f_max)
        return np.array([1, 0, 0, 0, 0, self.xy[1]]) * f
    
    def Ti(self, u:np.ndarray, nu:np.ndarray, current:Current, *args, **kwargs) -> np.ndarray:
        
        return


            

class fin:
    '''
    Represents a fin for hydrodynamic calculations.

    INPUTS:
        a:      fin area (m^2)
        CL:     coefficient of lift (dimensionless)
        x:      x distance (m) from center of vehicle (negative for behind COM)
        c:      radius (m) from COP on the fin to the COM in the YZ plane
        angle:  offset of fin angle around x axis (deg) starting from positive y
                0 deg: fin on left side looking from front
                90 deg: fin on bottom
        rho:    density of fluid (kg/m^3) 
    
    Coordinate system: Right-handed, x-forward, y-starboard, z-down
    '''
    
    def __init__(
            self,
            a,
            CL,
            x,
            c = 0,
            angle = 0,
            rho = 1026,  # Default density of seawater
    ):
        
        self.area = a  # Fin area (m^2)
        self.CL = CL   # Coefficient of lift (dimensionless)
        self.angle_rad = np.deg2rad(angle)  
        self.rho = rho  # Fluid density (kg/m^3)

        self.u_actual_fin = 0.0  #Actual position of the fin (rad)
        self.T_delta = 0.1              # fin time constant (s) 
        self.deltaMax = np.deg2rad(15) # max rudder angle (rad)

        # Calculate fin's Center of Pressure (COP) position relative to COB
        y = np.cos(self.angle_rad) * c  # y-component of COP (m)
        z = np.sin(self.angle_rad) * c  # z-component of COP (m)
        self.R = np.array([x, y, z])    # Location of COP of the fin relative to COB (m)

    def velocity_in_rotated_plane(self, nu_r):
        """
        Calculate velocity magnitude in a plane rotated around the x-axis.

        Parameters:
            nu_r (numpy array): Velocity vector [vx, vy, vz] (m/s) in ENU frame

        Returns:
            float: Magnitude of velocity (m/s) in the rotated plane.
        """
        # Extract velocity components
        vx, vy, vz = nu_r  # m/s

        # Rotate y component around x-axis to align with fin plane
        vy_rot = np.sqrt((vy * np.sin(self.angle_rad))**2 + (vz * np.cos(self.angle_rad))**2)

        # Calculate magnitude in the rotated plane (x, y')
        U_plane = np.sqrt(vx**2 + vy_rot**2)

        return U_plane  # m/s

    def tau(self, nu_r,  nu):
        """
        Calculate force vector generated by the fin.

        Parameters:
            nu_r (numpy array): Relative velocity [vx, vy, vz, p, q, r] 
                              (m/s for linear, rad/s for angular)
            nu (numpy array): Velocity [vx, vy, vz, p, q, r] 
                              (m/s for linear, rad/s for angular) 

        Returns:
            numpy array: tau vector [Fx, Fy, Fz, Tx, Ty, Tz] (N) and (N*m) in body-fixed frame
        """
        
        ur = self.velocity_in_rotated_plane(nu_r[:3])  # Calulate relative velocity in plane of the fin
        
        # Calculate lift force magnitude
        f = 0.5 * self.rho * self.area * self.CL * self.u_actual_fin * ur**2  # N

        # Decompose force into y and z components
        fy = np.sin(self.angle_rad) * f  # N
        fz = -np.cos(self.angle_rad) * f  # N 

        F = np.array([0, fy, fz])  # Force vector (N)

        # Calculate torque using cross product of force and moment arm
        torque = np.cross(self.R, F)  # N*m
        return np.append(F, torque)
    
    def actuate(self, sampleTime, command):
        # Actuator dynamics        
        delta_dot = (command - self.u_actual_fin) / self.T_delta  
        self.u_actual_fin += sampleTime * delta_dot  # Euler integration 

        # Amplitude Saturation
        if abs(self.u_actual_fin) >= self.deltaMax:
            self.u_actual_fin = np.sign(self.u_actual_fin) * self.deltaMax

        return self.u_actual_fin





class thruster:
    '''
    Represents a thruster for hydrodynamic calculations.

    INPUTS:
        rho:    density of fluid (kg/m^3) 
    
    Coordinate system: Right-handed, x-forward, y-starboard, z-down
    '''
    def __init__(self, rho):
        # Actuator dynamics
        self.nMax = 1525                # max propeller revolution (rpm)    
        self.T_n = 0.1                  # propeller time constant (s)
        self.u_actual_n = 0.0           # actual rpm of the thruster
        self.rho = rho

    def tau(self, nu_r, nu):
        """
        Calculate force vector generated by the fin.

        Parameters:
            nu_r (numpy array): Relative velocity [vx, vy, vz, p, q, r] 
                              (m/s for linear, rad/s for angular)
            nu (numpy array): Velocity [vx, vy, vz, p, q, r] 
                              (m/s for linear, rad/s for angular) 

        Returns:
            numpy array: tau vector [Fx, Fy, Fz, Tx, Ty, Tz] (N) and (N*m) in body-fixed frame
        """
        U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # vehicle speed

        # Commands and actual control signals
        n = self.u_actual_n            # actual propeller revolution (rpm)
        
        # Amplitude saturation of the control signals
        if abs(n) >= self.nMax:
            n = np.sign(n) * self.nMax       
        
        # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        D_prop = 0.14   # propeller diameter corresponding to 5.5 inches
        t_prop = 0.1    # thrust deduction number
        n_rps = n / 60  # propeller revolution (rps) 
        Va = 0.944 * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        Ja_max = 0.6632
        
        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        KT_0 = 0.4566
        KQ_0 = 0.0700
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        KT_max = 0.1798
        KQ_max = 0.0312
        
        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  
      
        if n_rps > 0:   # forward thrust

            X_prop = self.rho * pow(D_prop,4) * ( 
                KT_0 * abs(n_rps) * n_rps + (KT_max-KT_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )        
            K_prop = self.rho * pow(D_prop,5) * (
                KQ_0 * abs(n_rps) * n_rps + (KQ_max-KQ_0)/Ja_max * 
                (Va/D_prop) * abs(n_rps) )           
            
        else:    # reverse thrust (braking)
        
            X_prop = self.rho * pow(D_prop,4) * KT_0 * abs(n_rps) * n_rps 
            K_prop = self.rho * pow(D_prop,5) * KQ_0 * abs(n_rps) * n_rps 

        # Thrust force vector
        # K_Prop scaled down by a factor of 10 to match exp. results
        tau_thrust = np.array([(1-t_prop) * X_prop, 0, 0, K_prop / 10, 0, 0], float)
        return tau_thrust

    def actuate(self, sampleTime ,command):
        # Actuator dynamics
        n_dot = (command - self.u_actual_n) / self.T_n

        self.u_actual_n += sampleTime * n_dot
        
        return self.u_actual_n
        