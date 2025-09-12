import math, numpy as np
from python_vehicle_simulator.utils.math_fn import Smtrx
from typing import Tuple

GRAVITY = 9.81   # Gravity (m/s^2)
RHO = 1025 # Water density (kg/m^3)

#------------------------------------------------------------------------------

def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model
    
        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6] 
        M21 = M12.T
        M22 = M[3:6,3:6] 
    
        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
        dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros( (6,6) )    
        C[0:3,3:6] = -Smtrx(dt_dnu1)
        C[3:6,0:3] = -Smtrx(dt_dnu1)
        C[3:6,3:6] = -Smtrx(dt_dnu2)
            
    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
        C = np.zeros( (3,3) ) 
        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        C[1,2] =  M[0,0] * nu[0] 
        C[2,0] = -C[0,2]       
        C[2,1] = -C[1,2]
        
    return C

#------------------------------------------------------------------------------

def Hoerner(B,T):
    """
    CY_2D = Hoerner(B,T)
    Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
    B and draft T.The data is digitized and interpolation is used to compute 
    other data point than those in the table
    """
    
    # DATA = [B/2T  C_D]
    DATA1 = np.array([
        0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
        0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031 
        ])
    DATA2 = np.array([
        1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
        0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593 
        ])

    CY_2D = np.interp( B / (2 * T), DATA1, DATA2 )
        
    return CY_2D

#------------------------------------------------------------------------------

def crossFlowDrag(L,B,T,nu_r):
    """
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    integrals for a marine craft using strip theory. 

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    """

    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20             
    Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L/2
    
    for i in range(0,n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[5]               # yaw rate
        Ucf = abs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx
        
    tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh],float)

    return tau_crossflow

#------------------------------------------------------------------------------

def forceLiftDrag(b,S,CD_0,alpha,U_r):
    """
    tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
    lift and drag forces of a submerged "wing profile" for varying angle of
    attack (Beard and McLain 2012). Application:
    
      M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag
    
    Inputs:
        b:     wing span (m)
        S:     wing area (m^2)
        CD_0:  parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
        alpha: angle of attack, scalar or vector (rad)
        U_r:   relative speed (m/s)

    Returns:
        tau_liftdrag:  6x1 generalized force vector
    """

    # constants
    rho = 1026

    def coeffLiftDrag(b,S,CD_0,alpha,sigma):
        
        """
        [CL,CD] = coeffLiftDrag(b,S,CD_0,alpha,sigma) computes the hydrodynamic 
        lift CL(alpha) and drag CD(alpha) coefficients as a function of alpha
        (angle of attack) of a submerged "wing profile" (Beard and McLain 2012)

        CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
        CL(alpha) = CL_0 + CL_alpha * alpha
  
        where CD_p is the parasitic drag (profile drag of wing, friction and
        pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle 
        of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the  
        Oswald efficiency number. For lift it is assumed that
  
        CL_0 = 0
        CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );
  
        implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
        high angles of attack the linear lift model can be blended with a
        nonlinear model to describe stall
  
        CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
            sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

        where 0 <= sigma <= 1 is a blending parameter. 
        
        Inputs:
            b:       wing span (m)
            S:       wing area (m^2)
            CD_0:    parasitic drag (alpha = 0), typically 0.1-0.2 for a 
                     streamlined body
            alpha:   angle of attack, scalar or vector (rad)
            sigma:   blending parameter between 0 and 1, use sigma = 0 f
                     or linear lift 
            display: use 1 to plot CD and CL (optionally)
        
        Returns:
            CL: lift coefficient as a function of alpha   
            CD: drag coefficient as a function of alpha   

        Example:
            Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
            CD_0 = 0.3
            
            alpha = 0.1 * pi/180
            [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
        """
         
        e = 0.7             # Oswald efficiency number
        AR = b**2 / S       # wing aspect ratio

        # linear lift
        CL_alpha = math.pi * AR / ( 1 + math.sqrt(1 + (AR/2)**2) )
        CL = CL_alpha * alpha

        # parasitic and induced drag
        CD = CD_0 + CL**2 / (math.pi * e * AR)
        
        # nonlinear lift (blending function)
        CL = (1-sigma) * CL + sigma * 2 * np.sign(alpha) \
            * math.sin(alpha)**2 * math.cos(alpha)

        return CL, CD

    
    [CL, CD] = coeffLiftDrag(b,S,CD_0,alpha,0) 
    
    F_drag = 1/2 * rho * U_r**2 * S * CD    # drag force
    F_lift = 1/2 * rho * U_r**2 * S * CL    # lift force

    # transform from FLOW axes to BODY axes using angle of attack
    tau_liftdrag = np.array([
        math.cos(alpha) * (-F_drag) - math.sin(alpha) * (-F_lift),
        0,
        math.sin(alpha) * (-F_drag) + math.cos(alpha) * (-F_lift),
        0,
        0,
        0 ])

    return tau_liftdrag
    
#------------------------------------------------------------------------------

def gvect(W,B,theta,phi,r_bg,r_bb):
    """
    g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
    forces about an arbitrarily point CO for a submerged body. 
    
    Inputs:
        W, B: weight and buoyancy (kg)
        phi,theta: roll and pitch angles (rad)
        r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
        r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)
        
    Returns:
        g: 6x1 vector of restoring forces about CO
    """

    sth  = math.sin(theta)
    cth  = math.cos(theta)
    sphi = math.sin(phi)
    cphi = math.cos(phi)

    g = np.array([
        (W-B) * sth,
        -(W-B) * cth * sphi,
        -(W-B) * cth * cphi,
        -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
        (r_bg[2]*W-r_bb[2]*B) * sth         + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
        -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth      
        ])
    
    return g

def addedMassSurge(m, L) -> Tuple[float, float]:
    # % addedMassSurge is compatible with MATLAB and GNU Octave (www.octave.org). 
    # % The function A11 = addedMassSurge(m,L,rho) approximates the added mass 
    # % in surge bythe formula of Söding (1982):
    # %
    # %   A11 = -Xudot = 2.7 * rho * nabla^(5/3) / L^2
    # %
    # % Inputs:
    # %   m:     mass of the ship or vehicle (kg)
    # %   L:     length of the ship or vehicle (m)
    # %   rho:   density of water (kg/m3), default value is 1025 kg/m3
    # %
    # % Outputs:
    # %   A11:   added mass in surge (kg)
    # %   ratio: optional output, ratio of added mass to actual mass (A11/m)
    # %
    # % Example usage:
    # %   A11 = addedMassSurge(m, L)         - use default rho = 1025 (kg/m3)
    # %   A11 = addedMassSurge(m, L, rho)
    # %   [A11,ratio] = addedMassSurge(m, L, rho)
    # %
    # % Reference: H. Söding (1982). Prediction of Ship Steering Capabilities. 
    # %   Schiffstechnik, 3-29.
    # %  
    # % Author:    Thor I. Fossen
    # % Date:      2021-12-17
    # % Revisions: 


    nabla = m / RHO;                       # Volume displacement (m^3)
    A11 = 2.7 * RHO * nabla^(5/3) / L^2;   # Added mass in surge (kg)
    ratio = A11 / m;                       # Ratio of added mass to actual mass

    return A11, ratio