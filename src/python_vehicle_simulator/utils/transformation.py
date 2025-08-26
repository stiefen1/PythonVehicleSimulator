import numpy as np
from math import pi, cos, sin

D2R = pi/180
R2D = 180/pi

def R_body_to_ned(roll:float, pitch:float, yaw:float) -> np.array:
    """
    Returns rotation matrix to fransform vector from body frame (u, v, r) to NED frame.
    psi in RADIANS
    """
    cphi, sphi = cos(roll), sin(roll)
    cth, sth = cos(pitch), sin(pitch)
    cpsi, spsi = cos(yaw), sin(yaw)
    
    return np.array([
        [cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
        [spsi*cth, cpsi*cphi + sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
        [-sth, cth*sphi, cth*cphi]
    ])