import numpy as np
from math import pi, cos, sin

D2R = pi/180
R2D = 180/pi

def R_body_to_ned(psi:float, dim:int=3) -> np.array:
    """
    Returns rotation matrix to fransform vector from body frame (u, v, r) to NED frame.
    psi in RADIANS
    """
    return np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])[0:dim, 0:dim]