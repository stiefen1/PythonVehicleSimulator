from math import pi

KNOT_TO_M_PER_SEC = 0.514444
DEG2RAD = pi/180
RAD2DEG = 1/DEG2RAD


def knot_to_m_per_sec(speed:float) -> float:
    return speed*KNOT_TO_M_PER_SEC

def m_per_sec_to_knot(speed:float) -> float:
    return speed/KNOT_TO_M_PER_SEC