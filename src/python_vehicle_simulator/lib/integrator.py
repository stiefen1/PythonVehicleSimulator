import numpy as np

#------------------------------------------------------------------------------

def Euler(x:np.ndarray, x_dot:np.ndarray, dt:float) -> np.ndarray:
    """
    Forward Euler integration
    """
    return x + x_dot * dt

