from math import cos, sin, pi
from python_vehicle_simulator.utils.math_fn import ssa
from sys import float_repr_style
import numpy as np

RHO_AIR_AS_FUNC_OF_TEMP = { # From Handbook of Marine Craft Hydrodynamics and Motion Control, p.190
    -10: 1.342,
    -5: 1.317,
    0: 1.292,
    5: 1.269,
    10: 1.247,
    15: 1.225,
    20: 1.204,
    25: 1.184,
    30: 1.165
}

def get_air_density(temperature: float) -> float:
    """
    Get air density for a given temperature using linear interpolation.
    
    Parameters:
    temperature: Temperature in degrees Celsius
    
    Returns:
    Air density in kg/m³
    """
    temps = np.array(list(RHO_AIR_AS_FUNC_OF_TEMP.keys()))
    densities = np.array(list(RHO_AIR_AS_FUNC_OF_TEMP.values()))
    
    # Clamp temperature to available range
    temp_clamped = np.clip(temperature, temps.min(), temps.max())
    
    # Interpolate
    return float(np.interp(temp_clamped, temps, densities))

class UniformVectorField:
    def __init__(
            self,
            beta:float, # rad - clockwise positive w.r.t north (bearing angle) 
            norm:float, # m/s
            *args,
            **kwargs
    ):
        self.beta = beta
        self.norm = norm

    def beta_in_vessel(self, yaw:float) -> float:
        """yaw in radians"""
        return yaw - self.beta
    
    def gamma_w(self, yaw:float) -> float:
        """Angle of attack gamma relative to the bow"""
        return ssa(yaw - self.beta - pi)

    def u(self, yaw:float) -> float:
        return self.norm * cos(self.beta - yaw)
    
    def v(self, yaw:float) -> float:
        return self.norm * sin(self.beta - yaw)
        
    @property
    def v_east(self) -> float:
        return sin(self.beta) * self.norm
    
    @property
    def v_north(self) -> float:
        return cos(self.beta) * self.norm

class Wind(UniformVectorField):
    def __init__(
            self,
            beta:float, # clockwise positive w.r.t north (bearing angle) in radians
            v:float, # m/s
            *args,
            temperature: float = 10,
            **kwargs
    ):
        self.temperature = temperature
        super().__init__(beta, v, *args, **kwargs)

    def get_air_density(self) -> float:
        return get_air_density(self.temperature)
    
class Current(UniformVectorField):
    def __init__(
            self,
            beta:float, # clockwise positive w.r.t north (bearing angle) in radians
            v:float, # m/s
            *args,
            **kwargs
    ):
        super().__init__(beta, v, *args, **kwargs)

    
if __name__ == "__main__":
    # Test original dictionary access
    print(f"Air density at 0°C (dict): {RHO_AIR_AS_FUNC_OF_TEMP[0]}")
    
    # Create simple plot
    import matplotlib.pyplot as plt
    
    # Original data points
    temps = list(RHO_AIR_AS_FUNC_OF_TEMP.keys())
    densities = list(RHO_AIR_AS_FUNC_OF_TEMP.values())
    
    # Interpolated curve
    temp_range = np.linspace(-15, 35, 100)
    density_interp = [get_air_density(t) for t in temp_range]
    
    plt.figure(figsize=(8, 6))
    plt.plot(temps, densities, 'ro', markersize=8, label='Original data')
    plt.plot(temp_range, density_interp, 'b-', linewidth=2, label='Interpolated')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Air Density (kg/m³)')
    plt.title('Air Density vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()