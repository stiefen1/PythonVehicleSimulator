from math import cos, sin, pi
from python_vehicle_simulator.utils.math_fn import ssa
import numpy as np
from matplotlib.axes import Axes

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

    def plot(self, ax: Axes, color: str ='blue') -> Axes:
        # Get current axis limits to position arrow in top-left
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate axis dimensions
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        
        # Position arrow in top-left corner (10% margin from edges)
        arrow_x = xlim[0] + 0.1 * width
        arrow_y = ylim[1] - 0.1 * height
        
        # Scale arrow to be 1/10 of axis size, maintaining direction
        arrow_scale = 0.1 * min(width, height)
        if self.norm > 0:
            arrow_dx = arrow_scale * (self.v_east / self.norm)
            arrow_dy = arrow_scale * (self.v_north / self.norm)
        else:
            arrow_dx = arrow_dy = 0
        
        # Draw arrow with fixed visual size
        ax.annotate('', xy=(arrow_x + arrow_dx, arrow_y + arrow_dy), 
                    xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2), label=type(self).__name__)
        
        # Add speed text near the arrow
        ax.text(arrow_x + arrow_dx + 0.02 * width, arrow_y + arrow_dy + 0.02 * height, 
                f'{type(self).__name__}: {self.norm:.1f} m/s', 
                fontsize=10, color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8), label=type(self).__name__)
        return ax
        
    @property
    def v_east(self) -> float:
        return sin(self.beta) * self.norm
    
    @property
    def v_north(self) -> float:
        return cos(self.beta) * self.norm

    @property
    def beta(self) -> float:
        return self._beta
    
    @property
    def norm(self) -> float:
        return self._norm
    
    @beta.setter
    def beta(self, val: float) -> None:
        self._beta = ssa(val)

    @norm.setter
    def norm(self, val: float) -> None:
        if val < 0: # If norm is negative, it means wind direction changes
            self.beta = self._beta + pi
        self._norm = abs(val)

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