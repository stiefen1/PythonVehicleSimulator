from math import cos, sin, pi
from python_vehicle_simulator.utils.math_fn import ssa
import numpy as np, gymnasium as gym
from matplotlib.axes import Axes
from typing import Optional, Dict, Tuple

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
            attraction_beta: float = 0,
            amplitude_beta: float = 0,
            attraction_norm: float = 0,
            amplitude_norm: float = 0,
            dt: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        self._beta_0 = beta # Considered as mean if Ornstein-Uhlenbeck process is used
        self._norm_0 = norm

        # Ornstein-Uhlenbeck parameters
        self.ornstein_uhlenbeck_beta = attraction_beta > 0 and amplitude_beta > 0 and dt is not None
        self.attraction_beta = attraction_beta
        self.amplitude_beta = amplitude_beta
        self.ornstein_uhlenbeck_norm = attraction_norm > 0 and amplitude_norm > 0 and dt is not None
        self.attraction_norm = attraction_norm
        self.amplitude_norm = amplitude_norm
        self.dt = dt
        self.reset(seed=seed)

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
    
    def step(self) -> Dict:
        if self.ornstein_uhlenbeck_beta:
            dbeta = self.attraction_beta * (self._beta_0 - self._beta) * self.dt + self.amplitude_beta * self.np_random.normal(0, 1) * self.dt # type: ignore
            self.beta = self._beta + dbeta
        if self.ornstein_uhlenbeck_norm:
            dnorm = self.attraction_norm * (self._norm_0 - self._norm) * self.dt + self.amplitude_norm * self.np_random.normal(0, 1) * self.dt # type: ignore
            self.norm = self._norm + dnorm
        return {}
    
    def reset(self, seed: Optional[int] = None) -> None:
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore
        self.beta = self._beta_0 # Actual values returned (updated using the step() method if stochastic process is used)
        self.norm = self._norm_0

    def get_arrow_coords(self, ax: Axes, offset_factor: float = 0.0) -> Tuple[float, float, float, float]:
        """Get arrow coordinates for plotting in axis coordinates (0-1). Returns (start_x, start_y, end_x, end_y)"""
        # Fixed position in axis coordinates (independent of data limits)
        arrow_x = 0.1  # 10% from left edge
        arrow_y = 0.9 - offset_factor * 0.15  # 10% from top, with offset for multiple arrows
        
        # Fixed arrow scale in axis coordinates
        arrow_scale = 0.1
        if self.norm > 0:
            # Normalize direction vector
            arrow_dx = arrow_scale * (self.v_east / self.norm)
            arrow_dy = arrow_scale * (self.v_north / self.norm)
        else:
            arrow_dx = arrow_dy = 0
        
        return arrow_x, arrow_y, arrow_x + arrow_dx, arrow_y + arrow_dy

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
            attraction_beta: float = 0,
            amplitude_beta: float = 0,
            attraction_norm: float = 0,
            amplitude_norm: float = 0,
            dt: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        self.temperature = temperature
        super().__init__(
            beta,
            v,
            *args,
            attraction_beta = attraction_beta,
            amplitude_beta = amplitude_beta,
            attraction_norm = attraction_norm,
            amplitude_norm = amplitude_norm,
            dt = dt,
            seed = seed,
            **kwargs
        )

    def get_air_density(self) -> float:
        return get_air_density(self.temperature)
    
class Current(UniformVectorField):
    def __init__(
            self,
            beta:float, # clockwise positive w.r.t north (bearing angle) in radians
            v:float, # m/s
            *args,
            attraction_beta: float = 0,
            amplitude_beta: float = 0,
            attraction_norm: float = 0,
            amplitude_norm: float = 0,
            dt: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            beta, 
            v, 
            *args,
            attraction_beta = attraction_beta,
            amplitude_beta = amplitude_beta,
            attraction_norm = attraction_norm,
            amplitude_norm = amplitude_norm,
            dt = dt,
            seed = seed,
            **kwargs)

    
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


    # vector_field = Wind(0, 10, attraction_beta=0.01, amplitude_beta=0.05, attraction_norm=0.01, amplitude_norm=0.1, dt=1)
    vector_field = Current(0, 10, attraction_beta=0.01, amplitude_beta=0.05, attraction_norm=0.01, amplitude_norm=0.1, dt=1)
    # vector_field = Current(0, 10) # Static vector field

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    
    for i in range(20):
        ax.cla()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title(f'                       Step {i} (β={vector_field.beta:.2f} rad)')
        vector_field.plot(ax)
        plt.pause(0.01)
        vector_field.step()