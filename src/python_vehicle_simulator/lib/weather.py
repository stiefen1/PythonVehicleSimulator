from math import cos, sin

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

    def u(self, yaw:float) -> float:
        return self.norm * cos(self.beta - yaw)
    
    def v(self, yaw:float) -> float:
        return self.norm * sin(self.beta - yaw)
        
    @property
    def v_east(self) -> float:
        return sin(self.beta) * self.v
    
    @property
    def v_north(self) -> float:
        return cos(self.beta) * self.v

class Wind(UniformVectorField):
    def __init__(
            self,
            beta:float, # clockwise positive w.r.t north (bearing angle) in radians
            v:float, # m/s
            *args,
            **kwargs
    ):
        super().__init__(beta, v, *args, **kwargs)
        
    
class Current(UniformVectorField):
    def __init__(
            self,
            beta:float, # clockwise positive w.r.t north (bearing angle) in radians
            v:float, # m/s
            *args,
            **kwargs
    ):
        super().__init__(beta, v, *args, **kwargs)

    @property
    def v_east(self) -> float:
        return sin(self.beta) * self.v
    
    @property
    def v_north(self) -> float:
        return cos(self.beta) * self.v