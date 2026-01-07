from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, engine):
        self.engine = engine

    @abstractmethod
    def set_initial_state(self, seed=None):
        pass

    @abstractmethod
    def f_sys(self, x, t):
        """Compute derivatives for the model"""
        pass

    @abstractmethod
    def run(self, t_end, dt, integrator):
        pass
