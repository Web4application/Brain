from abc import ABC, abstractmethod

class Integrator(ABC):
    @abstractmethod
    def step(self, model, state, t, dt):
        """Perform one integration step."""
        pass
