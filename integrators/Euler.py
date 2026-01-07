import numpy as np
from .base import Integrator

class EulerIntegrator(Integrator):
    def step(self, model, state, t, dt):
        dx = model.f_sys(state, t)
        return state + dt * dx
