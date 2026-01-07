from dataclasses import dataclass
import numpy as np
from .base import BaseModel

@dataclass
class JRParams:
    A: float = 3.25
    B: float = 22.0
    a: float = 0.1
    b: float = 0.05
    v0: float = 6.0
    r: float = 0.56
    vmax: float = 0.005
    dt: float = 0.01
    t_end: float = 1000.0
    G: float = 1.0
    noise_amp: float = 0.01
    seed: int = None
    plasticity: bool = True
    adaptive_noise: bool = True
    num_nodes: int = 1

class JRModel(BaseModel):
    def __init__(self, params: JRParams, engine):
        super().__init__(engine)
        self.params = params
        self.set_initial_state(params.seed)

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.engine.randn((6, self.params.num_nodes)) * 0.01

    def S(self, x):
        """Sigmoid"""
        return self.params.vmax / (1 + np.exp(self.params.r*(self.params.v0 - x)))

    def f_sys(self, x, t):
        """Compute derivatives with optional plasticity & adaptive noise"""
        dx = np.zeros_like(x)
        # Simplified JR dynamics example
        A, B, a, b = self.params.A, self.params.B, self.params.a, self.params.b
        dx[0] = x[3]
        dx[3] = A*a*self.S(x[1] - x[2]) - 2*a*x[3] - a**2*x[0]

        # Add plasticity: simple Hebbian adjustment on C0
        if self.params.plasticity:
            # pretend state[1] is excitatory population
            self.params.A += 0.0001 * (x[1]**2)

        # Add adaptive noise
        if self.params.adaptive_noise:
            dx += self.params.noise_amp * self.engine.randn(x.shape)

        return dx

    def run(self, t_end=None, dt=None, integrator=None):
        t_end = t_end or self.params.t_end
        dt = dt or self.params.dt
        steps = int(t_end / dt)
        trajectory = []
        t_values = []

        x = self.state.copy()
        for i in range(steps):
            t = i*dt
            x = integrator.step(self, x, t, dt)
            trajectory.append(x.copy())
            t_values.append(t)

        self.state = x
        return np.array(t_values), np.array(trajectory)
