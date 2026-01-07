from dataclasses import dataclass
import numpy as np
from .base import BaseModel

@dataclass
class MPRParams:
    J: float = 15.0            # Coupling
    eta: float = 0.0            # Input current mean
    delta: float = 1.0          # Width of Lorentzian
    tau: float = 1.0            # Time constant
    dt: float = 0.01
    t_end: float = 1000.0
    noise_amp: float = 0.01
    seed: int = None
    plasticity: bool = True
    adaptive_noise: bool = True
    num_nodes: int = 1
    num_sim: int = 1

class MPRModel(BaseModel):
    def __init__(self, params: MPRParams, engine):
        super().__init__(engine)
        self.params = params
        self.set_initial_state(params.seed)

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # r: mean firing rate, v: mean membrane potential
        self.state = self.engine.randn((2, self.params.num_nodes, self.params.num_sim)) * 0.01

    def f_sys(self, x, t):
        """MPR dynamics with plasticity & adaptive noise"""
        dx = np.zeros_like(x)
        r, v = x[0], x[1]
        tau, J, eta, delta = self.params.tau, self.params.J, self.params.eta, self.params.delta

        dx[0] = (delta / np.pi + 2 * r * v) / tau
        dx[1] = (v**2 + eta + J*r) - tau**2 * r  # simplified dynamics

        # Plasticity: scale J slightly based on activity
        if self.params.plasticity:
            self.params.J += 0.00001 * np.mean(r**2)

        # Adaptive noise
        if self.params.adaptive_noise:
            dx += self.params.noise_amp * self.engine.randn(x.shape)

        return dx

    def compute_bold(self, r):
        """Simple hemodynamic model placeholder"""
        # BOLD = convolve r with HRF (simplified)
        return np.convolve(r.flatten(), np.exp(-np.linspace(0,5,50)), mode='same')

    def run(self, t_end=None, dt=None, integrator=None, record_bold=True):
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
        trajectory = np.array(trajectory)
        t_values = np.array(t_values)

        bold = self.compute_bold(trajectory[:,0]) if record_bold else None
        return t_values, trajectory, bold
