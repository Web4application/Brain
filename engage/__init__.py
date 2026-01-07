# brain_sim_v4_interactive_gpu.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

# ==========================
# Engine abstraction
# ==========================
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class Engine:
    def __init__(self, device='gpu'):
        self.device = device.lower()
        if self.device == 'gpu' and not GPU_AVAILABLE:
            print("CuPy not installed. Falling back to CPU.")
            self.device = 'cpu'

    def array(self, x):
        return cp.array(x) if self.device == 'gpu' else np.array(x)

    def randn(self, shape):
        return cp.random.randn(*shape) if self.device == 'gpu' else np.random.randn(*shape)

    def to_host(self, x):
        return cp.asnumpy(x) if self.device == 'gpu' else x

# ==========================
# Integrator
# ==========================
class HeunIntegrator:
    def step(self, model, x, t, dt):
        k1 = model.f_sys(x, t)
        k2 = model.f_sys(x + dt*k1, t + dt)
        return x + dt/2*(k1 + k2)

# ==========================
# JR Model
# ==========================
class JRModel:
    def __init__(self, num_nodes=1000, engine=None):
        self.num_nodes = num_nodes
        self.engine = engine
        self.A = 3.25
        self.B = 22.0
        self.a = 0.1
        self.b = 0.05
        self.v0 = 6.0
        self.r = 0.56
        self.vmax = 0.005
        self.noise_amp = 0.01
        self.set_initial_state()

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.engine.randn((6, self.num_nodes)) * 0.01

    def S(self, x):
        return self.vmax / (1 + np.exp(self.r*(self.v0 - x)))

    def f_sys(self, x, t):
        dx = np.zeros_like(x)
        dx[0] = x[3]
        dx[3] = self.A*self.a*self.S(x[1]-x[2]) - 2*self.a*x[3] - self.a**2*x[0]
        dx += self.noise_amp * self.engine.randn(x.shape)
        return dx

# ==========================
# MPR Model
# ==========================
class MPRModel:
    def __init__(self, num_nodes=1000, engine=None):
        self.num_nodes = num_nodes
        self.engine = engine
        self.J = 15.0
        self.eta = 0.0
        self.delta = 1.0
        self.tau = 1.0
        self.noise_amp = 0.01
        self.set_initial_state()

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.engine.randn((2, self.num_nodes)) * 0.01

    def f_sys(self, x, t):
        dx = np.zeros_like(x)
        r, v = x[0], x[1]
        dx[0] = (self.delta/np.pi + 2*r*v)/self.tau
        dx[1] = v**2 + self.eta + self.J*r - self.tau**2 * r
        dx += self.noise_amp * self.engine.randn(x.shape)
        return dx

    def compute_bold(self, r):
        return np.convolve(r.flatten(), np.exp(-np.linspace(0,5,50)), mode='same')

# ==========================
# Interactive Pipeline
# ==========================
class BrainPipeline:
    def __init__(self, jr_model, mpr_model, integrator, t_end=100.0, dt=0.01):
        self.jr_model = jr_model
        self.mpr_model = mpr_model
        self.integrator = integrator
        self.t_end = t_end
        self.dt = dt
        self.steps = int(t_end/dt)
        self.jr_traj = []
        self.mpr_traj = []
        self.bold_traj = []

        # Setup figure
        self.fig, self.ax = plt.subplots(3,1, figsize=(12,8))
        plt.subplots_adjust(bottom=0.25)
        self.lines = [self.ax[i].plot([], [])[0] for i in range(3)]
        self.ax[0].set_title("JR excitatory")
        self.ax[1].set_title("MPR firing rate")
        self.ax[2].set_title("BOLD")

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        self.slider_A = Slider(plt.axes([0.15,0.1,0.65,0.03], facecolor=axcolor), 'JR A', 0.1, 10.0, valinit=self.jr_model.A)
        self.slider_J = Slider(plt.axes([0.15,0.05,0.65,0.03], facecolor=axcolor), 'MPR J', 0.1, 20.0, valinit=self.mpr_model.J))
        self.slider_A.on_changed(self.update_params)
        self.slider_J.on_changed(self.update_params)

        # Run button
        self.button = Button(plt.axes([0.85, 0.025, 0.1, 0.04]), 'Reset')
        self.button.on_clicked(self.reset)

    def update_params(self, val):
        self.jr_model.A = self.slider_A.val
        self.mpr_model.J = self.slider_J.val

    def reset(self, event):
        self.jr_model.set_initial_state()
        self.mpr_model.set_initial_state()
        self.jr_traj.clear()
        self.mpr_traj.clear()
        self.bold_traj.clear()

    def animate(self):
        x_jr = self.jr_model.state.copy()
        x_mpr = self.mpr_model.state.copy()

        for i in range(self.steps):
            t = i*self.dt
            x_jr = self.jr_model.f_sys(x_jr, t) * self.dt + x_jr
            self.jr_traj.append(self.jr_model.engine.to_host(x_jr.copy()))
            x_mpr[0] = x_jr[1,:self.mpr_model.num_nodes]
            x_mpr = self.mpr_model.f_sys(x_mpr, t) * self.dt + x_mpr
            self.mpr_traj.append(self.mpr_model.engine.to_host(x_mpr.copy()))
            self.bold_traj.append(self.mpr_model.compute_bold(x_mpr[0]))

            # Update plots
            for idx, line in enumerate(self.lines):
                if idx==0:
                    line.set_data(np.arange(len(self.jr_traj)), np.array(self.jr_traj)[:,1,0])
                    self.ax[idx].relim(); self.ax[idx].autoscale_view()
                elif idx==1:
                    line.set_data(np.arange(len(self.mpr_traj)), np.array(self.mpr_traj)[:,0,0])
                    self.ax[idx].relim(); self.ax[idx].autoscale_view()
                else:
                    line.set_data(np.arange(len(self.bold_traj)), np.array(self.bold_traj))
                    self.ax[idx].relim(); self.ax[idx].autoscale_view()
            plt.pause(0.001)

# ==========================
# Run Example
# ==========================
if __name__ == "__main__":
    engine = Engine('gpu')  # 'gpu' for CuPy
    integrator = HeunIntegrator()

    jr_model = JRModel(num_nodes=1000, engine=engine)
    mpr_model = MPRModel(num_nodes=1000, engine=engine)

    pipeline = BrainPipeline(jr_model, mpr_model, integrator, t_end=50.0, dt=0.01)
    pipeline.animate()
