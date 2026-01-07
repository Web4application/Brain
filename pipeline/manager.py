import numpy as np

class PipelineManager:
    """Connect multiple models and run them in sequence"""
    def __init__(self):
        self.models = []
        self.connections = []

    def add(self, model, name=None):
        self.models.append((name or str(model), model))

    def connect(self, source, target):
        """source: 'JR.state_index', target: 'MPR.input_index'"""
        self.connections.append((source, target))

    def run(self, t_end, dt, integrator):
        """Run all models with connections"""
        results = {}
        for name, model in self.models:
            if 'JR' in name:
                t, traj = model.run(t_end=t_end, dt=dt, integrator=integrator)
                results[name] = traj
            elif 'MPR' in name:
                # For simplicity, feed mean JR output to MPR
                jr_traj = results.get('JRModel', None)
                if jr_traj is not None:
                    input_to_mpr = np.mean(jr_traj[:,1], axis=1)  # excitatory pop
                    model.state[0] = input_to_mpr.reshape(-1,1)[:model.params.num_nodes]
                t, traj, bold = model.run(t_end=t_end, dt=dt, integrator=integrator)
                results[name] = {'trajectory': traj, 'bold': bold}

        return results
