from engines.numpy_engine import NumpyEngine
from integrators.heun import HeunIntegrator
from models.jr import JRParams, JRModel
from models.mpr import MPRParams, MPRModel
from pipeline.manager import PipelineManager

# Engine & integrator
engine = NumpyEngine()
integrator = HeunIntegrator()

# JR setup
jr_params = JRParams(num_nodes=5, t_end=200.0, plasticity=True, adaptive_noise=True)
jr_model = JRModel(jr_params, engine)

# MPR setup
mpr_params = MPRParams(num_nodes=5, t_end=200.0, plasticity=True, adaptive_noise=True)
mpr_model = MPRModel(mpr_params, engine)

# Pipeline
pipeline = PipelineManager()
pipeline.add(jr_model, 'JRModel')
pipeline.add(mpr_model, 'MPRModel')

# Run
results = pipeline.run(t_end=200.0, dt=0.01, integrator=integrator)

# Access outputs
jr_traj = results['JRModel']
mpr_traj = results['MPRModel']['trajectory']
bold = results['MPRModel']['bold']

print("JR trajectory shape:", jr_traj.shape)
print("MPR trajectory shape:", mpr_traj.shape)
print("BOLD shape:", bold.shape)
