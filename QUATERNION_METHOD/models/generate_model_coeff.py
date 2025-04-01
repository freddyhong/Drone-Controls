'''DO NOT USE YET'''

from models.mav_dynamics_control import MavDynamics
from models.trim import compute_trim
from models.compute_models import compute_model
from parameters.simulation_parameters import ts_simulation

# Create MAV object
mav = MavDynamics(ts_simulation)

# Set desired trim conditions
Va_trim = 25.0  # desired airspeed
gamma_trim = 0.0  # level flight

# Compute trimmed state and input
trim_state, trim_input = compute_trim(mav, Va_trim, gamma_trim)

# Generate model coefficient file
compute_model(mav, trim_state, trim_input)

print("Model coefficients saved to models/model_coef.py")
