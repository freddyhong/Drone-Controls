import numpy as np
from Q2 import MAV  
import parameters as par
from compute_trim import compute_trim 

mav = MAV(
    ts=par.ts_simulation, 
    mass=par.mass, 
    Jx=par.Jx, 
    Jy=par.Jy, 
    Jz=par.Jz, 
    Jxz=par.Jxz, 
    initial_state=par.initial_state
)

Va = 25  # m/s
gamma = 0  # Level flight

# Compute trim conditions
trim_state, trim_input = compute_trim(mav, Va, gamma)

# Print results
print("\n--- Trim Results ---")
print("Trimmed State:\n", trim_state)
trim_input.print()