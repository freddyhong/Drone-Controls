import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics_control import MavDynamics
from message_types.msg_delta import MsgDelta
from mpl_toolkits.mplot3d import Axes3D
from models.wind_simulation import WindSimulation
from models.trim import compute_trim
from tools.rotations import quaternion_to_euler

# testing mavism_ch5_coeff trim conditons values
# Time setup
dt = 0.01
sim_time = 1000  # total simulation time (s)
steps = int(sim_time / dt)

# Initialize MAV
mav = MavDynamics(Ts=dt)

# Set initial state to trimmed state manually put in
trim_state_vector = np.array([
    -0, -0, -100,            # pn, pe, pd
    24.97, 0, 1.1945, # u, v, w
    0.9938, 0.0, 0.1109, 0.0,  # quaternion
    0, 0, 0             # p, q, r
]).reshape((13, 1))
mav._state = trim_state_vector

# Set control input to trimmed control input manually put in
trim_input = MsgDelta()
trim_input.elevator = -0.118662
trim_input.aileron = 0.009775
trim_input.rudder = -0.001611
trim_input.throttle = 0.857721

# Lists to store trajectory
north = []
east = []
altitude = []

# Simulate
for _ in range(steps):
    mav.update(delta=trim_input, wind=np.zeros((6, 1)))
    north.append(mav.true_state.north)
    east.append(mav.true_state.east)
    altitude.append(mav.true_state.altitude)  # Note: positive up

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(north, east, altitude)
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Flight Path from Trim Condition')
plt.tight_layout()
plt.show()
