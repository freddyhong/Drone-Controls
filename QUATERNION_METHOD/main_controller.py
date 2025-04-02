"""
    - Main controller to test the autopilot implementation
    - Implements both lateral and longitudinal control using successive loop closure
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from message_types.msg_autopilot import MsgAutopilot
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from models.trim import compute_trim
from controllers.autopilot import Autopilot
from tools.rotations import quaternion_to_euler

# Simulation parameters
dt = 0.01  # time step
sim_time = 50.0  # total simulation time
num_steps = int(sim_time / dt)
t = np.linspace(0, sim_time, num_steps)

# Initialize MAV and wind simulation
mav = MavDynamics(Ts=dt)
wind_sim = WindSimulation(Ts=dt, gust_flag=False, steady_state=np.array([[0., 0., 0.]]).T) # no wind nor gust

# Trim conditions
Va = 25.0  # desired airspeed (m/s)
gamma = np.radians(0.0)  # desired flight path angle
wind_trim = wind_sim.update()  # get current wind
trim_state, trim_input = compute_trim(mav, Va, gamma, wind=wind_trim)

# Set initial conditions
mav._state = trim_state
delta = MsgDelta(
    elevator=trim_input.elevator,
    aileron=trim_input.aileron,
    rudder=trim_input.rudder,
    throttle=trim_input.throttle
)

# Initialize autopilot
autopilot = Autopilot(ts_control=dt)

# Create autopilot commands
autopilot_cmd = MsgAutopilot()
autopilot_cmd.altitude_command = -100.0  # initial altitude (NED frame, negative for up)
autopilot_cmd.airspeed_command = Va
autopilot_cmd.course_command = np.radians(0.0)

# Data storage
state_history = np.zeros((num_steps, 13))
cmd_history = np.zeros((num_steps, 3))  # altitude, airspeed, course
delta_history = np.zeros((num_steps, 4))  # elevator, aileron, rudder, throttle

# Main simulation loop
for i in range(num_steps):
    # Update wind
    wind = wind_sim.update()
    
    # Update autopilot commands at specific times
    if t[i] >= 10.0 and t[i] < 20.0:
        autopilot_cmd.altitude_command = -150.0  # step up in altitude
    elif t[i] >= 20.0 and t[i] < 30.0:
        autopilot_cmd.altitude_command = -50.0  # step down in altitude
    elif t[i] >= 30.0 and t[i] < 40.0:
        autopilot_cmd.course_command = np.radians(45.0)  # turn right
    elif t[i] >= 40.0 and t[i] < 50.0:
        autopilot_cmd.course_command = np.radians(-225.0)  # test >180° wrap (left turn)
    elif t[i] >= 50.0:
        autopilot_cmd.airspeed_command = 30.0  # increase airspeed
    
    # Update autopilot
    delta, cmd_state = autopilot.update(autopilot_cmd, mav.true_state)
    
    # Update MAV dynamics
    mav.update(delta, wind)
    
    # Store data
    state_history[i, :] = mav._state[:13, 0]
    cmd_history[i, :] = [autopilot_cmd.altitude_command, 
                        autopilot_cmd.airspeed_command, 
                        autopilot_cmd.course_command]
    delta_history[i, :] = [delta.elevator, delta.aileron, delta.rudder, delta.throttle]

# Convert states to more readable variables
north = state_history[:, 0]
east = state_history[:, 1]
down = state_history[:, 2]
u = state_history[:, 3]
v = state_history[:, 4]
w = state_history[:, 5]
e0 = state_history[:, 6]
e1 = state_history[:, 7]
e2 = state_history[:, 8]
e3 = state_history[:, 9]
p = state_history[:, 10]
q = state_history[:, 11]
r = state_history[:, 12]

# Convert quaternions to Euler angles
euler_angles = np.array([quaternion_to_euler(q) for q in state_history[:, 6:10]])
phi, theta, psi = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

# Calculate airspeed and groundspeed
Va = np.sqrt(u**2 + v**2 + w**2)
Vg = np.sqrt((u*np.cos(psi) + v*np.sin(psi))**2 + (v*np.cos(psi) - u*np.sin(psi))**2 + w**2)

# Calculate course angle (chi)
chi = np.arctan2(Vg*np.sin(psi), Vg*np.cos(psi))

# Plot 3D trajectory
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(north, east, -down, label='Flight Path')
ax3d.set_xlabel('North [m]')
ax3d.set_ylabel('East [m]')
ax3d.set_zlabel('Altitude [m]')
ax3d.set_title('3D Flight Trajectory')
ax3d.legend()
plt.tight_layout()

# Plot position and attitude
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8))
ax1[0].plot(t, north, label='North')
ax1[0].plot(t, east, label='East')
ax1[0].plot(t, -down, label='Altitude')
ax1[0].plot(t, cmd_history[:, 0], '--', label='Cmd Altitude')
ax1[0].set_ylabel('Position [m]')
ax1[0].legend()

ax1[1].plot(t, np.degrees(phi), label='Roll (φ)')
ax1[1].plot(t, np.degrees(theta), label='Pitch (θ)')
ax1[1].plot(t, np.degrees(psi), label='Yaw (ψ)')
ax1[1].set_ylabel('Attitude [deg]')
ax1[1].legend()

ax1[2].plot(t, np.degrees(chi), label='Course (χ)')
ax1[2].plot(t, np.degrees(cmd_history[:, 2]), '--', label='Cmd Course')
ax1[2].set_xlabel('Time [s]')
ax1[2].set_ylabel('Course [deg]')
ax1[2].legend()
fig1.suptitle('Position and Attitude Response')

# Plot velocities and control inputs
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8))
ax2[0].plot(t, Va, label='Airspeed (Va)')
ax2[0].plot(t, Vg, label='Groundspeed (Vg)')
ax2[0].plot(t, cmd_history[:, 1], '--', label='Cmd Airspeed')
ax2[0].set_ylabel('Speed [m/s]')
ax2[0].legend()

ax2[1].plot(t, np.degrees(p), label='Roll rate (p)')
ax2[1].plot(t, np.degrees(q), label='Pitch rate (q)')
ax2[1].plot(t, np.degrees(r), label='Yaw rate (r)')
ax2[1].set_ylabel('Angular rates [deg/s]')
ax2[1].legend()

ax2[2].plot(t, np.degrees(delta_history[:, 0]), label='Elevator')
ax2[2].plot(t, np.degrees(delta_history[:, 1]), label='Aileron')
ax2[2].plot(t, np.degrees(delta_history[:, 2]), label='Rudder')
ax2[2].plot(t, delta_history[:, 3], label='Throttle')
ax2[2].set_xlabel('Time [s]')
ax2[2].set_ylabel('Control Inputs')
ax2[2].legend()
fig2.suptitle('Velocities and Control Inputs')

plt.show()