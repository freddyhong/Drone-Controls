"""
    - Main controller to test the autopilot implementation
    - Implements both lateral and longitudinal control using successive loop closure
    - Adjust gains in control_parameters.py 
"""

# using luch's trim states, inputs, and a_phi stuff...


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
from controllers.autopilot import Autopilot
from tools.rotations import quaternion_to_euler

# Direct trim values
x_trim = np.array([[-0.000000, -0.000000, -100.000000, 24.971443, 0.000000, 
                    1.194576, 0.993827, 0.000000, 0.110938, 0.000000, 
                    0.000000, 0.000000, 0.000000]]).T
u_trim = np.array([[-0.118662, 0.009775, -0.001611, 0.857721]]).T
Va_trim = 25.000000

# Simulation parameters
dt = 0.01  # time step
sim_time = 100.0  # total simulation time
num_steps = int(sim_time / dt)
t = np.linspace(0, sim_time, num_steps)

# Initialize MAV
mav = MavDynamics(Ts=dt)

# Set initial conditions using direct trim values
mav._state = x_trim
delta = MsgDelta(
    elevator=u_trim[0,0],  
    aileron=u_trim[1,0],   
    rudder=u_trim[2,0],   
    throttle=u_trim[3,0]  
)

# Initialize autopilot
autopilot = Autopilot(ts_control=dt)

# Create autopilot commands
autopilot_cmd = MsgAutopilot()
autopilot_cmd.altitude_command = x_trim[2,0]  # Use trim altitude
autopilot_cmd.airspeed_command = Va_trim
autopilot_cmd.course_command = np.radians(0.0)

# Data storage
state_history = np.zeros((num_steps, 13))
cmd_history = np.zeros((num_steps, 5))  # altitude, airspeed, course, phi, theta
delta_history = np.zeros((num_steps, 4))  # elevator, aileron, rudder, throttle

# Main simulation loop 
for i in range(num_steps):
    wind = np.array([[0.], [0.], [0.], [0.], [0.], [0.]])

    # Straight/level flight (0-15s)
    if t[i] < 15.0:
        autopilot_cmd.altitude_command = -100.0
        autopilot_cmd.airspeed_command = 25.0
        autopilot_cmd.course_command = np.radians(0.0)
        
    # Gradual climb to 150m (15-30s)
    elif t[i] < 30.0:
        autopilot_cmd.altitude_command = -100 - (50*(t[i]-15)/15)
        autopilot_cmd.airspeed_command = 25.0
        
    # Gentle right turn to 45° (30-45s)
    elif t[i] < 45.0:
        autopilot_cmd.altitude_command = -150.0
        autopilot_cmd.course_command = np.radians(45*(t[i]-30)/15)
        
    # Descend to 100m (45-60s)
    elif t[i] < 60.0:
        autopilot_cmd.altitude_command = -150 + (50*(t[i]-45)/15)
        autopilot_cmd.course_command = np.radians(45)
        
    # Turn left to -90° (60-75s)
    elif t[i] < 75.0:
        autopilot_cmd.altitude_command = -100.0
        autopilot_cmd.airspeed_command = 25 + 5*(t[i]-60)/15
        autopilot_cmd.course_command = np.radians(45 - 135*(t[i]-60)/15)
        
    # Final cruise (75-100s)
    else:
        autopilot_cmd.airspeed_command = 30.0
    
    delta, cmd_state = autopilot.update(autopilot_cmd, mav.true_state)
    mav.update(delta, wind)  # No wind for now
    
    # Store results
    state_history[i, :] = mav._state[:13, 0]
    cmd_history[i, :] = [autopilot_cmd.altitude_command, 
                        autopilot_cmd.airspeed_command, 
                        autopilot_cmd.course_command,
                        cmd_state.phi,
                        cmd_state.theta]
    delta_history[i, :] = [delta.elevator, delta.aileron, delta.rudder, delta.throttle]

# Post-processing
euler_angles = np.array([quaternion_to_euler(q) for q in state_history[:, 6:10]])
phi, theta, psi = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
u, v, w = state_history[:, 3], state_history[:, 4], state_history[:, 5]
Va = np.sqrt(u**2 + v**2 + w**2)
chi = np.arctan2(np.sin(psi), np.cos(psi))  # Course angle

# Plotting functions
def plot_3d_trajectory():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_history[:, 1], state_history[:, 0], -state_history[:, 2])
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title('3D Flight Trajectory')
    plt.tight_layout()
    return fig

def plot_position_attitude():
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    
    ax[0].plot(t, state_history[:, 0], label='North')
    ax[0].plot(t, state_history[:, 1], label='East')
    ax[0].plot(t, -state_history[:, 2], label='Altitude')
    ax[0].plot(t, cmd_history[:, 0], '--', label='Cmd Altitude')
    ax[0].set_ylabel('Position [m]')
    ax[0].legend()
    
    ax[1].plot(t, np.degrees(phi), label='Roll (φ)')
    ax[1].plot(t, np.degrees(cmd_history[:,3]), '--', label='Cmd Roll')
    ax[1].plot(t, np.degrees(theta), label='Pitch (θ)')
    ax[1].plot(t, np.degrees(cmd_history[:,4]), '--', label='Cmd Pitch')
    ax[1].set_ylabel('Attitude [deg]')
    ax[1].legend()
    
    ax[2].plot(t, np.degrees(chi), label='Course (χ)')
    ax[2].plot(t, np.degrees(cmd_history[:, 2]), '--', label='Cmd Course')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Course [deg]')
    ax[2].legend()
    
    fig.suptitle('Position and Attitude Response')
    return fig

def plot_velocities_inputs():
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    
    ax[0].plot(t, Va, label='Airspeed (Va)')
    ax[0].plot(t, cmd_history[:, 1], '--', label='Cmd Airspeed')
    ax[0].set_ylabel('Speed [m/s]')
    ax[0].legend()
    
    ax[1].plot(t, np.degrees(state_history[:, 10]), label='Roll rate (p)')
    ax[1].plot(t, np.degrees(state_history[:, 11]), label='Pitch rate (q)')
    ax[1].plot(t, np.degrees(state_history[:, 12]), label='Yaw rate (r)')
    ax[1].set_ylabel('Angular rates [deg/s]')
    ax[1].legend()
    
    ax[2].plot(t, np.degrees(delta_history[:, 0]), label='Elevator')
    ax[2].plot(t, np.degrees(delta_history[:, 1]), label='Aileron')
    ax[2].plot(t, np.degrees(delta_history[:, 2]), label='Rudder')
    ax[2].plot(t, delta_history[:, 3], label='Throttle')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Control Inputs')
    ax[2].legend()
    
    fig.suptitle('Velocities and Control Inputs')
    return fig

# Generate plots
plot_3d_trajectory()
plot_position_attitude()
plot_velocities_inputs()
plt.show()