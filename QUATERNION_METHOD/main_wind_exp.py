import sys
import os

# Add the parent directory of 'models' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from models.trim import compute_trim
from message_types.msg_delta import MsgDelta
from tools.rotations import quaternion_to_euler

# === Configuration ===
dt = 0.01
num_steps = 10000
t = np.linspace(0, dt * num_steps, num_steps)
Va = 25.0

trim_log = {}

def simulate_case(case_name, wind_vec, gust_flag):
    print(f"\n=== Running Case: {case_name} ===")

    # Initialize MAV and Wind
    mav = MavDynamics(Ts=dt)
    wind_sim = WindSimulation(Ts=dt, gust_flag=gust_flag, steady_state=wind_vec[:3])

    # Compute trim with wind
    trim_state, trim_input = compute_trim(mav, Va, gamma=0.0, wind=wind_vec)
    trim_log[case_name] = (trim_state, trim_input)

    mav._state = trim_state
    delta = MsgDelta()
    delta.from_array(np.array([
        trim_input.elevator,
        trim_input.aileron,
        trim_input.rudder,
        trim_input.throttle
    ]))

    # Storage
    state_history = np.zeros((num_steps, 13))
    wind_history = np.zeros((num_steps, 6))
    euler_history = np.zeros((num_steps, 3))

    # Run sim
    for i in range(num_steps):
        wind = wind_sim.update()
        mav.update(delta, wind)
        state_history[i, :] = mav._state[:13, 0]
        wind_history[i, :] = wind[:6, 0]
        euler = quaternion_to_euler(mav._state[6:10])
        euler_history[i, :] = euler

    return state_history, wind_history, euler_history

# === Define Test Cases ===
cases = {
    "No Wind": (np.zeros((6, 1)), False),
    "Steady Wind (5 m/s Down)": (np.array([[0.], [0.], [5.], [0.], [0.], [0.]]), False),
    "Steady + Gust": (np.array([[0.], [0.], [5.], [0.], [0.], [0.]]), True)
}

results = {}
for case_name, (wind_vec, gust_flag) in cases.items():
    states, winds, eulers = simulate_case(case_name, wind_vec, gust_flag)
    results[case_name] = (states, winds, eulers)

# === Print Final Trim States ===
print("\n=== Final Trim States and Inputs ===")
for case_name, (trim_state, trim_input) in trim_log.items():
    print(f"\n--- {case_name} ---")
    print("Trim State:", trim_state.T)
    print("Trim Inputs:", trim_input.to_array().T)

# === 3D Flight Path ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for case_name, (states, _, _) in results.items():
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label=case_name)
ax.set_xlabel("North (m)")
ax.set_ylabel("East (m)")
ax.set_zlabel("Down (m)")
ax.set_title("3D Flight Path")
ax.legend()
plt.tight_layout()
plt.show()

# === Angular Position (Euler Angles) ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
for case_name, (_, _, eulers) in results.items():
    axs[0].plot(t, eulers[:, 0], label=case_name)
    axs[1].plot(t, eulers[:, 1], label=case_name)
    axs[2].plot(t, eulers[:, 2], label=case_name)
axs[0].set_ylabel("phi (rad)")
axs[1].set_ylabel("theta (rad)")
axs[2].set_ylabel("psi (rad)")
axs[2].set_xlabel("Time (s)")
axs[0].set_title("Angular Position (Euler Angles)")
for ax in axs: ax.legend()
plt.tight_layout()
plt.show()

# === Angular Rates (p, q, r) ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
for case_name, (states, _, _) in results.items():
    axs[0].plot(t, states[:, 10], label=case_name)
    axs[1].plot(t, states[:, 11], label=case_name)
    axs[2].plot(t, states[:, 12], label=case_name)
axs[0].set_ylabel("p (rad/s)")
axs[1].set_ylabel("q (rad/s)")
axs[2].set_ylabel("r (rad/s)")
axs[2].set_xlabel("Time (s)")
axs[0].set_title("Angular Rates Over Time")
for ax in axs: ax.legend()
plt.tight_layout()
plt.show()

# === Body Velocities (u, v, w) ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
for case_name, (states, _, _) in results.items():
    axs[0].plot(t, states[:, 3], label=case_name)
    axs[1].plot(t, states[:, 4], label=case_name)
    axs[2].plot(t, states[:, 5], label=case_name)
axs[0].set_ylabel("u (m/s)")
axs[1].set_ylabel("v (m/s)")
axs[2].set_ylabel("w (m/s)")
axs[2].set_xlabel("Time (s)")
axs[0].set_title("Body Velocities Over Time")
for ax in axs: ax.legend()
plt.tight_layout()
plt.show()

# === Wind Components ===
fig, axs = plt.subplots(2, 1, figsize=(12, 8))
for case_name, (_, winds, _) in results.items():
    axs[0].plot(t, winds[:, 0], label=f'{case_name} - u_s')
    axs[0].plot(t, winds[:, 1], label=f'{case_name} - v_s')
    axs[0].plot(t, winds[:, 2], label=f'{case_name} - w_s')
    axs[0].plot(t, winds[:, 3], label=f'{case_name} - u_g')
    axs[0].plot(t, winds[:, 4], label=f'{case_name} - v_g')
    axs[0].plot(t, winds[:, 5], label=f'{case_name} - w_g')
    wind_mag = np.linalg.norm(winds[:, 0:3] + winds[:, 3:6], axis=1)
    axs[1].plot(t, wind_mag, label=case_name)
axs[0].set_title("Wind Components")
axs[0].legend()
axs[1].set_title("Total Wind Magnitude")
axs[1].legend()
plt.tight_layout()
plt.show()
