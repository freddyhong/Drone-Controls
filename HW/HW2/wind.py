import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aerosonde_parameters import *

initial_state = np.array([0, 0, 10, 0, 0, 0, 1, 0, 2, 10, 2, 2])  # [north, east, down, u, v, w, p, q, r, phi, theta, psi]
wind = np.array([[5.0], [0.0], [0.0], [1.0], [0.0], [0.0]])  # example wind vector (steady-state + gust)
delta = np.array([0, 0, 0, 0.5])  # example control inputs

def update_velocity_data(state, wind=np.zeros((6, 1))):
    steady_state = wind[0:3]  # in NED
    gust = wind[3:6]  # in body frame

    phi, theta, psi = np.radians(state[9:12])

    # Rotation matrix from body to NED
    R = np.array([
        [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
        [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
         np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
         np.sin(phi) * np.cos(theta)],
        [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
         np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
         np.cos(phi) * np.cos(theta)]
    ])

    # convert steady-state wind vector from NED to body frame
    wind_body_steady = R.T @ steady_state
    # add the gust
    wind_body = wind_body_steady + gust
    # Convert total wind to NED frame
    wind_ned = R @ wind_body

    # velocity in body frame
    u, v, w = state[3:6]

    # velocity vector relative to the airmass in body frame
    ur = u - wind_body[0]
    vr = v - wind_body[1]
    wr = w - wind_body[2]

    # compute airspeed, angle of attack, sideslip angle
    Va = np.sqrt(ur**2 + vr**2 + wr**2)
    alpha = np.arctan2(wr, ur)
    beta = np.arcsin(vr / Va)

    return ur, vr, wr, Va, alpha, beta, wind_ned

def calculate_forces(state, Va, delta, alpha, beta):
    delta_a = delta[0]  # aileron
    delta_e = delta[1]  # elevator
    delta_r = delta[2]  # rudder
    delta_t = delta[3]  # throttle

    # extract states (phi, theta, psi, p, q, r)
    p, q, r = state[6:9]
    phi, theta, psi = np.radians(state[9:12])

    # gravitational forces in body frame
    fg_ned = np.array([0, 0, -mass * gravity])
    R = np.array([
        [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
        [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
         np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
         np.sin(phi) * np.cos(theta)],
        [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
         np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
         np.cos(phi) * np.cos(theta)]])
    fg_body = R.T @ fg_ned
    fg_x, fg_y, fg_z = fg_body

    # Compute Lift and Drag coefficients (CL, CD)
    C_L = C_L_0 + C_L_alpha * alpha
    C_D = C_D_0 + C_D_alpha * alpha

    # Compute longitudinal forces in body frame (fx, fz)
    fx_fz = 0.5 * rho * Va**2 * S_wing * np.array([
        (-C_D * np.cos(alpha) + C_L * np.sin(alpha))
        + (-C_D_q * np.cos(alpha) + C_L_q * np.sin(alpha)) * (c / (2 * Va)) * q
        + (-C_D_delta_e * np.cos(alpha) + C_L_delta_e * np.sin(alpha)) * delta_e,
        
        (-C_D * np.sin(alpha) - C_L * np.cos(alpha))
        + (-C_D_q * np.sin(alpha) - C_L_q * np.cos(alpha)) * (c / (2 * Va)) * q
        + (-C_D_delta_e * np.sin(alpha) - C_L_delta_e * np.cos(alpha)) * delta_e])

    f_x = fx_fz[0] 
    f_z = fx_fz[1]

    # Compute lateral forces in body frame (fy)
    f_y = 0.5 * rho * Va**2 * S_wing * (C_Y_0 + C_Y_beta * beta
        + C_Y_p * (b / (2 * Va)) * p
        + C_Y_r * (b / (2 * Va)) * r
        + C_Y_delta_a * delta_a
        + C_Y_delta_r * delta_r)
    
    forces = np.array([f_x + fg_x, f_y + fg_y, f_z + fg_z])
    return forces


def calculate_moments(state, delta, Va, alpha, beta):
    p, q, r = state[6:9]

    delta_a = delta[0]  # aileron
    delta_e = delta[1]  # elevator
    delta_r = delta[2]  # rudder

    # Compute longitudinal torque in body frame (My)
    My = 0.5 * rho * Va**2 * S_wing * c * (
        C_m_0 + C_m_alpha * alpha
        + C_m_q * (c / (2 * Va)) * q
        + C_m_delta_e * delta_e)

    # Compute lateral torques in body frame (Mx, Mz)
    Mx = 0.5 * rho * Va**2 * S_wing * b * (
        C_ell_0 + C_ell_beta * beta
        + C_ell_p * (b / (2 * Va)) * p
        + C_ell_r * (b / (2 * Va)) * r
        + C_ell_delta_a * delta_a
        + C_ell_delta_r * delta_r)
   
    Mz = 0.5 * rho * Va**2 * S_wing * b * (
        C_n_0 + C_n_beta * beta
        + C_n_p * (b / (2 * Va)) * p
        + C_n_r * (b / (2 * Va)) * r
        + C_n_delta_a * delta_a
        + C_n_delta_r * delta_r)

    moments = np.array([Mx, My, Mz])
    return moments

def equations_of_motion(state, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va):
    north, east, down, u, v, w, p, q, r, phi, theta, psi = state
    ur, vr, wr, _, _, _, wind_ned = update_velocity_data(state, wind)
    f_x_total, f_y_total, f_z_total = calculate_forces(state, Va, delta, alpha, beta)
    Mx, My, Mz = calculate_moments(state, delta, Va, alpha, beta)

    
    phi = np.radians(phi)
    theta = np.radians(theta)
    psi = np.radians(psi)

    north_dot = np.cos(theta) * np.cos(psi) * u + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w
    east_dot = np.cos(theta) * np.sin(psi) * u + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w
    down_dot = -np.sin(theta) * u + np.sin(phi) * np.cos(theta) * v + np.cos(phi) * np.cos(theta) * w

    u_dot = r * v - q * w + f_x_total / mass
    v_dot = p * w - r * u + f_y_total / mass
    w_dot = q * u - p * v + f_z_total / mass

    Gamma1 = (Jxz * (Jx - Jy + Jz)) / (Jx * Jz - Jxz**2)
    Gamma2 = (Jz * (Jz - Jy) + Jxz**2) / (Jx * Jz - Jxz**2)
    Gamma3 = Jz / (Jx * Jz - Jxz**2)
    Gamma4 = Jxz / (Jx * Jz - Jxz**2)
    Gamma5 = (Jz - Jx) / Jy
    Gamma6 = Jxz / Jy
    Gamma7 = ((Jx - Jy) * Jx + Jxz**2) / (Jx * Jz - Jxz**2)
    Gamma8 = Jx / (Jx * Jz - Jxz**2)

    p_dot = Gamma1 * p * q - Gamma2 * q * r + Gamma3 * Mx + Gamma4 * Mz
    q_dot = Gamma5 * p * r - Gamma6 * (p**2 - r**2) + My / Jy
    r_dot = Gamma7 * p * q - Gamma1 * q * r + Gamma4 * Mx + Gamma8 * Mz

    phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot])

def rk4_step(state, dt, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va):
    k1 = equations_of_motion(state, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va)
    k2 = equations_of_motion(state + dt/2 * k1, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va)
    k3 = equations_of_motion(state + dt/2 * k2, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va)
    k4 = equations_of_motion(state + dt * k3, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va)

    state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return state_new

# Simulation parameters
dt = 0.01  
t_end = 10  
num_steps = int(t_end / dt)
time = np.linspace(0, t_end, num_steps)

# Simulation
states = np.zeros((num_steps, len(initial_state)))
states[0, :] = initial_state

# Compute initial alpha, beta, and Va
ur, vr, wr, Va, alpha, beta, wind_ned = update_velocity_data(initial_state, wind)

for i in range(1, num_steps):
    # Update state using RK4
    states[i, :] = rk4_step(states[i-1, :], dt, mass, Jx, Jy, Jz, Jxz, delta, wind, alpha, beta, Va)
    
    # Recompute alpha, beta, and Va for the next time step
    ur, vr, wr, Va, alpha, beta, wind_ned = update_velocity_data(states[i, :], wind)


# Extract simulation results
north = states[:, 0]
east = states[:, 1]
down = states[:, 2]
u = states[:, 3]
v = states[:, 4]
w = states[:, 5]
p = states[:, 6]
q = states[:, 7]
r = states[:, 8]
phi = states[:, 9]
theta = states[:, 10]
psi = states[:, 11]


"""
def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = V_max * delta_t

        # Angular speed of propeller (omega_p = ?)
        a = rho * D_prop**5 / ((2 * np.pi)**2) * C_Q0
        b = (rho * D_prop**4 / (2 * np.pi)) * C_Q1 * Va + KQ**2 / R_motor
        c = rho * D_prop**3 * C_Q2 * Va**2 - (KQ * v_in / R_motor) + KQ * i0

        omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # thrust and torque due to propeller
        C_T = C_T0 + C_T1 * (2 * np.pi * omega_p / Va) + C_T2 * (2 * np.pi * omega_p / Va)**2
        thrust_prop = rho * (omega_p / (2 * np.pi))**2 * D_prop**4 * C_T

        C_Q = C_Q0 + C_Q1 * (2 * np.pi * omega_p / Va) + C_Q2 * (2 * np.pi * omega_p / Va)**2
        torque_prop = rho * (omega_p / (2 * np.pi))**2 * D_prop**5 * C_Q

        return thrust_prop, torque_prop
"""


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Position Projection of Aircraft')
ax.legend()

plt.figure(figsize=(10, 6))
plt.plot(time, u, label="u (velocity in x)")
plt.plot(time, v, label="v (velocity in y)")
plt.plot(time, w, label="w (velocity in z)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Translational Velocities Over Time")
plt.grid()

fig, ax = plt.subplots(2, figsize=(11, 8))
ax[0].plot(time, p, label="p")
ax[0].plot(time, q, label="q")
ax[0].plot(time, r, label="r")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angular Velocity (rad/s)")
ax[0].legend()
ax[0].set_title("Simulation Rotational Velocities")
ax[0].grid()

plt.subplots_adjust(hspace=0.4)
plt.show()
