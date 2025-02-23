import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mass = 11  
Jx = 1 
Jy = 1 
Jz = 2  
Jxz = 0  
forces = np.array([0, 0, -mass * 9.81])  # Gravity force
moments = np.array([0, 0, 0])  # No moments applied

def equations_of_motion(state, mass, Jx, Jy, Jz, Jxz, forces, moments):

    north, east, down, u, v, w, p, q, r, phi, theta, psi = state

    fx, fy, fz = forces
    Mx, My, Mz = moments

    phi = np.radians(phi)
    theta = np.radians(theta)
    psi = np.radians(psi)

    north_dot = np.cos(theta) * np.cos(psi) * u + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w
    east_dot = np.cos(theta) * np.sin(psi) * u + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w
    down_dot = -np.sin(theta) * u + np.sin(phi) * np.cos(theta) * v + np.cos(phi) * np.cos(theta) * w

    u_dot = r * v - q * w + fx / mass
    v_dot = p * w - r * u + fy / mass
    w_dot = q * u - p * v + fz / mass

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

def analytic_equations_of_motion(state, J1, J2, J3):

    p,q,r = state

    p_dot = (J1-J3)*q*r/J1
    q_dot = (J3-J1)*p*r/J1
    r_dot = 0

    return np.array([p_dot, q_dot, r_dot])

# solving EOMs using RK4 method
def rk4_step(state, dt, mass, Jx, Jy, Jz, Jxz, forces, moments):
    k1 = equations_of_motion(state, mass, Jx, Jy, Jz, Jxz, forces, moments)
    k2 = equations_of_motion(state + dt/2 * k1, mass, Jx, Jy, Jz, Jxz, forces, moments)
    k3 = equations_of_motion(state + dt/2 * k2, mass, Jx, Jy, Jz, Jxz, forces, moments)
    k4 = equations_of_motion(state + dt * k3, mass, Jx, Jy, Jz, Jxz, forces, moments)

    state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return state_new

def rk4_step_analytic(state, dt, J1, J2, J3):
    k1 = analytic_equations_of_motion(state, J1, J2, J3)
    k2 = analytic_equations_of_motion(state + dt/2 * k1, J1, J2, J3)
    k3 = analytic_equations_of_motion(state + dt/2 * k2, J1, J2, J3)
    k4 = analytic_equations_of_motion(state + dt * k3, J1, J2, J3)

    state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return state_new

initial_state = np.array([0, 0, 10, 0, 0, 0, 1, 0, 2, 10, 2, 2])  # [north, east, down, u, v, w, p, q, r, phi, theta, psi]
initial_state2 = np.array([1, 0, 2])  # [p, q, r]

dt = 0.01  
t_end = 10  
num_steps = int(t_end / dt)
time = np.linspace(0, t_end, num_steps)

# simulation
states = np.zeros((num_steps, len(initial_state)))
states[0, :] = initial_state

for i in range(1, num_steps):
    states[i, :] = rk4_step(states[i-1, :], dt, mass, Jx, Jy, Jz, Jxz, forces, moments)

# extract simulation results
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

# analytical solution
states_analytic = np.zeros((num_steps, len(initial_state2)))
states_analytic[0, :] = initial_state2

for i in range(1, num_steps):
    states_analytic[i, :] = rk4_step_analytic(states_analytic[i-1, :], dt, Jx, Jy, Jz)

# extract analytical results
p_ana = states_analytic[:, 0]
q_ana = states_analytic[:, 1]
r_ana = states_analytic[:, 2]

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

ax[1].plot(time, p_ana, label="p")
ax[1].plot(time, q_ana, label="q")
ax[1].plot(time, r_ana, label="r")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Angular Velocity (rad/s)")
ax[1].legend()
ax[1].set_title("Analytical Rotational Velocities")
ax[1].grid()

plt.subplots_adjust(hspace=0.4)
plt.show()
