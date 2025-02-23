import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from integrators import RungeKutta

def aerodynamic_forces_and_moments(state, aerodynamic_params, rho, S, c, b, delta_e, delta_a, delta_r):

    (CL0, CL_alpha, CD0, CD_alpha, CLq, CDq, CL_delta_e, CD_delta_e, 
     CY0, CY_beta, CY_p, CY_r, CY_delta_a, CY_delta_r, 
     Cm0, Cm_alpha, Cm_q, Cm_delta_e, 
     Cl0, Cl_beta, Cl_p, Cl_r, Cl_delta_a, Cl_delta_r, 
     Cn0, Cn_beta, Cn_p, Cn_r, Cn_delta_a, Cn_delta_r) = aerodynamic_params
    
    _, _, _, u, v, w, p, q, r, phi, theta, psi = state

    Va = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)  
    beta = np.arcsin(v / Va) if Va != 0 else 0  

    CL = CL0 + CL_alpha * alpha
    CD = CD0 + CD_alpha * alpha

    fx = -0.5 * rho * Va**2 * S * (CD * np.cos(alpha) - CL * np.sin(alpha) 
        + (CDq * np.cos(alpha) - CLq * np.sin(alpha)) * (c / (2 * Va)) * q
        + (CD_delta_e * np.cos(alpha) - CL_delta_e * np.sin(alpha)) * delta_e)

    fz = -0.5 * rho * Va**2 * S * (CD * np.sin(alpha) + CL * np.cos(alpha) 
        + (CDq * np.sin(alpha) + CLq * np.cos(alpha)) * (c / (2 * Va)) * q
        + (CD_delta_e * np.sin(alpha) + CL_delta_e * np.cos(alpha)) * delta_e)

    fy = 0.5 * rho * Va**2 * S * (CY0 + CY_beta * beta 
        + CY_p * (b / (2 * Va)) * p 
        + CY_r * (b / (2 * Va)) * r 
        + CY_delta_a * delta_a + CY_delta_r * delta_r)

    Mx = 0.5 * rho * Va**2 * S * b * (Cl0 + Cl_beta * beta 
        + Cl_p * (b / (2 * Va)) * p 
        + Cl_r * (b / (2 * Va)) * r 
        + Cl_delta_a * delta_a + Cl_delta_r * delta_r)

    My = 0.5 * rho * Va**2 * S * c * (Cm0 + Cm_alpha * alpha
        + Cm_q * (c / (2 * Va)) * q
        + Cm_delta_e * delta_e)

    Mz = 0.5 * rho * Va**2 * S * b * (Cn0 + Cn_beta * beta 
        + Cn_p * (b / (2 * Va)) * p 
        + Cn_r * (b / (2 * Va)) * r 
        + Cn_delta_a * delta_a + Cn_delta_r * delta_r)

    return np.array([fx, fy, fz]), np.array([Mx, My, Mz])

def equations_of_motion(t, state, mass, Jx, Jy, Jz, Jxz, rho, S, c, b, aerodynamic_params, control_inputs):
    north, east, down, u, v, w, p, q, r, phi, theta, psi = state
    
    delta_e, delta_a, delta_r = control_inputs
    
    forces, moments = aerodynamic_forces_and_moments(state, aerodynamic_params, rho, S, c, b, delta_e, delta_a, delta_r)
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

    p_dot = (Mx - (Jz - Jy) * q * r) / Jx
    q_dot = (My - (Jx - Jz) * p * r) / Jy
    r_dot = (Mz - (Jy - Jx) * p * q) / Jz

    phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot])


dt = 0.01  
T = 10  
N = int(T / dt)

mass = 11  
Jx, Jy, Jz, Jxz = 1, 1, 2, 0  
rho = 1.225  
S, c, b = 0.5, 0.2, 1.0  

aerodynamic_params = (0.23, 5.61, 0.0424, 0.132, 7.95, 0.0, 0.13, 0.0135,  
                      0.0, -0.98, 0.0, 0.0, 0.075, 0.19,  
                      0.0135, -2.74, -38.21, -0.99,  
                      0.0, -0.13, -0.51, 0.25, 0.17, 0.0024,  
                      0.0, 0.073, 0.069, -0.095, -0.011, -0.069)

# north, east, down, u, v, w, p, q, r, phi, theta, psi
initial_state = np.array([0, 0, -10, 10, 0, 0, 0, 0, 0, 0, 5, 0])  

forces_no_wind = np.array([0, 0, 0])
moments_no_wind = np.array([0, 0, 0])

def equations_of_motion_no_wind(t, state, _): 
    north, east, down, u, v, w, p, q, r, phi, theta, psi = state

    fx, fy, fz = forces_no_wind  # Gravity-only forces
    Mx, My, Mz = moments_no_wind  # No aerodynamic moments

    # Convert angles to radians
    phi = np.radians(phi)
    theta = np.radians(theta)
    psi = np.radians(psi)

    # Translational Motion
    north_dot = np.cos(theta) * np.cos(psi) * u + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w
    east_dot = np.cos(theta) * np.sin(psi) * u + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w
    down_dot = -np.sin(theta) * u + np.sin(phi) * np.cos(theta) * v + np.cos(phi) * np.cos(theta) * w

    u_dot = r * v - q * w + fx / mass
    v_dot = p * w - r * u + fy / mass
    w_dot = q * u - p * v + fz / mass

    # Rotational Motion Constants
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


def equations_of_motion_with_wind(t, state, _):
    return equations_of_motion(t, state, mass, Jx, Jy, Jz, Jxz, rho, S, c, b, aerodynamic_params, [0, 0, 0])

integrator_no_wind = RungeKutta(dt, equations_of_motion_no_wind)
integrator_wind = RungeKutta(dt, equations_of_motion_with_wind)

states_no_wind = np.zeros((N, len(initial_state)))
states_no_wind[0, :] = initial_state

states_wind = np.zeros((N, len(initial_state)))
states_wind[0, :] = initial_state

t = 0
for i in range(1, N):
    states_no_wind[i, :] = integrator_no_wind.step(t, states_no_wind[i-1, :], None)
    states_wind[i, :] = integrator_wind.step(t, states_wind[i-1, :], None)
    t += dt

time = np.linspace(0, T, N)

# No Wind Case
north1, east1, down1 = states_no_wind[:, 0], states_no_wind[:, 1], states_no_wind[:, 2]
u1, v1, w1 = states_no_wind[:, 3], states_no_wind[:, 4], states_no_wind[:, 5]
p1, q1, r1 = states_no_wind[:, 6], states_no_wind[:, 7], states_no_wind[:, 8]

# Wind Case
north2, east2, down2 = states_wind[:, 0], states_wind[:, 1], states_wind[:, 2]
u2, v2, w2 = states_wind[:, 3], states_wind[:, 4], states_wind[:, 5]
p2, q2, r2 = states_wind[:, 6], states_wind[:, 7], states_wind[:, 8]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

axes[0].plot(north1, east1, down1, label='Without Wind', color='b')
axes[0].set_xlabel('North (m)')
axes[0].set_ylabel('East (m)')
axes[0].set_zlabel('Down (m)')
axes[0].set_title("Without Wind")
axes[0].legend()

axes[1].plot(north2, east2, down2, label='With Wind', color='r')
axes[1].set_xlabel('North (m)')
axes[1].set_ylabel('East (m)')
axes[1].set_zlabel('Down (m)')
axes[1].set_title("With Wind")
axes[1].legend()

plt.suptitle("Comparison of Aircraft Motion With and Without Wind", fontsize=16)

plt.tight_layout()
plt.show()

### **Figure 2: Translational Velocities Comparison**
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(time, u1, label="u (x-velocity)")
axes[0].plot(time, v1, label="v (y-velocity)")
axes[0].plot(time, w1, label="w (z-velocity)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Velocity (m/s)")
axes[0].legend()
axes[0].set_title("Without Wind")
axes[0].grid()

axes[1].plot(time, u2, label="u (x-velocity)")
axes[1].plot(time, v2, label="v (y-velocity)")
axes[1].plot(time, w2, label="w (z-velocity)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].legend()
axes[1].set_title("With Wind")
axes[1].grid()

plt.suptitle(r"Translational Velocities Comparison (u=10, $\theta$ = 5°)", fontsize=16)

plt.tight_layout()
plt.show()

### **Figure 3: Rotational Velocities Comparison**
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(time, p1, label="p (roll rate)")
axes[0].plot(time, q1, label="q (pitch rate)")
axes[0].plot(time, r1, label="r (yaw rate)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Angular Velocity (rad/s)")
axes[0].legend()
axes[0].set_title("Without Wind")
axes[0].grid()

axes[1].plot(time, p2, label="p (roll rate)")
axes[1].plot(time, q2, label="q (pitch rate)")
axes[1].plot(time, r2, label="r (yaw rate)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].legend()
axes[1].set_title("With Wind")
axes[1].grid()

plt.suptitle(r"Rotational Velocities Comparison (u=10, $\theta$ = 5°)", fontsize=16)

plt.tight_layout()
plt.show()
