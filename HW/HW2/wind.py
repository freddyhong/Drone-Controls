import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def equations_of_motion(t, state, mass, Jx, Jy, Jz, Jxz, forces, moments):
    
    north, east, down, u, v, w, p, q, r, phi, theta, pi = state

    fx, fy, fz = forces
    Mx, My, Mz = moments
    
    phi = np.radians(phi)
    theta = np.radians(theta)
    pi = np.radians(pi)

    north_dot = np.cos(theta) * np.cos(pi) * u + (np.sin(phi) * np.sin(theta) * np.cos(pi) - np.cos(phi) * np.sin(pi)) * v + (np.cos(phi) * np.sin(theta) * np.cos(pi) + np.sin(phi) * np.sin(pi)) * w
    east_dot = np.cos(theta) * np.sin(pi) * u + (np.sin(phi) * np.sin(theta) * np.sin(pi) + np.cos(phi) * np.cos(pi)) * v + (np.cos(phi) * np.sin(theta) * np.sin(pi) - np.sin(phi) * np.cos(pi)) * w 
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
    pi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)
    
    return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, pi_dot])


def analytic_equations_of_motion(t, state, J1, J2, J3):

    p,q,r = state
    p_dot = (J1-J3)*q*r/J1
    q_dot = (J3-J1)*p*r/J1
    r_dot = 0
    
    return np.array([p_dot, q_dot, r_dot])


initial_state = np.array([0, 0, 10, 0, 0, 0, 1, 0, 2, 10, 2, 2])  
initial_state2 = np.array([1,0,2])

mass = 11  #kg*m^2
Jx = 1  #kg*m^2
Jy = 1  #kg*m^2
Jz = 2  #kg*m^2
Jxz = 0  #kg*m^2
forces = np.array([0, 0, -mass*9.81])  # These are example forces
moments = np.array([0, 0, 0])  # These are example moments

C_L_0 = 0.23
C_D_0 = 0.0424
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.132
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99
M = 50.0
alpha0 = 0.47
epsilon = 0.16
C_D_p = 0.043


C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_ell_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_ell_p = -0.51
C_n_p = 0.069
C_Y_r = 0.0
C_ell_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_ell_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_ell_delta_r = 0.0024
C_n_delta_r = -0.069


def aerodynamic_forces(rho, Va, S, c, b,
                        C_L_0, C_D_0, 
                        C_L_alpha, C_D_alpha,
                        C_L_q, C_D_q,
                        C_L_delta_e, C_D_delta_e,
                        C_Y_0, C_Y_beta, C_Y_p, C_Y_r, C_Y_delta_a, C_Y_delta_r,
                        alpha, beta, p, q, r, delta_e, delta_a, delta_r):


    C_L = C_L_0 + C_L_alpha * alpha + C_L_q * (c / (2 * Va)) * q + C_L_delta_e * delta_e
    C_D = C_D_0 + C_D_alpha * alpha + C_D_q * (c / (2 * Va)) * q + C_D_delta_e * delta_e

    fx_fz = 0.5 * rho * Va**2 * S * np.array([
        (-C_D * np.cos(alpha) + C_L * np.sin(alpha)) 
        + (-C_D_q * np.cos(alpha) + C_L_q * np.sin(alpha)) * (c / (2 * Va)) * q
        + (-C_D_delta_e * np.cos(alpha) + C_L_delta_e * np.sin(alpha)) * delta_e,
        
        (-C_D * np.sin(alpha) - C_L * np.cos(alpha)) 
        + (-C_D_q * np.sin(alpha) - C_L_q * np.cos(alpha)) * (c / (2 * Va)) * q
        + (-C_D_delta_e * np.sin(alpha) - C_L_delta_e * np.cos(alpha)) * delta_e
    ])

    f_x, f_z = fx_fz

    f_y = 0.5 * rho * Va**2 * S * (
        C_Y_0 + C_Y_beta * beta
        + C_Y_p * (b / (2 * Va)) * p
        + C_Y_r * (b / (2 * Va)) * r
        + C_Y_delta_a * delta_a
        + C_Y_delta_r * delta_r
    )

    return f_x, f_y, f_z

f_x, f_y, f_z = aerodynamic_forces(rho, Va, S, c, b, 
                                   C_L_0, C_D_0,
                                   C_L_alpha, C_D_alpha,
                                   C_L_q, C_D_q,
                                   C_L_delta_e, C_D_delta_e,
                                   C_Y_0, C_Y_beta, C_Y_p, C_Y_r, C_Y_delta_a, C_Y_delta_r,
                                   alpha, beta, p, q, r, delta_e, delta_a, delta_r)




def aerodynamic_moments(rho, Va, S, c, b, 
                        C_m_0, C_m_alpha, C_m_q, C_m_delta_e, 
                        C_ell_0, C_ell_beta, C_ell_p, C_ell_r, C_ell_delta_a, C_ell_delta_r, 
                        C_n_0, C_n_beta, C_n_p, C_n_r, C_n_delta_a, C_n_delta_r, 
                        alpha, beta, p, q, r, delta_e, delta_a, delta_r):
    
    l = 0.5 * rho * Va**2 * S * b * (
        C_ell_0 + C_ell_beta * beta 
        + C_ell_p * (b / (2 * Va)) * p 
        + C_ell_r * (b / (2 * Va)) * r 
        + C_ell_delta_a * delta_a 
        + C_ell_delta_r * delta_r
    )

    m = 0.5 * rho * Va**2 * S * c * (
        C_m_0 + C_m_alpha * alpha
        + C_m_q * (c / (2 * Va)) * q
        + C_m_delta_e * delta_e
    )
 
    n = 0.5 * rho * Va**2 * S * b * (
        C_n_0 + C_n_beta * beta 
        + C_n_p * (b / (2 * Va)) * p 
        + C_n_r * (b / (2 * Va)) * r 
        + C_n_delta_a * delta_a 
        + C_n_delta_r * delta_r
    )

    return l, m, n

t_span = (0, 10)  
t_eval = np.linspace(0, 10, 100)  

sim_sol = solve_ivp(equations_of_motion, t_span, initial_state, t_eval=t_eval, args=(mass, Jx, Jy, Jz, Jxz, forces, moments))
ana_sol = solve_ivp(analytic_equations_of_motion, t_span, initial_state2, t_eval=t_eval, args=(Jx, Jy, Jz))

north,east,down,u,v,w,p,q,r,phi,theta,pi = sim_sol.y
p_ana, q_ana, r_ana = ana_sol.y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Position Projection of Aircraft')
ax.legend()

plt.figure(figsize=(10, 6))
plt.plot(sim_sol.t, u, label="u (velocity in x)")
plt.plot(sim_sol.t, v, label="v (velocity in y)")
plt.plot(sim_sol.t, w, label="w (velocity in z)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Translational Velocities Over Time")
plt.grid()

fig,ax = plt.subplots(2, figsize=(11, 8))
ax[0].plot(sim_sol.t, p, label="p")
ax[0].plot(sim_sol.t, q, label="q")
ax[0].plot(sim_sol.t, r, label="r")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angular Velocity (rad/s)")
ax[0].legend()
plt.grid()

ax[0].set_title("Simiulation Rotational Velocities")
ax[1].plot(ana_sol.t, p_ana, label="p")
ax[1].plot(ana_sol.t, q_ana, label="q")
ax[1].plot(ana_sol.t, r_ana, label="r")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Angular Velocity (rad/s)")
ax[1].legend()
ax[1].set_title("Analytical Rotational Velocities")
plt.grid()
plt.subplots_adjust(hspace=0.4) 

plt.show()
