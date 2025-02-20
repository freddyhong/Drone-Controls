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

<<<<<<< HEAD
=======
# C values
CL0 = 0.23
CD0 = 0.043
Cm0 = 0.0135
CLa= 5.61
CDa = 0.03
Cma = -2.74
CLq = 7.95
CDq = 0
Cmq = -38.21
CLde = 0.13
CDde = 0.0135
Cmde = -0.99
M = 50
alpha0 = 0.47
CDp = 0.043
CY0 = 0
Cl0 = 0
Cn0 = 0
CYb = -0.83
Clb = -0.13
Cnb = 0.073
CYp = 0
Clp = -0.51
Cnp = -0.069
CYr = 0
Clr = 0.25
Cnr = -0.095
CYda = 0.075
Clda = 0.17
Cnda = -0.011
CYdr = 0.19
Cldr = 0.0024
Cndr = -0.069
>>>>>>> e5b14df850d966ec33b96cdb62d53992f248ce02




<<<<<<< HEAD
def aerodynamic_forces(rho, Va, S, alpha, CD, CL, CD_q, CL_q, CD_delta_e, CL_delta_e, c, q, delta_e):
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    F_drag_lift = np.array([
        -0.5 * rho * Va**2 * S * (CD * np.cos(alpha) - CL * np.sin(alpha) 
                                  + (CD_q * np.cos(alpha) - CL_q * np.sin(alpha)) * (c / (2 * Va)) * q
                                  + (CD_delta_e * np.cos(alpha) - CL_delta_e * np.sin(alpha)) * delta_e),
        
        -0.5 * rho * Va**2 * S * (CD * np.sin(alpha) + CL * np.cos(alpha) 
                                  + (CD_q * np.sin(alpha) + CL_q * np.cos(alpha)) * (c / (2 * Va)) * q
                                  + (CD_delta_e * np.sin(alpha) + CL_delta_e * np.cos(alpha)) * delta_e)
    ])
    
    f_xz = R @ F_drag_lift
    
    return f_xz

=======
>>>>>>> e5b14df850d966ec33b96cdb62d53992f248ce02

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
