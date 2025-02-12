import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def equations_of_motion(t, state, mass, Jx, Jy, Jz, Jxz, forces, moments):
    
    u, v, w, p, q, r = state

    fx, fy, fz = forces
    Mx, My, Mz = moments
 
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

    return np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot])


initial_state = np.array([0, 0, 0, 0, 0, 0])  

mass = 11  #kg*m^2
Jx = 0.824  #kg*m^2
Jy = 1.135  #kg*m^2
Jz = 1.759  #kg*m^2
Jxz = 0.12  #kg*m^2
forces = np.array([10, 0, -9.81 * mass])  # These are example forces
moments = np.array([0, 0, 5])  # These are example moments


t_span = (0, 10)  
t_eval = np.linspace(0, 10, 100)  

sol = solve_ivp(equations_of_motion, t_span, initial_state, t_eval=t_eval, args=(mass, Jx, Jy, Jz, Jxz, forces, moments))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="u (velocity in x)")
plt.plot(sol.t, sol.y[1], label="v (velocity in y)")
plt.plot(sol.t, sol.y[2], label="w (velocity in z)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Translational Velocities Over Time")
plt.grid()

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[3], label="p (angular velocity in x)")
plt.plot(sol.t, sol.y[4], label="q (angular velocity in y)")
plt.plot(sol.t, sol.y[5], label="r (angular velocity in z)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.title("Rotational Velocities Over Time")
plt.grid()
plt.show()
