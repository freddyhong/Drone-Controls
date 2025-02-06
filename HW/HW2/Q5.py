import numpy as np

#Pysical Param
mass = 11 #kg
Jx = 0.824 #kgm^2
Jy = 1.135 #kgm^2
Jz = 1.759 #kgm^2
Jxz = 0.12 #kgm^2


def equations_of_motion(t, state, mass, Jx, Jy, Jz, Jxz, forces, moments):
    # Unpack the state vector
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    
    # Forces and moments
    fx, fy, fz = forces
    l, m, n = moments
    
    # Precompute trigonometric functions
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    ttheta = np.tan(theta)
    
    # Kinematic equations
    p_dot = np.array([
        [ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
        [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
        [-stheta, sphi * ctheta, cphi * ctheta]
    ]) @ np.array([u, v, w])
    
    # Dynamic equations
    u_dot = r * v - q * w + fx / mass
    v_dot = p * w - r * u + fy / mass
    w_dot = q * u - p * v + fz / mass
    
    # Angular rates
    phi_dot = p + sphi * ttheta * q + cphi * ttheta * r
    theta_dot = cphi * q - sphi * r
    psi_dot = sphi / ctheta * q + cphi / ctheta * r
    
    # Moments of inertia
    Gamma1 = (Jxz * (Jx - Jy + Jz)) / (Jx * Jz - Jxz**2)
    Gamma2 = (Jz * (Jz - Jy) + Jxz**2) / (Jx * Jz - Jxz**2)
    Gamma3 = Jz / (Jx * Jz - Jxz**2)
    Gamma4 = Jxz / (Jx * Jz - Jxz**2)
    Gamma5 = (Jz - Jx) / Jy
    Gamma6 = Jxz / Jy
    Gamma7 = ((Jx - Jy) * Jx + Jxz**2) / (Jx * Jz - Jxz**2)
    Gamma8 = Jx / (Jx * Jz - Jxz**2)
    
    # Angular accelerations
    p_dot_ang = Gamma1 * p * q - Gamma2 * q * r + Gamma3 * l + Gamma4 * n
    q_dot_ang = Gamma5 * p * r - Gamma6 * (p**2 - r**2) + m / Jy
    r_dot_ang = Gamma7 * p * q - Gamma1 * q * r + Gamma4 * l + Gamma8 * n
    
    # Combine derivatives into a single array
    state_dot = np.array([
        p_dot[0], p_dot[1], p_dot[2],
        u_dot, v_dot, w_dot,
        phi_dot, theta_dot, psi_dot,
        p_dot_ang, q_dot_ang, r_dot_ang
    ])
    
    return state_dot

# Example usage
initial_state = np.zeros(12)  # Replace with actual initial conditions
mass = 1.0  # Replace with actual mass
Jx, Jy, Jz, Jxz = 0.1, 0.1, 0.1, 0.01  # Replace with actual moments and products of inertia
forces = np.array([0, 0, -9.81 * mass])  # Replace with actual forces
moments = np.array([0, 0, 0])  # Replace with actual moments

# Time span for simulation
t = 0  # Replace with actual time

# Compute the state derivatives
state_dot = equations_of_motion(t, initial_state, mass, Jx, Jy, Jz, Jxz, forces, moments)
print(state_dot)