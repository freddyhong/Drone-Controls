import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import parameters as par

class Wind:
    def __init__(self):
        self.steady_wind = np.array(par.steady_wind)  # Steady wind in NED frame
        self.gust_amplitude = np.array(par.gust_amplitude)  # Gust amplitude
    
    def generate_wind_gust(self):
        # Generate random gust in body frame
        return np.random.uniform(-1, 1, size=3) * self.gust_amplitude
    
    def get_wind(self):
        # Return total wind (steady + gust) in NED frame
        return np.array(self.steady_wind)

class MAV:
    def __init__(self, ts, mass, Jx, Jy, Jz, Jxz, initial_state):
        self.ts = ts
        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxz = Jxz
        self.state = np.array(initial_state)  # MAV state vector
        self.Va = 0  # Airspeed
        self.alpha = 0.0  # Angle of attack
        self.beta = 0.0  # Sideslip angle

    def update(self, delta):
        def f(state, forces, moments):
            return self.equations_of_motion(state, forces, moments)

        forces, moments = self.compute_forces_moments(delta)

        # RK4 Integration
        k1 = f(self.state, forces, moments)
        k2 = f(self.state + 0.5 * self.ts * k1, forces, moments)
        k3 = f(self.state + 0.5 * self.ts * k2, forces, moments)
        k4 = f(self.state + self.ts * k3, forces, moments)

        self.state = self.state + (self.ts / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self._update_velocity_data(delta)

    def _update_velocity_data(self, delta):
        self.compute_forces_moments(delta)

    def equations_of_motion(self, state, forces, moments):
        u, v, w = state[3], state[4], state[5]  # Velocity in body frame
        phi, theta, psi = state[6], state[7], state[8]  # Euler angles
        p, q, r = state[9], state[10], state[11]  # Angular rates
        fx, fy, fz = forces  # Forces in body frame
        Mx, My, Mz = moments  # Moments in body frame

        # Position derivatives (NED frame)
        north_dot = u * np.cos(theta) * np.cos(psi) + v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        east_dot = u * np.cos(theta) * np.sin(psi) + v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
        down_dot = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(phi) * np.cos(theta)

        # Velocity derivatives (body frame)
        u_dot = r * v - q * w + fx / self.mass
        v_dot = p * w - r * u + fy / self.mass
        w_dot = q * u - p * v + fz / self.mass

        Gamma1 = (self.Jxz * (self.Jx - self.Jy + self.Jz)) / (self.Jx * self.Jz - self.Jxz**2)
        Gamma2 = (self.Jz * (self.Jz - self.Jy) + self.Jxz**2) / (self.Jx * self.Jz - self.Jxz**2)
        Gamma3 = self.Jz / (self.Jx * self.Jz - self.Jxz**2)
        Gamma4 = self.Jxz / (self.Jx * self.Jz - self.Jxz**2)
        Gamma5 = (self.Jz - self.Jx) / self.Jy
        Gamma6 = self.Jxz / self.Jy
        Gamma7 = ((self.Jx - self.Jy) * self.Jx + self.Jxz**2) / (self.Jx * self.Jz - self.Jxz**2)
        Gamma8 = self.Jx / (self.Jx * self.Jz - self.Jxz**2)

        # Angular rate derivatives (body frame)
        p_dot = Gamma1 * p * q - Gamma2 * q * r + Gamma3 * Mx + Gamma4 * Mz
        q_dot = Gamma5 * p * r - Gamma6 * (p**2 - r**2) + My / self.Jy
        r_dot = Gamma7 * p * q - Gamma1 * q * r + Gamma4 * Mx + Gamma8 * Mz

        # Euler angle derivatives
        phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])

    def rotation_matrix(self, phi, theta, psi):
        # Rotation matrix from body to NED frame
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
             np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
             np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
             np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
             np.cos(phi) * np.cos(theta)]
        ])
        return R

class MAV_wind(MAV):
    def __init__(self, ts, mass, Jx, Jy, Jz, Jxz, initial_state):
        super().__init__(ts, mass, Jx, Jy, Jz, Jxz, initial_state)
        self.wind = Wind()  # Initialize wind object

    def compute_forces_moments(self, delta):
        wind_ned = self.wind.get_wind()  # Get wind in NED frame

        # Transform wind to body frame
        phi, theta, psi = self.state[6], self.state[7], self.state[8]
        R = self.rotation_matrix(phi, theta, psi)
        wind_body = (R.T @ wind_ned[:3]).flatten()  # Ensure it's a 1D array

        # Velocity vector relative to the airmass in body frame
        u, v, w = self.state[3], self.state[4], self.state[5]
        ur = u - wind_body[0]
        vr = v - wind_body[1]
        wr = w - wind_body[2]

        # Compute airspeed, angle of attack, and sideslip angle
        self.Va = 25 #np.sqrt(ur**2 + vr**2 + wr**2)
        self.alpha = np.arctan2(wr, ur)
        self.beta = np.arcsin(vr / self.Va)

        # Compute forces and moments
        fx = -0.5 * par.rho * self.Va**2 * par.S_wing * (
            (par.C_D_0 + par.C_D_alpha * self.alpha) * np.cos(self.alpha) 
            - (par.C_L_0 + par.C_L_alpha * self.alpha) * np.sin(self.alpha)
            + (par.C_D_q * np.cos(self.alpha) - par.C_L_q * np.sin(self.alpha)) * (par.c / (2 * self.Va)) * self.state[10]
            + (par.C_D_delta_e * np.cos(self.alpha) - par.C_L_delta_e * np.sin(self.alpha)) * delta.elevator
            )

        fz = -0.5 * par.rho * self.Va**2 * par.S_wing * (
            (par.C_D_0 + par.C_D_alpha * self.alpha) * np.sin(self.alpha) 
            + (par.C_L_0 + par.C_L_alpha * self.alpha) * np.cos(self.alpha)
            + (par.C_D_q * np.sin(self.alpha) + par.C_L_q * np.cos(self.alpha)) * (par.c / (2 * self.Va)) * self.state[10]
            + (par.C_D_delta_e * np.sin(self.alpha) + par.C_L_delta_e * np.cos(self.alpha)) * delta.elevator
            )

        fy = 0.5 * par.rho * self.Va**2 * par.S_wing * (
            par.C_Y_0 + par.C_Y_beta * self.beta 
            + par.C_Y_p * (par.b / (2 * self.Va)) * self.state[9] 
            + par.C_Y_r * (par.b / (2 * self.Va)) * self.state[11] 
            + par.C_Y_delta_a * delta.aileron 
            + par.C_Y_delta_r * delta.rudder
            )

        # Add gravitational forces
        fx += self.mass * par.gravitational_acceleration * np.sin(theta)
        fy -= self.mass * par.gravitational_acceleration * np.cos(theta) * np.sin(phi)
        fz -= self.mass * par.gravitational_acceleration * np.cos(theta) * np.cos(phi)

        # Compute moments
        q_bar = 0.5 * par.rho * self.Va**2 * par.S_wing  # Dynamic pressure

        Mx = q_bar * par.S_wing * par.b * (
            par.C_ell_0 + par.C_ell_beta * self.beta +
            par.C_ell_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_ell_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_ell_delta_a * delta.aileron +
            par.C_ell_delta_r * delta.rudder
        )
        
        Mz = q_bar * par.S_wing * par.b * (
            par.C_n_0 + par.C_n_beta * self.beta +
            par.C_n_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_n_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_n_delta_a * delta.aileron +
            par.C_n_delta_r * delta.rudder
        )

        My = q_bar * par.S_wing * par.c * (
            par.C_m_0 + par.C_m_alpha * self.alpha +
            par.C_m_q * (par.c / (2 * self.Va)) * self.state[10] +
            par.C_m_delta_e * delta.elevator
        )

        return np.array([fx, fy, fz]), np.array([Mx, My, Mz])

class MAV_nowind(MAV):
    def compute_forces_moments(self, delta):
        # Velocity vector relative to the airmass in body frame (no wind)
        u, v, w = self.state[3], self.state[4], self.state[5]
        ur, vr, wr = u, v, w  # No wind, so relative velocity = absolute velocity

        # Compute airspeed, angle of attack, and sideslip angle
        self.Va = 25 #np.sqrt(ur**2 + vr**2 + wr**2)
        self.alpha = np.arctan2(wr, ur)
        self.beta = np.arcsin(vr / self.Va)

        # Compute forces and moments
        fx = -0.5 * par.rho * self.Va**2 * par.S_wing * (
            (par.C_D_0 + par.C_D_alpha * self.alpha) * np.cos(self.alpha) 
            - (par.C_L_0 + par.C_L_alpha * self.alpha) * np.sin(self.alpha)
            + (par.C_D_q * np.cos(self.alpha) - par.C_L_q * np.sin(self.alpha)) * (par.c / (2 * self.Va)) * self.state[10]
            + (par.C_D_delta_e * np.cos(self.alpha) - par.C_L_delta_e * np.sin(self.alpha)) * delta.elevator
            )

        fz = -0.5 * par.rho * self.Va**2 * par.S_wing * (
            (par.C_D_0 + par.C_D_alpha * self.alpha) * np.sin(self.alpha) 
            + (par.C_L_0 + par.C_L_alpha * self.alpha) * np.cos(self.alpha)
            + (par.C_D_q * np.sin(self.alpha) + par.C_L_q * np.cos(self.alpha)) * (par.c / (2 * self.Va)) * self.state[10]
            + (par.C_D_delta_e * np.sin(self.alpha) + par.C_L_delta_e * np.cos(self.alpha)) * delta.elevator
            )

        fy = 0.5 * par.rho * self.Va**2 * par.S_wing * (
            par.C_Y_0 + par.C_Y_beta * self.beta 
            + par.C_Y_p * (par.b / (2 * self.Va)) * self.state[9] 
            + par.C_Y_r * (par.b / (2 * self.Va)) * self.state[11] 
            + par.C_Y_delta_a * delta.aileron 
            + par.C_Y_delta_r * delta.rudder
            )

        # Add gravitational forces
        phi, theta, psi = self.state[6], self.state[7], self.state[8]
        R = self.rotation_matrix(phi, theta, psi)
        fg_ned = np.array([0, 0, self.mass * par.gravitational_acceleration])
        fg_body = (R.T @ fg_ned).flatten()
        fx += fg_body[0]
        fy += fg_body[1]
        fz += fg_body[2]

        # Compute moments
        q_bar = 0.5 * par.rho * self.Va**2 * par.S_wing  # Dynamic pressure

        Mx = q_bar * par.S_wing * par.b * (
            par.C_ell_0 + par.C_ell_beta * self.beta +
            par.C_ell_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_ell_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_ell_delta_a * delta.aileron +
            par.C_ell_delta_r * delta.rudder
        )
        
        Mz = q_bar * par.S_wing * par.b * (
            par.C_n_0 + par.C_n_beta * self.beta +
            par.C_n_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_n_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_n_delta_a * delta.aileron +
            par.C_n_delta_r * delta.rudder
        )

        My = q_bar * par.S_wing * par.c * (
            par.C_m_0 + par.C_m_alpha * self.alpha +
            par.C_m_q * (par.c / (2 * self.Va)) * self.state[10] +
            par.C_m_delta_e * delta.elevator
        )

        return np.array([fx, fy, fz]), np.array([Mx, My, Mz])