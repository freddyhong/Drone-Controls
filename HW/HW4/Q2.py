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
        return self.steady_wind + self.generate_wind_gust()

class MAV:
    def __init__(self, ts, mass, Jx, Jy, Jz, Jxz, initial_state):
        self.ts = ts
        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxz = Jxz
        self.state = np.array(initial_state)  # MAV state vector
        self.wind = Wind()  # Wind model
        self.Va = 0.0  # Airspeed
        self.alpha = 0.0  # Angle of attack
        self.beta = 0.0  # Sideslip angle

    def update(self, delta):
        # Compute forces and moments
        forces, moments = self.compute_forces_moments(delta)
        
        # Update state using equations of motion
        self.state = self.state + self.ts * self.equations_of_motion(self.state, forces, moments)
    
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

        # Angular rate derivatives (body frame)
        p_dot = (Mx - (par.Jz - par.Jy) * q * r) / par.Jx
        q_dot = (My - (par.Jx - par.Jz) * p * r) / par.Jy
        r_dot = (Mz - (par.Jy - par.Jx) * p * q) / par.Jz

        # Euler angle derivatives
        phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
    
    def compute_forces_moments(self, delta):
        # Get wind in NED frame
        wind_ned = self.wind.get_wind()
        
        # Transform wind to body frame
        phi, theta, psi = self.state[6], self.state[7], self.state[8]
        R = self.rotation_matrix(phi, theta, psi)
        wind_body = R.T @ wind_ned[:3]  # Steady wind in body frame
        gust_body = self.wind.generate_wind_gust()  # Gust in body frame
        wind_total_body = wind_body + gust_body  # Total wind in body frame

        # Compute relative velocity
        u, v, w = self.state[3], self.state[4], self.state[5]
        ur = u - wind_total_body[0]
        vr = v - wind_total_body[1]
        wr = w - wind_total_body[2]

        # Compute airspeed, angle of attack, and sideslip angle
        self.Va = np.sqrt(ur**2 + vr**2 + wr**2)
        self.alpha = np.arctan2(wr, ur)
        self.beta = np.arcsin(vr / self.Va)

        # Compute aerodynamic coefficients
        CL = par.C_L_0 + par.C_L_alpha * self.alpha
        CD = par.C_D_0 + par.C_D_alpha * self.alpha
        CY = par.C_Y_0 + par.C_Y_beta * self.beta

        # Compute forces in body frame
        q_bar = 0.5 * par.rho * self.Va**2 * par.S_wing  # Dynamic pressure
        lift = q_bar * CL
        drag = q_bar * CD
        side_force = q_bar * CY

        # Longitudinal forces (fx, fz)
        fx = -drag * np.cos(self.alpha) + lift * np.sin(self.alpha)
        fz = -drag * np.sin(self.alpha) - lift * np.cos(self.alpha)

        # Lateral force (fy)
        fy = side_force


        # Add gravitational forces
        fg_ned = np.array([0, 0, self.mass * par.gravitational_acceleration])
        fg_body = R.T @ fg_ned
        fx += fg_body[0]
        fy += fg_body[1]
        fz += fg_body[2]

        # Compute moments
        Mx = q_bar * par.S_wing * par.b * (
            par.C_ell_0 + par.C_ell_beta * self.beta +
            par.C_ell_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_ell_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_ell_delta_a * delta.aileron +
            par.C_ell_delta_r * delta.rudder
        )

        My = q_bar * par.S_wing * par.c * (
            par.C_m_0 + par.C_m_alpha * self.alpha +
            par.C_m_q * (par.c / (2 * self.Va)) * self.state[10] +
            par.C_m_delta_e * delta.elevator
        )

        Mz = q_bar * par.S_wing * par.b * (
            par.C_n_0 + par.C_n_beta * self.beta +
            par.C_n_p * (par.b / (2 * self.Va)) * self.state[9] +
            par.C_n_r * (par.b / (2 * self.Va)) * self.state[11] +
            par.C_n_delta_a * delta.aileron +
            par.C_n_delta_r * delta.rudder
        )

        return np.array([fx, fy, fz]), np.array([Mx, My, Mz])
    
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

# Test Case
sim_time = par.start_time
positions = []
velocities = []
angular_velocities = []
times = []
mav = MAV(par.ts_simulation, par.mass, par.Jx, par.Jy, par.Jz, par.Jxz, par.initial_state)
delta = lambda: None
delta.elevator, delta.aileron, delta.rudder, delta.throttle = -0.2, 0.0, 0.005, 0.5

while sim_time < par.end_time:
    mav.update(delta)
    positions.append(mav.state[:3])
    velocities.append(mav.state[3:6])
    angular_velocities.append(mav.state[9:12])
    times.append(sim_time)
    sim_time += par.ts_simulation

# Plotting
positions = np.array(positions)
velocities = np.array(velocities)
angular_velocities = np.array(angular_velocities)

# 3D Position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='3D Flight Path')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Altitude (m)')
ax.legend()
plt.show()

# Velocities
plt.figure()
plt.plot(times, velocities[:, 0], label='u (m/s)')
plt.plot(times, velocities[:, 1], label='v (m/s)')
plt.plot(times, velocities[:, 2], label='w (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Angular Velocities
plt.figure()
plt.plot(times, angular_velocities[:, 0], label='p (rad/s)')
plt.plot(times, angular_velocities[:, 1], label='q (rad/s)')
plt.plot(times, angular_velocities[:, 2], label='r (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.show()