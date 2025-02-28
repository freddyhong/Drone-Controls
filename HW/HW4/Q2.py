import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import parameters as par

class Wind:
    def __init__(self):
        self.steady_wind = np.array(par.steady_wind)
        self.gust_amplitude = np.array(par.gust_amplitude)
    
    def generate_wind_gust(self):
        return np.random.uniform(-1, 1, size=3) * self.gust_amplitude
    
    def get_wind(self):
        return self.steady_wind + self.generate_wind_gust()

class MAV:
    def __init__(self, ts, mass, Jx, Jy, Jz, Jxz, initial_state):
        self.ts = ts
        self.mass = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxz = Jxz
        self.state = np.array(initial_state)
        self.wind = Wind()

    def update(self, delta):
        forces, moments = self.compute_forces_moments(delta)
        wind_effect = self.wind.get_wind()
        forces[:3] += wind_effect
        self.state = self.state + self.ts * self.equations_of_motion(self.state, forces, moments)
    
    def equations_of_motion(self, state, forces, moments):
        u, v, w = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]
        p, q, r = state[9], state[10], state[11]
        fx, fy, fz = forces
        Mx, My, Mz = moments
        
        north_dot = u * np.cos(theta) * np.cos(psi) + v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        east_dot = u * np.cos(theta) * np.sin(psi) + v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
        down_dot = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(phi) * np.cos(theta)

        u_dot = r * v - q * w + fx / self.mass
        v_dot = p * w - r * u + fy / self.mass
        w_dot = q * u - p * v + fz / self.mass

        p_dot = moments[0] / self.Jx
        q_dot = moments[1] / self.Jy
        r_dot = moments[2] / self.Jz

        phi_dot = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        return np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot])
    
    def compute_forces_moments(self, delta):
        CL = par.C_L_0 + par.C_L_alpha * delta.elevator
        CD = par.C_D_0 + par.C_D_alpha * delta.elevator
        L = 0.5 * par.rho * par.S_wing * CL
        D = 0.5 * par.rho * par.S_wing * CD
        fx = delta.throttle * 50.0 - D
        fz = -L - self.mass * par.gravitational_acceleration
        fy = 0.5 * par.rho * par.S_wing * delta.aileron
        Mx = delta.aileron * 5.0
        My = delta.elevator * 10.0
        Mz = delta.rudder * 2.0
        return np.array([fx, fy, fz]), np.array([Mx, My, Mz])

# Test Case 1
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

positions = np.array(positions)
velocities = np.array(velocities)
angular_velocities = np.array(angular_velocities)

# Plot 3D Position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='3D Flight Path')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Altitude (m)')
ax.legend()
plt.show()

# Plot Velocities
plt.figure()
plt.plot(times, velocities[:, 0], label='u (m/s)')
plt.plot(times, velocities[:, 1], label='v (m/s)')
plt.plot(times, velocities[:, 2], label='w (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.show()

# Plot Angular Velocities
plt.figure()
plt.plot(times, angular_velocities[:, 0], label='p (rad/s)')
plt.plot(times, angular_velocities[:, 1], label='q (rad/s)')
plt.plot(times, angular_velocities[:, 2], label='r (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.show()
