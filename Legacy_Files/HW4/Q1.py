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

    def update(self, forces, moments):
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

# Initialize MAV
mav = MAV(par.ts_simulation, par.mass, par.Jx, par.Jy, par.Jz, par.Jxz, par.initial_state)

# Initialize simulation time
sim_time = par.start_time
positions = []
times = []
velocities = []
angular_velocities = []

print("Starting MAV Simulation with Wind...")
while sim_time < par.end_time:
    forces = np.array([0, 0, -par.mass * par.gravitational_acceleration])
    moments = np.array([0, 0, 0])
    
    mav.update(forces, moments)
    positions.append([mav.state[0], mav.state[1], -mav.state[2]])
    velocities.append([mav.state[3], mav.state[4], mav.state[5]])
    angular_velocities.append([mav.state[9], mav.state[10], mav.state[11]])
    times.append(sim_time)
    
    print(f"Time: {sim_time:.2f} s, Position: {mav.state[:3]}")
    
    sim_time += par.ts_simulation

# Convert results for plotting
positions = np.array(positions)
velocities = np.array(velocities)
angular_velocities = np.array(angular_velocities)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Flight Path')
ax.set_xlabel('North Position (m)')
ax.set_ylabel('East Position (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Flight Path with Wind')
ax.legend()
plt.show()

# Plot velocities
plt.figure()
plt.plot(times, velocities[:, 0], label='u (m/s)')
plt.plot(times, velocities[:, 1], label='v (m/s)')
plt.plot(times, velocities[:, 2], label='w (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Linear Velocities')
plt.legend()
plt.show()

# Plot angular velocities
plt.figure()
plt.plot(times, angular_velocities[:, 0], label='p (rad/s)')
plt.plot(times, angular_velocities[:, 1], label='q (rad/s)')
plt.plot(times, angular_velocities[:, 2], label='r (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocities')
plt.legend()
plt.show()
