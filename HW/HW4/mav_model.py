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
        self.wind = Wind()  
        self.Va = 0.0  # Airspeed
        self.alpha = 0.0  # Angle of attack
        self.beta = 0.0  # Sideslip angle

    def update(self, delta, wind_enabled=True):
        def f(state, forces, moments):
            return self.equations_of_motion(state, forces, moments)

        forces, moments = self.compute_forces_moments(delta, wind_enabled)

        # RK4 Integration
        k1 = f(self.state, forces, moments)
        k2 = f(self.state + 0.5 * self.ts * k1, forces, moments)
        k3 = f(self.state + 0.5 * self.ts * k2, forces, moments)
        k4 = f(self.state + self.ts * k3, forces, moments)

        self.state = self.state + (self.ts / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self._update_velocity_data(delta)

        # Debug output
        print(f"--- Simulation Step ---")
        print(f"State: {self.state}")
        print(f"State Derivatives: {k1}")
        print(f"Forces: {forces}")
        print(f"Moments: {moments}")
        print(f"Wind (NED frame): {self.wind.get_wind()}")
        print(f"Airspeed (Va): {self.Va}")
        print(f"Alpha (Angle of Attack): {self.alpha}")
        print(f"Beta (Sideslip Angle): {self.beta}")
        print("----------------------\n")
    
    def _update_velocity_data(self, delta):
        self.compute_forces_moments(delta, wind_enabled=True)

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

    def compute_forces_moments(self, delta, wind_enabled=True):
        if wind_enabled:
            wind_ned = self.wind.get_wind()
        else:
            wind_ned = np.array([0.0, 0.0, 0.0])  # No wind
        
        # Transform wind to body frame
        phi, theta, psi = self.state[6], self.state[7], self.state[8]
        R = self.rotation_matrix(phi, theta, psi)
        wind_body = R.T @ wind_ned[:3]  # convert steady wind from NED to body frame
        gust_body = self.wind.generate_wind_gust() if wind_enabled else np.array([0.0, 0.0, 0.0])  # Gust in body frame
        wind_total_body = wind_body + gust_body  # Total wind in body frame

        # velocity vector relative to the airmass in body frame
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

        # compute forces in body frame w/ states

        # compute longitudinal forces in body frame (fx, fz)
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

        # compute lateral forces in body frame (fy)
        fy = 0.5 * par.rho * self.Va**2 * par.S_wing * (
            par.C_Y_0 + par.C_Y_beta * self.beta 
            + par.C_Y_p * (par.b / (2 * self.Va)) * self.state[9] 
            + par.C_Y_r * (par.b / (2 * self.Va)) * self.state[11] 
            + par.C_Y_delta_a * delta.aileron 
            + par.C_Y_delta_r * delta.rudder
            )

        # add gravitational forces
        fg_ned = np.array([0, 0, self.mass * par.gravitational_acceleration])
        fg_body = R.T @ fg_ned
        fx += fg_body[0]
        fy += fg_body[1]
        fz += fg_body[2]


        # Compute moments
        # compute lateral torques in body frame (Mx, Mz)
        q_bar = 0.5 * par.rho * self.Va**2 * par.S_wing # Dynamic pressure

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

        # compute longitudinal torque in body frame (My)
        My = q_bar * par.S_wing * par.c * (
            par.C_m_0 + par.C_m_alpha * self.alpha +
            par.C_m_q * (par.c / (2 * self.Va)) * self.state[10] +
            par.C_m_delta_e * delta.elevator
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

# Store positions for both cases
positions_with_wind = []
positions_without_wind = []
velocities_with_wind = []
velocities_without_wind = []
angular_velocities_with_wind = []
angular_velocities_without_wind = []
times = []

mav_with_wind = MAV(par.ts_simulation, par.mass, par.Jx, par.Jy, par.Jz, par.Jxz, par.initial_state)
mav_without_wind = MAV(par.ts_simulation, par.mass, par.Jx, par.Jy, par.Jz, par.Jxz, par.initial_state)

delta = lambda: None
delta.elevator, delta.aileron, delta.rudder, delta.throttle = -0.2, 0.0, 0.005, 0.5

while sim_time < par.end_time:
    mav_with_wind.update(delta, wind_enabled=True)
    mav_without_wind.update(delta, wind_enabled=False)

    positions_with_wind.append(mav_with_wind.state[:3])
    positions_without_wind.append(mav_without_wind.state[:3])
    velocities_with_wind.append(mav_with_wind.state[3:6])
    velocities_without_wind.append(mav_without_wind.state[3:6])
    angular_velocities_with_wind.append(mav_with_wind.state[9:12])
    angular_velocities_without_wind.append(mav_without_wind.state[9:12])
    times.append(sim_time)
    sim_time += par.ts_simulation

positions_with_wind = np.array(positions_with_wind)
positions_without_wind = np.array(positions_without_wind)
velocities_with_wind = np.array(velocities_with_wind)
velocities_without_wind = np.array(velocities_without_wind)
angular_velocities_with_wind = np.array(angular_velocities_with_wind)
angular_velocities_without_wind = np.array(angular_velocities_without_wind)

# Plotting
# 3D Position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_with_wind[:, 0], positions_with_wind[:, 1], positions_with_wind[:, 2], label='With Wind')
ax.plot(positions_without_wind[:, 0], positions_without_wind[:, 1], positions_without_wind[:, 2], label='Without Wind', linestyle='dashed')
ax.set_xlabel('North')
ax.set_ylabel('East')
ax.set_zlabel('Altitude')
ax.legend()
plt.show()

# Velocities (u, v, w)
plt.figure()
plt.plot(times, velocities_with_wind[:, 0], label='u with Wind')
plt.plot(times, velocities_without_wind[:, 0], label='u without Wind', linestyle='dashed')
plt.plot(times, velocities_with_wind[:, 1], label='v with Wind')
plt.plot(times, velocities_without_wind[:, 1], label='v without Wind', linestyle='dashed')
plt.plot(times, velocities_with_wind[:, 2], label='w with Wind')
plt.plot(times, velocities_without_wind[:, 2], label='w without Wind', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Components (u, v, w) With and Without Wind')
plt.show()

# Angular Velocities (p, q, r)
plt.figure()
plt.plot(times, angular_velocities_with_wind[:, 0], label='p with Wind')
plt.plot(times, angular_velocities_without_wind[:, 0], label='p without Wind', linestyle='dashed')
plt.plot(times, angular_velocities_with_wind[:, 1], label='q with Wind')
plt.plot(times, angular_velocities_without_wind[:, 1], label='q without Wind', linestyle='dashed')
plt.plot(times, angular_velocities_with_wind[:, 2], label='r with Wind')
plt.plot(times, angular_velocities_without_wind[:, 2], label='r without Wind', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.legend()
plt.title('Angular Velocity Components (p, q, r) With and Without Wind')
plt.show()
