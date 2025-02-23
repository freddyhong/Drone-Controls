import numpy as np
from aerosonde_parameters import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MsgState:
    def __init__(self):
        self.north = 0.      # inertial north position in meters
        self.east = 0.      # inertial east position in meters
        self.down = 10.       # inertial altitude in meters
        self.phi = 10.     # roll angle in radians
        self.theta = 2.   # pitch angle in radians
        self.psi = 2.     # yaw angle in radians
        self.Va = 25.      # airspeed in meters/sec
        self.alpha = 0.   # angle of attack in radians
        self.beta = 0.    # sideslip angle in radians
        self.p = 1.       # roll rate in radians/sec
        self.q = 0.       # pitch rate in radians/sec
        self.r = 2.       # yaw rate in radians/sec
        self.Vg = 25.      # groundspeed in meters/sec
        self.gamma = 0.   # flight path angle in radians
        self.chi = 0.     # course angle in radians
        self.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.we = 0.      # inertial windspeed in east direction in meters/sec

class MsgDelta:
    def __init__(self,
                 elevator=0.0,
                 aileron=0.0,
                 rudder=0.0,
                 throttle=0.5):
        self.elevator = elevator 
        self.aileron = aileron  
        self.rudder = rudder  
        self.throttle = throttle 

    def to_array(self):
        return np.array([[self.elevator],
                         [self.aileron],
                         [self.rudder],
                         [self.throttle]
                         ])

    def from_array(self, u):
        self.elevator = u.item(0)
        self.aileron = u.item(1)
        self.rudder = u.item(2)
        self.throttle = u.item(3)

    def print(self):
        print('elevator=', self.elevator,
              'aileron=', self.aileron,
              'rudder=', self.rudder,
              'throttle=', self.throttle)

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts

        self._state = np.array([
            [north0],  
            [east0],   
            [down0],   
            [u0],     
            [v0],     
            [w0],   
            [p0],      
            [q0],      
            [r0],      
            [phi0],   
            [theta0], 
            [psi0],   
        ])
        self.true_state = MsgState()
        self._wind = np.zeros((6, 1))
        self._Va = u0 
        self._alpha = 0  
        self._beta = 0  


    def update(self, delta, wind):
        self._update_velocity_data(wind)
        forces_moments = self._forces_moments(delta) # compute forces and moments based on control inputs (delta) and wind
        self._rk4_step(forces_moments)
        self._update_true_state()
        
    def external_set_state(self, new_state):
        self._state = new_state

    def _rk4_step(self, forces_moments):
        dt = self._ts_simulation
        k1 = self._f(self._state, forces_moments)
        k2 = self._f(self._state + dt/2 * k1, forces_moments)
        k3 = self._f(self._state + dt/2 * k2, forces_moments)
        k4 = self._f(self._state + dt * k3, forces_moments)
        self._state[0:12] = dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def _f(self, state, forces_moments):
        north, east, down, u, v, w, p, q, r, phi, theta, psi = state.flatten()
        fx, fy, fz, Mx, My, Mz = forces_moments.flatten()

        north_dot = np.cos(theta) * np.cos(psi) * u + (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w
        east_dot = np.cos(theta) * np.sin(psi) * u + (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w
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
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        return np.array([[north_dot, east_dot, down_dot, u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot]]).T

    def _update_velocity_data(self, wind=np.zeros((6, 1))):
        steady_state = wind[0:3]  # in NED
        gust = wind[3:6]  # in body frame

        phi = self._state.item(9)
        theta = self._state.item(10)
        psi = self._state.item(11)

        # rotation matrix from body to NED
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
             np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
             np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
             np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
             np.cos(phi) * np.cos(theta)]])  

        wind_body_steady = R.T @ steady_state  # convert steady-state wind vector from NED to body frame
        wind_body = wind_body_steady + gust    # add the gust
        self._wind = R @ wind_body             # convert total wind to NED frame

        u, v, w = self._state[3:6]  # velocity in body frame

        # velocity vector relative to the airmass in body frame
        ur = u - wind_body[0]
        vr = v - wind_body[1]
        wr = w - wind_body[2]

        # compute airspeed, AoA, and sideslip angle
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        self._alpha = np.arctan2(wr, ur)
        self._beta = np.arcsin(vr / self._Va)

    def _motor_thrust_torque(self, Va, delta_t):
        v_in = V_max * delta_t

        # angular speed of propeller
        a = rho * D_prop**5 / ((2 * np.pi)**2) * C_Q0
        b = (rho * D_prop**4 / (2 * np.pi)) * C_Q1 * Va + KQ**2 / R_motor
        c = rho * D_prop**3 * C_Q2 * Va**2 - (KQ * v_in / R_motor) + KQ * i0
        omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # thrust and torque due to propeller
        C_T = C_T0 + C_T1 * (2 * np.pi * omega_p / Va) + C_T2 * (2 * np.pi * omega_p / Va)**2
        C_Q = C_Q0 + C_Q1 * (2 * np.pi * omega_p / Va) + C_Q2 * (2 * np.pi * omega_p / Va)**2
        thrust_prop = rho * (omega_p / (2 * np.pi))**2 * D_prop**4 * C_T
        torque_prop = rho * (omega_p / (2 * np.pi))**2 * D_prop**5 * C_Q

        return thrust_prop, torque_prop

    def _forces_moments(self, delta):
        phi, theta, psi = self._state[6:9]
        p, q, r = self._state[9:12]

        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        # compute gravitational forces
        fg_ned = np.array([[0], [0], [-mass * gravity]])
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)],
            [np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), 
             np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), 
             np.sin(phi) * np.cos(theta)],
            [np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi), 
             np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi), 
             np.cos(phi) * np.cos(theta)]
        ])  # rotation matrix from body to NED
        fg_body = R.T @ fg_ned
        fg_x, fg_y, fg_z = fg_body.flatten()

        # compute Lift and Drag coefficients (CL, CD)
        C_L = C_L_0 + C_L_alpha * self._alpha
        C_D = C_D_0 + C_D_alpha * self._alpha

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)
        f_x = thrust_prop

        # compute longitudinal forces in body frame (fx, fz)
        fx_fz = 0.5 * rho * self._Va**2 * S_wing * np.array([
            (-C_D * np.cos(self._alpha) + C_L * np.sin(self._alpha))
            + (-C_D_q * np.cos(self._alpha) + C_L_q * np.sin(self._alpha)) * (c / (2 * self._Va)) * q
            + (-C_D_delta_e * np.cos(self._alpha) + C_L_delta_e * np.sin(self._alpha)) * delta_e,
            
            (-C_D * np.sin(self._alpha) - C_L * np.cos(self._alpha))
            + (-C_D_q * np.sin(self._alpha) - C_L_q * np.cos(self._alpha)) * (c / (2 * self._Va)) * q
            + (-C_D_delta_e * np.sin(self._alpha) - C_L_delta_e * np.cos(self._alpha)) * delta_e])

        f_x = fx_fz[0] + thrust_prop
        f_z = fx_fz[1]

        # compute lateral forces in body frame (fy)
        f_y = 0.5 * rho * self._Va**2 * S_wing * (C_Y_0 + C_Y_beta * self._beta
            + C_Y_p * (b / (2 * self._Va)) * p
            + C_Y_r * (b / (2 * self._Va)) * r
            + C_Y_delta_a * delta_a
            + C_Y_delta_r * delta_r)
        
        # compute longitudinal torque in body frame (My)
        My = 0.5 * rho * self._Va**2 * S_wing * c * (
            C_m_0 + C_m_alpha * self._alpha
            + C_m_q * (c / (2 * self._Va)) * q
            + C_m_delta_e * delta_e)

        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * rho * self._Va**2 * S_wing * b * (
            C_ell_0 + C_ell_beta * self._beta
            + C_ell_p * (b / (2 * self._Va)) * p
            + C_ell_r * (b / (2 * self._Va)) * r
            + C_ell_delta_a * delta_a
            + C_ell_delta_r * delta_r)
   
        Mz = 0.5 * rho * self._Va**2 * S_wing * b * (
            C_n_0 + C_n_beta * self._beta
            + C_n_p * (b / (2 * self._Va)) * p
            + C_n_r * (b / (2 * self._Va)) * r
            + C_n_delta_a * delta_a
            + C_n_delta_r * delta_r)
   
        forces_moments = np.array([[f_x + fg_x, f_y + fg_y, f_z + fg_z, Mx, My, Mz]]).T
        return forces_moments

    def _update_true_state(self):
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.down = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = self._state.item(9)
        self.true_state.theta = self._state.item(10)
        self.true_state.psi = self._state.item(11)
        self.true_state.p = self._state.item(6)
        self.true_state.q = self._state.item(7)
        self.true_state.r = self._state.item(8)


# simulation 
dt = 0.01  # time step 
t_end = 10  # end time 
num_steps = int(t_end / dt)  
time = np.linspace(0, t_end, num_steps) 

mav = MavDynamics(Ts=dt)

north = np.zeros(num_steps)
east = np.zeros(num_steps)
down = np.zeros(num_steps)
u = np.zeros(num_steps)
v = np.zeros(num_steps)
w = np.zeros(num_steps)
p = np.zeros(num_steps)
q = np.zeros(num_steps)
r = np.zeros(num_steps)

delta = MsgDelta(elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.5) # control inputs
wind = np.zeros((6, 1)) # wind vector (steadystate + gust)

for i in range(num_steps):
    mav.update(delta, wind)
    
    north[i] = mav._state.item(0)
    east[i] = mav._state.item(1)
    down[i] = mav._state.item(2)
    u[i] = mav._state.item(3)
    v[i] = mav._state.item(4)
    w[i] = mav._state.item(5)
    p[i] = mav._state.item(6)
    q[i] = mav._state.item(7)
    r[i] = mav._state.item(8)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(north, east, down, label='Aircraft Position')
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.set_title('3D Position Projection of Aircraft')
ax.legend()

plt.figure(figsize=(10, 6))
plt.plot(time, u, label="u (velocity in x)")
plt.plot(time, v, label="v (velocity in y)")
plt.plot(time, w, label="w (velocity in z)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Translational Velocities Over Time")
plt.grid()

fig, ax = plt.subplots(2, figsize=(11, 8))
ax[0].plot(time, p, label="p")
ax[0].plot(time, q, label="q")
ax[0].plot(time, r, label="r")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angular Velocity (rad/s)")
ax[0].legend()
ax[0].set_title("Simulation Rotational Velocities")
ax[0].grid()

plt.tight_layout()
plt.show()
