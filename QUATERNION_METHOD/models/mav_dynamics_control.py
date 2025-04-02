"""
mavDynamics Control
    - wind
    - forces/moments
    - aerodynamic model implementation
    - motor thrust/propulsion neglected
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._update_velocity_data(wind)
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3] # NED frame
        gust = wind[3:6] # body frame

        # convert steady-state wind from NED to body frame
        R = quaternion_to_rotation(self._state[6:10]) # bosy to NED
        wind_body = R.T @ steady_state + gust  # total wind in body frame 
        self._wind = R @ wind_body # convert total wind to NED frame

        # velocity vector relative to the airmass
        ur = self._state.item(3) - wind_body.item(0) # u - wind_x
        vr = self._state.item(4) - wind_body.item(1) # v - wind_y
        wr = self._state.item(5) - wind_body.item(2) # w - wind_z
        
        # compute airspeed 
        self._Va = np.linalg.norm([ur, vr, wr])

        # compute angle of attack
        if ur == 0:
            self._alpha = 0
        else:
            self._alpha = np.arctan2(wr, ur)

        # compute sideslip angle
        if self._Va == 0:
            self._beta = 0
        else:
            self._beta = np.arcsin(vr / self._Va)


    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
 
        # extract states 
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p, q, r = self._state[10:13]

        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        # Compute gravitational forces in body frame
        fg_x = -MAV.mass * MAV.gravity * np.sin(theta)
        fg_y = MAV.mass * MAV.gravity * np.cos(theta) * np.sin(phi)
        fg_z = MAV.mass * MAV.gravity * np.cos(theta) * np.cos(phi)

        # Compute Lift and Drag coefficients
        CL = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        CD = MAV.C_D_0 + MAV.C_D_alpha * self._alpha

        # Compute dynamic pressure
        q_dynamic = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing

        # Compute longitudinal force coefficients
        C_X = -CD * np.cos(self._alpha) + CL * np.sin(self._alpha)
        C_X_q = -MAV.C_D_q * np.cos(self._alpha) + MAV.C_L_q * np.sin(self._alpha)
        C_X_delta_e = -MAV.C_D_delta_e * np.cos(self._alpha) + MAV.C_L_delta_e * np.sin(self._alpha)

        # Compute lateral force coefficients
        C_Y = MAV.C_Y_0 + MAV.C_Y_beta * self._beta + MAV.C_Y_p * (MAV.b * p) / (2 * self._Va) + MAV.C_Y_r * (MAV.b * r) / (2 * self._Va)
        C_Y_delta_a = MAV.C_Y_delta_a
        C_Y_delta_r = MAV.C_Y_delta_r

        # Compute normal force coefficients
        C_Z = -CD * np.sin(self._alpha) - CL * np.cos(self._alpha)
        C_Z_q = -MAV.C_D_q * np.sin(self._alpha) - MAV.C_L_q * np.cos(self._alpha)
        C_Z_delta_e = -MAV.C_D_delta_e * np.sin(self._alpha) - MAV.C_L_delta_e * np.cos(self._alpha)

        # Propeller thrust and torque
        # thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)

        # Compute forces in body frame
        fx = fg_x + q_dynamic * (C_X + C_X_q * (MAV.c * q) / (2 * self._Va) + C_X_delta_e * delta_e) #+ thrust_prop
        fy = fg_y + q_dynamic * (C_Y + C_Y_delta_a * delta_a + C_Y_delta_r * delta_r)
        fz = fg_z + q_dynamic * (C_Z + C_Z_q * (MAV.c * q) / (2 * self._Va) + C_Z_delta_e * delta_e)

        # Compute moments in body frame
        Mx = q_dynamic * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + MAV.C_ell_p * (MAV.b * p) / (2 * self._Va) + MAV.C_ell_r * (MAV.b * r) / (2 * self._Va) + MAV.C_ell_delta_a * delta_a + MAV.C_ell_delta_r * delta_r) #- torque_prop
        My = q_dynamic * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + MAV.C_m_q * (MAV.c * q) / (2 * self._Va) + MAV.C_m_delta_e * delta_e)
        Mz = q_dynamic * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + MAV.C_n_p * (MAV.b * p) / (2 * self._Va) + MAV.C_n_r * (MAV.b * r) / (2 * self._Va) + MAV.C_n_delta_a * delta_a + MAV.C_n_delta_r * delta_r)

        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        return forces_moments
    '''
    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max * delta_t

        # Angular speed of propeller
        a = MAV.rho * MAV.C_Q0 * MAV.D_prop**5 / ((2 * np.pi)**2)
        b = MAV.rho * MAV.C_Q1 * MAV.D_prop**4 / (2 * np.pi) * Va + MAV.KQ**2 / MAV.R_motor
        c = MAV.rho * MAV.C_Q2 * MAV.D_prop**3 * Va**2 - MAV.KQ * v_in / MAV.R_motor + MAV.KQ * MAV.i0
        omega_p = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # thrust and torque due to propeller
        thrust_prop = MAV.rho * MAV.C_T0 * MAV.D_prop**4 * omega_p**2 / (4 * np.pi**2)
        torque_prop = MAV.rho * MAV.C_Q0 * MAV.D_prop**5 * omega_p**2 / (4 * np.pi**2)

        return thrust_prop, torque_prop
    '''

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)