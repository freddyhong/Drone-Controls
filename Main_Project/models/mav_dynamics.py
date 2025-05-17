import numpy as np
# load message types
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from message_types.msg_state import MsgState
import parameters.aerosonde_parameters_max as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()

    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

    def _f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        
        # Extract the States
        north = state.item(0)
        east = state.item(1)
        down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
    

        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # Position Kinematics
        quaternions = np.array([[e0], [e1], [e2], [e3]])
        pos_dot = quaternion_to_rotation(quaternions)@np.array([[u], [v], [w]])
        north_dot = pos_dot.item(0)
        east_dot = pos_dot.item(1)
        down_dot = pos_dot.item(2)

        # Position Dynamics
        vel_dot = np.array([[r*v - q*w], [p*w - r*u], [q*u - p*v]]) + np.array([[fx], [fy], [fz]])/MAV.mass
        u_dot = vel_dot.item(0)
        v_dot = vel_dot.item(1)
        w_dot = vel_dot.item(2)

        # rotational kinematics
        e_all_dot = 0.5 *  np.array([[0, -p, -q, -r],
                                   [p, 0, r, -q],
                                   [q, -r, 0, p],
                                   [r, q, -p, 0]])@quaternions
        e0_dot = e_all_dot.item(0)
        e1_dot = e_all_dot.item(1)
        e2_dot = e_all_dot.item(2)
        e3_dot = e_all_dot.item(3)
        
        # rotatonal dynamics
        pqr_dot = np.array([[MAV.gamma1*p*q - MAV.gamma2*q*r], [MAV.gamma5*p*r - MAV.gamma6*(p**2 - r**2)], [MAV.gamma7*p*q - MAV.gamma1*q*r]]) + np.array([[MAV.gamma3*l + MAV.gamma4*n], [1/MAV.Jy*m], [MAV.gamma4*l + MAV.gamma8*n]])

        p_dot = pqr_dot.item(0)
        q_dot = pqr_dot.item(1)
        r_dot = pqr_dot.item(2)

        # collect the derivative of the states
        x_dot = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
        x_dot[0] = north_dot
        x_dot[1] = east_dot
        x_dot[2] = down_dot
        x_dot[3] = u_dot
        x_dot[4] = v_dot
        x_dot[5] = w_dot
        x_dot[6] = e0_dot
        x_dot[7] = e1_dot
        x_dot[8] = e2_dot
        x_dot[9] = e3_dot
        x_dot[10] = p_dot
        x_dot[11] = q_dot
        x_dot[12] = r_dot
        
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0