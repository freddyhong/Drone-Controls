"""
Copy and Pasted from github
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_rotation
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete


class Observer:
    def __init__(self, ts):
        # initialized estimated state message     
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f, 
            Q = np.diag([
                0.**2,  # pn
                0.**2,  # pe
                0.**2,  # pd
                0.**2,  # u
                0.**2,  # v
                0.**2,  # w
                0.**2,  # phi
                0.**2,  # theta
                0.**2,  # psi
                0.**2,  # bx
                0.**2,  # by
                0.**2,  # bz
                0.**2,  # wn
                0.**2,  # we
            ]),
            P0 = np.diag([
                0.**2,  # pn
                0.**2,  # pe
                0.**2,  # pd
                0.**2,  # u
                0.**2,  # v
                0.**2,  # w
                np.radians(0.)**2,  # phi
                np.radians(0.)**2,  # theta
                np.radians(0.)**2,  # psi
                np.radians(0.)**2,  # bx
                np.radians(0.)**2,  # by
                np.radians(0.)**2,  # bz
                0.**2,  # wn
                0.**2,  # we
            ]),
            xhat0 = np.array([[
                MAV.pn0,  # pn
                MAV.pe0,   # pe
                MAV.pd0,   # pd
                MAV.Va0,     # u
                0.,          # v
                0.,          # w
                0.,          # phi
                0.,          # theta
                MAV.psi0,    # psi
                0.,          # bx
                0.,          # by
                0.,          # bz
                0.,          # wn
                0.           # we
            ]]).T,
            Qu = np.diag([
                SENSOR.gyro_sigma**2,
                SENSOR.gyro_sigma**2,
                SENSOR.gyro_sigma**2,
                SENSOR.accel_sigma**2,
                SENSOR.accel_sigma**2,
                SENSOR.accel_sigma**2
            ]),
            Ts=ts,
            N=10
            )
        self.R_analog = np.diag([
            SENSOR.abs_pres_sigma**2,
            SENSOR.diff_pres_sigma**2,
            (0.01)**2
        ])
        self.R_gps = np.diag([
            SENSOR.gps_n_sigma**2,
            SENSOR.gps_e_sigma**2,
            SENSOR.gps_Vg_sigma**2,
            SENSOR.gps_course_sigma**2
        ])
        self.R_pseudo = np.diag([
                    (0.1)**2,  # pseudo measurement #1        
                    (0.1)**2,  # pseudo measurement #2
                    ])
        initial_measurements = MsgSensors()

        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        self.analog_threshold = stats.chi2.isf(q=0.01, df=3)
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=2)
        self.gps_n_old = None
        self.gps_e_old = None
        self.gps_Vg_old = None
        self.gps_course_old = None

        self.estimated_state = MsgState()
        self.elapsed_time = 0.0

    def update(self, measurement):
        # system input
        u = np.array([[
            measurement.gyro_x, 
            measurement.gyro_y, 
            measurement.gyro_z,
            measurement.accel_x, 
            measurement.accel_y, 
            measurement.accel_z,
            ]]).T
        xhat, P = self.ekf.propagate_model(u)
        # update with analog measurement
        y_analog = np.array([
            [measurement.abs_pressure],
            [measurement.diff_pressure],
            [0.0], # sideslip pseudo measurement
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_analog, 
            u=u,
            h=self.h_analog,
            R=self.R_analog)
        # update with wind triangle pseudo measurement
        y_pseudo = np.array([
            [0.],
            [0.], 
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_pseudo, 
            u=u,
            h=self.h_pseudo,
            R=self.R_pseudo)
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            state = to_MsgState(xhat) 
                # need to do this to get the current chi to wrap meaurement
            y_chi = wrap(measurement.gps_course, state.chi)
            y_gps = np.array([
                [measurement.gps_n], 
                [measurement.gps_e], 
                [measurement.gps_Vg], 
                [y_chi]])
            xhat, P = self.ekf.measurement_update(
                y=y_gps, 
                u=u,
                h=self.h_gps,
                R=self.R_gps)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course
        # convert internal xhat to MsgState format
        self.estimated_state = to_MsgState(xhat)
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) \
            - self.estimated_state.bx
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) \
            - self.estimated_state.by
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) \
            - self.estimated_state.bz
        self.elapsed_time += SIM.ts_control
        return self.estimated_state

    def f(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # system dynamics for propagation model: xdot = f(x, u)

        """ Continuous‐time dynamics ẋ = f(x,u) for the EKF. """
        # # unpack state
        # pos      = x[0:3]    # pn, pe, pd
        # vel_body = x[3:6]    # u, v, w
        # # phi, theta, psi = x[6], x[7], x[8]
        # # bias     = x[9:12]   # bx, by, bz
        # # wn, we   = x[12], x[13]
        # phi      = x.item(6)          # scalar
        # theta    = x.item(7)          # scalar
        # psi      = x.item(8)          # scalar
        # bias     = x[9:12]            # array (3×1)
        # wn       = x.item(12)         # scalar
        # we       = x.item(13)         # scalar

        # # unpack inputs
        # y_gyro  = u[0:3]   # gyro measurements
        # y_accel = u[3:6]   # accel measurements

        # # remove biases from body rates
        # p = y_gyro[0] - bias[0]
        # q = y_gyro[1] - bias[1]
        # r = y_gyro[2] - bias[2]
        # omega_body = np.array([[p], [q], [r]])

        # # rotation and wind in inertial frame
        # R = euler_to_rotation(phi, theta, psi)
        # wind_world = np.array([[wn], [we], [0.0]])

        # # 1) Position kinematics (ground velocity)
        # pos_dot = R @ vel_body + wind_world

        # # 2) Body‐frame accelerations from specific force
        # u_b, v_b, w_b = vel_body.flatten()
        # ax_m, ay_m, az_m = y_accel.flatten()
        # g = MAV.gravity


        # u_dot = ax_m - ( q*w_b - r*v_b ) +   g * np.sin(theta)
        # v_dot = ay_m - ( r*u_b - p*w_b ) -   g * np.cos(theta)*np.sin(phi)
        # w_dot = az_m - ( p*v_b - q*u_b ) -   g * np.cos(theta)*np.cos(phi)
        # vel_dot = np.array([[u_dot], [v_dot], [w_dot]])

        # # 3) Euler‐angle kinematics
        # Theta = x[6:9]
        # Theta_dot = S(Theta) @ omega_body

        # # 4) Gyro‐biases and wind assumed constant
        # bias_dot = np.zeros((3,1))
        # wind_dot = np.zeros((2,1))

        # # concatenate into full state‐derivative
        # xdot = np.concatenate([
        #     pos_dot,
        #     vel_dot,
        #     Theta_dot,
        #     bias_dot,
        #     wind_dot
        # ], axis=0)

        # return xdot
        # 1) Unpack state as scalars or explicit column vectors
        pn, pe, pd = x.item(0), x.item(1), x.item(2)
        u_b = x.item(3)
        v_b = x.item(4)
        w_b = x.item(5)
        phi   = x.item(6)
        theta = x.item(7)
        psi   = x.item(8)
        bx     = x.item(9)
        by     = x.item(10)
        bz     = x.item(11)
        wn     = x.item(12)
        we     = x.item(13)

        vel_body = np.array([[u_b], [v_b], [w_b]])  # 3×1

        # 2) Unpack inputs as scalars
        gyro_x  = u.item(0)
        gyro_y  = u.item(1)
        gyro_z  = u.item(2)
        accel_x = u.item(3)
        accel_y = u.item(4)
        accel_z = u.item(5)

        # 3) Body‐rates removing bias
        p = gyro_x - bx
        q = gyro_y - by
        r = gyro_z - bz
        omega_body = np.array([[p], [q], [r]])     # 3×1

        # 4) Build rotation matrix and wind in NED
        R = euler_to_rotation(phi, theta, psi)     # 3×3
        wind_world = np.array([[wn], [we], [0.0]]) # 3×1

        # 5) Position kinematics: inertial velocity = R·v_body + wind
        pos_dot = R @ vel_body + wind_world       # 3×1

        # 6) Specific‐force model → body‐frame accelerations
        g = MAV.gravity
        u_dot = accel_x - (q*w_b - r*v_b) +   g * np.sin(theta)
        v_dot = accel_y - (r*u_b - p*w_b) -   g * np.cos(theta)*np.sin(phi)
        w_dot = accel_z - (p*v_b - q*u_b) -   g * np.cos(theta)*np.cos(phi)
        vel_dot = np.array([[u_dot], [v_dot], [w_dot]])  # 3×1

        # 7) Euler‐angle kinematics: Θ̇ = S(Θ)·ω_body
        Theta = np.array([phi, theta, psi])    # 1D of length 3
        Theta_dot = S(Theta) @ omega_body      # 3×1

        # 8) Biases and wind are assumed constant
        bias_dot = np.zeros((3,1))
        wind_dot = np.zeros((2,1))

        # 9) Stack into full ẋ (14×1)
        xdot = np.vstack([
            pos_dot,      # pṅ, pė, pḋ
            vel_dot,      # u̇, v̇, ẇ
            Theta_dot,    # φ̇, θ̇, ψ̇
            bias_dot,     # ḃx, ḃy, ḃz
            wind_dot      # ẇn, ẇe
        ])

        return xdot

    def h_analog(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        
        """Analog sensor model: [abs_pres; diff_pres; sideslip]"""
        # unpack state
        pd       = x.item(2)           # down position
        vel_body = x[3:6]              # body velocities u, v, w
        phi      = x.item(6)
        theta    = x.item(7)
        psi      = x.item(8)
        wn       = x.item(12)          # wind north
        we       = x.item(13)          # wind east

        # 1) Absolute pressure from hydrostatic model
        h = -pd  # altitude above reference
        abs_pres = MAV.rho * MAV.gravity * h

        # 2) Differential pressure from dynamic pressure: q = ½ ρ Va²
        R = euler_to_rotation(phi, theta, psi)
        wind_world = np.array([[wn], [we], [0.0]])
        wind_body = R.T @ wind_world
        vel_rel = vel_body - wind_body  # relative airspeed in body frame
        Va = np.linalg.norm(vel_rel)
        diff_pres = 0.5 * MAV.rho * Va**2

        # 3) Sideslip angle β = asin(v_rel / Va)
        sideslip = np.arcsin(vel_rel[1, 0] / Va)

        return np.array([
            [abs_pres],
            [diff_pres],
            [sideslip]
        ])


    def h_gps(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        """GPS measurement model: [pn; pe; Vg; chi]"""
        # unpack state
        pn = x.item(0)
        pe = x.item(1)
        # explicit 3×1 body‐frame velocity
        u_b = x.item(3)
        v_b = x.item(4)
        w_b = x.item(5)
        vel_body = np.array([[u_b], [v_b], [w_b]])
        # angles and wind as scalars
        phi   = x.item(6)
        theta = x.item(7)
        psi   = x.item(8)
        wn    = x.item(12)
        we    = x.item(13)

        # rotation to inertial frame
        R = euler_to_rotation(phi, theta, psi)  # gets three floats now
        wind_world = np.array([[wn], [we], [0.0]])

        # ground‐frame velocity
        vel_world = R @ vel_body + wind_world    # 3×1

        # ground speed and course
        Vg  = np.hypot(vel_world[0,0], vel_world[1,0])
        chi = np.arctan2(vel_world[1,0], vel_world[0,0])

        return np.array([
            [pn],
            [pe],
            [Vg],
            [chi]
        ])

    def h_pseudo(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        """Wind‐triangle pseudo measurements:
           h1 = (R·vel_body)_N + wn − Vg·cos(chi)
           h2 = (R·vel_body)_E + we − Vg·sin(chi)
        """
        vel_body = x[3:6]  # shape (3,1)
        # unpack angles and wind as scalars
        phi   = x.item(6)
        theta = x.item(7)
        psi   = x.item(8)
        wn    = x.item(12)
        we    = x.item(13)

        # rotation to inertial frame
        R = euler_to_rotation(phi, theta, psi)  # now gets three floats

        # predicted inertial velocity from body motion
        vel_pred = R @ vel_body  # shape (3,1)

        # add wind to get total ground‐frame velocity
        vel_world = vel_pred + np.array([[wn], [we], [0.0]])

        # ground speed and course
        Vg  = np.hypot(vel_world[0,0], vel_world[1,0])
        chi = np.arctan2(vel_world[1,0], vel_world[0,0])

        # residuals for wind triangle (should be zero)
        h1 = vel_pred[0,0] + wn - Vg * np.cos(chi)
        h2 = vel_pred[1,0] + we - Vg * np.sin(chi)

        return np.array([[h1], [h2]])

def to_MsgState(x: np.ndarray) -> MsgState:
    state = MsgState()
    state.north = x.item(0)
    state.east = x.item(1)
    state.altitude = -x.item(2)
    vel_body = x[3:6]
    state.phi = x.item(6)
    state.theta = x.item(7)
    state.psi = x.item(8)
    state.bx = x.item(9)
    state.by = x.item(10)
    state.bz = x.item(11)
    state.wn = x.item(12)
    state.we = x.item(13)
    # estimate needed quantities that are not part of state
    R = euler_to_rotation(
        state.phi,
        state.theta,
        state.psi)
    vel_world = R @ vel_body
    wind_world = np.array([[state.wn], [state.we], [0]])
    wind_body = R.T @ wind_world
    vel_rel = vel_body - wind_body
    state.Va = np.linalg.norm(vel_rel)
    state.alpha = np.arctan(vel_rel.item(2) / vel_rel.item(0))
    state.beta = np.arcsin(vel_rel.item(1) / state.Va)
    state.Vg = np.linalg.norm(vel_world)
    state.chi = np.arctan2(vel_world.item(1), vel_world.item(0))
    return state


def cross(vec: np.ndarray)->np.ndarray:
    return np.array([[0, -vec.item(2), vec.item(1)],
                     [vec.item(2), 0, -vec.item(0)],
                     [-vec.item(1), vec.item(0), 0]])


def S(Theta:np.ndarray)->np.ndarray:
    return np.array([[1,
                      np.sin(Theta.item(0)) * np.tan(Theta.item(1)),
                      np.cos(Theta.item(0)) * np.tan(Theta.item(1))],
                     [0,
                      np.cos(Theta.item(0)),
                      -np.sin(Theta.item(0))],
                     [0,
                      (np.sin(Theta.item(0)) / np.cos(Theta.item(1))),
                      (np.cos(Theta.item(0)) / np.cos(Theta.item(1)))]
                     ])