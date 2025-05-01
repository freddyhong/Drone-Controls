# """
# autopilot block for mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last Update:
#         2/6/2019 - RWB
# """
# # change values in control_parameters.py file to tune this controller

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
# from tools.wrap import wrap
# # from controllers.pi_control import PIControl
# from controllers.pid_control import pidControl, piControl, pdControlWithRate
# # from controllers.pd_control_with_rate import PDControlWithRate
# from message_types.msg_state import MsgState
# from message_types.msg_delta import MsgDelta


# class Autopilot:
#     def __init__(self, ts_control):
#         # instantiate lateral controllers
#         self.roll_from_aileron = pdControlWithRate(
#                         kp=AP.roll_kp,
#                         kd=AP.roll_kd,
#                         limit=np.radians(45))
#         self.course_from_roll = pidControl(
#                         kp=AP.course_kp,
#                         ki=AP.course_ki,
#                         Ts=ts_control,
#                         limit=np.radians(30))
#         self.yaw_damper = TransferFunction(
#                         num=np.array([[AP.yaw_damper_kr, 0]]),
#                         den=np.array([[1, AP.yaw_damper_p_wo]]),
#                         Ts=ts_control)

#         # instantiate lateral controllers
#         self.pitch_from_elevator = pdControlWithRate(
#                         kp=AP.pitch_kp,
#                         kd=AP.pitch_kd,
#                         limit=np.radians(45))
#         self.altitude_from_pitch = pidControl(
#                         kp=AP.altitude_kp,
#                         ki=AP.altitude_ki,
#                         Ts=ts_control,
#                         limit=np.radians(30))
#         self.airspeed_from_throttle = pidControl(
#                         kp=AP.airspeed_throttle_kp,
#                         ki=AP.airspeed_throttle_ki,
#                         Ts=ts_control,
#                         limit=1.0)
#         self.commanded_state = MsgState()

#     def update(self, cmd, state):

#         # lateral autopilot
#         chi_c = wrap (cmd.course_command , state.chi )
#         phi_c_unsaturated = self.course_from_roll.update(chi_c, state.chi)
#         phi_c_limit = np.pi / 4
#         phi_c = self.saturate(phi_c_unsaturated, -phi_c_limit, phi_c_limit)
#         delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
#         delta_r = self.yaw_damper.update(state.r)

#         # longitudinal autopilot
#         h_c = self.saturate(cmd.altitude_command, state.altitude - AP.altitude_zone, state.altitude + AP.altitude_zone)
#         theta_c = self.altitude_from_pitch.update(h_c, state.altitude)
#         delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
#         delta_t_unsat = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
#         delta_t = self.saturate(delta_t_unsat, 0, 1.0)
#         Va_command = cmd.airspeed_command

#         # construct control output and commanded states
#         delta = MsgDelta(elevator=delta_e,
#                          aileron=delta_a,
#                          rudder=delta_r,
#                          throttle=delta_t)
#         self.commanded_state.altitude = cmd.altitude_command
#         self.commanded_state.Va = cmd.airspeed_command
#         self.commanded_state.phi = phi_c
#         self.commanded_state.theta = theta_c
#         self.commanded_state.chi = cmd.course_command
#         return delta, self.commanded_state

#     def saturate(self, input, low_limit, up_limit):
#         if input <= low_limit:
#             output = low_limit
#         elif input >= up_limit:
#             output = up_limit
#         else:
#             output = input
#         return output















"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from tools.transfer_function import TransferFunction

def saturate(self, input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output

class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral-directional controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)
        # self.yaw_damper = TFControl(
        #                 k=AP.yaw_damper_kr,
        #                 n0=0.0,
        #                 n1=1.0,
        #                 d0=AP.yaw_damper_p_wo,
        #                 d1=1,
        #                 Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        
        # Course hold
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        
        # Roll hold
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        
        # Yaw damper
        delta_r = self.yaw_damper.update(state.r)
        
        # longitudinal autopilot
        # Airspeed hold with throttle
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        
        # Altitude hold
        theta_c = self.altitude_from_pitch.update(cmd.altitude_command, state.altitude)
        
        # Pitch hold
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        
        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                        aileron=delta_a,
                        rudder=delta_r,
                        throttle=delta_t)
        
        # Update commanded state to match format in LQR version
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = chi_c
        
        return delta, self.commanded_state