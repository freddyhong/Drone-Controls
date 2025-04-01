import sys
import os

# Add the parent directory of 'models' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
'''using model_coeff from luctenburg temporary -> fix this later'''
import parameters.mavsim_python_chap5_model_coef as TF
import parameters.aerosonde_parameters as MAV


#### TODO #####
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim
rho = 0 # density of air
sigma = 0  # low pass filter gain for derivative

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 0
zeta_roll = 0
roll_kp = 0
roll_kd = 0

#----------course loop-------------
wn_course = 0
zeta_course = 0
course_kp = 0
course_ki = 0

#----------yaw damper-------------
yaw_damper_p_wo = 0
yaw_damper_kr = 0

#----------pitch loop-------------
wn_pitch = 0
zeta_pitch = 0 
pitch_kp = 0
pitch_kd = 0
K_theta_DC = 0

#----------altitude loop-------------
wn_altitude = 0
zeta_altitude = 0
altitude_kp = 0
altitude_ki = 0
altitude_zone = 0

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 0
zeta_airspeed_throttle = 0
airspeed_throttle_kp = 0
airspeed_throttle_ki = 0