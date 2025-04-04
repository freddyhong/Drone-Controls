import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import parameters.mavsim_python_chap5_model_coef as TF  # using model_coeff from luctenburg temporarily
import parameters.aerosonde_parameters as MAV

"""we are tuning gains based on mavsim_python_chap5_model_coef
   not our own trim conditions as we still need to correct compute_models.py file
    which would output our own model_coef file """

gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim       # trim airspeed
rho = MAV.rho          # density of air
sigma = 0.05           # low pass filter gain for derivative

#----------roll loop-------------
wn_roll = 20 #7.0         
zeta_roll = 0.707     
roll_kp = (wn_roll**2)/TF.a_phi2
roll_kd = (2*zeta_roll*wn_roll - TF.a_phi1)/TF.a_phi2

#----------course loop-------------
wn_course = wn_roll/20.0  # course loop should be slower than roll loop
zeta_course = 1.0
course_kp = 2.0*zeta_course*wn_course*Va0/gravity
course_ki = (wn_course**2)*Va0/gravity

#----------yaw damper-------------
yaw_damper_p_wo = 0.45 #1.0  
yaw_damper_kr = 0.2 #0.5    

#----------pitch loop-------------
wn_pitch = 24 #5.0        
zeta_pitch = 0.7  
pitch_kp = (wn_pitch**2 - TF.a_theta2)/TF.a_theta3
pitch_kd = (2 * zeta_pitch*wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = pitch_kp*TF.a_theta3 / (TF.a_theta2 + pitch_kp * TF.a_theta3)

#----------altitude loop-------------
wn_altitude = wn_pitch/30.0  # altitude loop should be slower than pitch loop
zeta_altitude = 1.0
altitude_kp = 2.0*zeta_altitude*wn_altitude / (K_theta_DC * Va0)
altitude_ki = (wn_altitude**2) / (K_theta_DC * Va0)
altitude_zone = 10.0  # dead zone for altitude (m)

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 25
zeta_airspeed_throttle = 1.5 #0.707
airspeed_throttle_kp = (2.0*zeta_airspeed_throttle*wn_airspeed_throttle - TF.a_V1)/TF.a_V2
airspeed_throttle_ki = (wn_airspeed_throttle**2)/TF.a_V2