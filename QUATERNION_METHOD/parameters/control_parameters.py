import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import parameters.mavsim_python_chap5_model_coef as TF  # using model_coeff from luctenburg temporarily
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim       # trim airspeed
rho = 1.293            # density of air
sigma = 0

a_phi1 = TF.a_phi1
a_phi2 = TF.a_phi2
delta_a_max = np.radians(35)  # reduce to prevent excessive roll rate
phi_max = np.radians(25)

#----------roll loop-------------
# wn_roll = 10.1 #7.0  
wn_roll = np.sqrt(abs(a_phi2) * delta_a_max / phi_max) / 0.8   
zeta_roll = 0.707 #3.5     
roll_kp = (wn_roll**2)/TF.a_phi2
roll_kd = (2*zeta_roll*wn_roll - TF.a_phi1)/TF.a_phi2

#----------course loop-------------
# wn_course = wn_roll/20.0  # course loop should be slower than roll loop
wn_course = wn_roll/6.0
zeta_course = 1.0
course_kp = 2.0*zeta_course*wn_course*Va0/gravity
course_ki = (wn_course**2)*Va0/gravity

#----------yaw damper-------------
yaw_damper_p_wo = 4.0 #10 #1.0  
yaw_damper_kr = 2.0 #0.5    

# #----------pitch loop-------------
# wn_pitch = 10.1 #5.0        15
# zeta_pitch = 0.707 #2.6 
# pitch_kp = (wn_pitch**2 - TF.a_theta2)/TF.a_theta3
# pitch_kd = (2 * zeta_pitch*wn_pitch - TF.a_theta1) / TF.a_theta3
# K_theta_DC = pitch_kp*TF.a_theta3 / (TF.a_theta2 + pitch_kp * TF.a_theta3)

#---------- PITCH LOOP -------------
a_theta1 = TF.a_theta1
a_theta2 = TF.a_theta2
a_theta3 = -TF.a_theta3

zeta_pitch = 0.55
wn_pitch = np.sqrt(abs(a_theta2) + abs(a_theta3)) / 2  # slower response
pitch_kp = (wn_pitch**2 - a_theta2) / a_theta3
pitch_kd = (2 * zeta_pitch * wn_pitch - a_theta1) / a_theta3
K_theta_DC = a_theta3 / (wn_pitch**2 + a_theta2)


# #----------altitude loop-------------
# wn_altitude = wn_pitch/30.0  # altitude loop should be slower than pitch loop
# zeta_altitude = 1.0
# altitude_kp = 2.0*zeta_altitude*wn_altitude / (K_theta_DC * Va0)
# altitude_ki = (wn_altitude**2) / (K_theta_DC * Va0)
# altitude_zone = 10.0  # dead zone for altitude (m)

# #---------airspeed hold using throttle---------------
# wn_airspeed_throttle = 1.5 #17
# zeta_airspeed_throttle = 2.1 #0.707
# airspeed_throttle_kp = (2.0*zeta_airspeed_throttle*wn_airspeed_throttle - TF.a_V1)/TF.a_V2
# airspeed_throttle_ki = (wn_airspeed_throttle**2)/TF.a_V2


#----------altitude loop-------------
zeta_altitude = 0.9
wn_altitude = wn_pitch / 8  # slower outer loop
altitude_kp = 2 * zeta_altitude * wn_altitude / (K_theta_DC * Va0)
altitude_ki = wn_altitude**2 / (K_theta_DC * Va0)
altitude_zone = 5

#---------airspeed hold using throttle---------------
a_V1 = TF.a_V1
a_V2 = TF.a_V2
zeta_airspeed_throttle = 1.0  # Keep a reasonable damping
wn_airspeed_throttle = 0.75  # Slower throttle response to smooth out the oscillations
airspeed_throttle_kp = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - a_V1) / a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / a_V2