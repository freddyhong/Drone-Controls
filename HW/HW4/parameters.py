import numpy as np

mass = 11.0
Jx = 0.8244
Jy = 1.135
Jz = 1.759
Jxz = 0.1204
ts_simulation = 0.01
start_time = 0.0
end_time = 10.0

gravitational_acceleration = 9.81

# Wind parameters
steady_wind = [5.0, 0.0, 0.0]  # Steady wind in NED frame (m/s)
gust_amplitude = np.array([1.0, 1.0, 1.0])  # Maximum wind gust magnitude (m/s)

def generate_wind_gust():
    import numpy as np
    return np.random.uniform(-1, 1, size=3) * gust_amplitude

# Aerodynamic parameters
rho = 1.2682  # Air density (kg/m^3)
S_wing = 0.55  # Wing surface area (m^2)
b = 2.8956  # Wing span (m)
c = 0.18994  # Mean aerodynamic chord (m)
AR = 15.0  # Aspect ratio of the wing

# Aerodynamic coefficients
C_L_0 = 0.23
C_D_0 = 0.0424
C_m_0 = 0.0135
C_L_alpha = 5.61
C_D_alpha = 0.132
C_m_alpha = -2.74
C_L_q = 7.95
C_D_q = 0.0
C_m_q = -38.21
C_L_delta_e = 0.13
C_D_delta_e = 0.0135
C_m_delta_e = -0.99

C_Y_0 = 0.0
C_ell_0 = 0.0
C_n_0 = 0.0
C_Y_beta = -0.98
C_ell_beta = -0.13
C_n_beta = 0.073
C_Y_p = 0.0
C_ell_p = -0.51
C_n_p = 0.069
C_Y_r = 0.0
C_ell_r = 0.25
C_n_r = -0.095
C_Y_delta_a = 0.075
C_ell_delta_a = 0.17
C_n_delta_a = -0.011
C_Y_delta_r = 0.19
C_ell_delta_r = 0.0024
C_n_delta_r = -0.069

initial_state = [
    0.0,  # North position (m)
    0.0,  # East position (m)
    -100.0,  # Down position (m)
    25.0,  # u velocity (m/s)
    0.0,  # v velocity (m/s)
    0.0,  # w velocity (m/s)
    0.0,  # phi (roll angle, rad)
    0.0,  # theta (pitch angle, rad)
    0.0,  # psi (yaw angle, rad)
    0.0,  # p (roll rate, rad/s)
    0.0,  # q (pitch rate, rad/s)
    0.0   # r (yaw rate, rad/s)
]
