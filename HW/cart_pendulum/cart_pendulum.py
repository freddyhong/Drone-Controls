import numpy as np
import control as ct
from numpy.linalg import matrix_rank

import matplotlib.pyplot as plt
from scipy import signal

# Parameters
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

# State space
den = I*(M+m)+M*m*l**2
A = np.array([
    [0,      1,              0,           0],
    [0, -(I+m*l**2)*b/den,  (m**2*g*l**2)/den,  0],
    [0,      0,              0,           1],
    [0, -(m*l*b)/den,       m*g*l*(M+m)/den,  0]
    ])
B = np.array([
    [0],
    [(I+m*l**2)/den],
    [0],
    [m*l/den]
    ])
C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
    ])
D = np.array([
    [0],
    [0]
    ])

print(A)

sys = ct.ss(A, B, C, D)
poles = ct.poles(sys)
print('poles:', poles)

# controllability
Co = ct.ctrb(A, B)
nc = matrix_rank(Co)
print('Controllability matrix:', Co)
print('Controllability rank:', nc)  #system is controllable rank = 4

#LQR
Q = np.array([
    [1, 0, 0, 0],
    [0, 0.1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0.1]
])
R = np.array([
    [1]
])

K, _, _ = ct.lqr(A, B, Q, R)
print('LQR gain:', K)

# closed loop system linearized around pendulum up position
sys_cl = ct.ss(A - B @ K, B, C, D)
print('closed loop system:', sys_cl)

poles_cl = np.linalg.eigvals(sys_cl.A)
print('closed loop poles:', poles_cl)

# plot step response of closed loop system
t = np.linspace(0, 10, 1000)
t, y = ct.step_response(sys_cl, T=t)

plt.figure()
plt.plot(t, y[0, 0, :], label='cart position')   # y[0] is cart position
plt.plot(t, y[1, 0, :], label='pendulum angle')  # y[1] is pendulum angle
plt.title('Step response of closed loop system')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.legend()
plt.grid()
plt.show()