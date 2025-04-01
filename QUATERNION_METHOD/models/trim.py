"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""

import sys
import os

# Add the parent directory of 'models' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion
from message_types.msg_delta import MsgDelta
import time

def compute_trim(mav, Va, gamma, wind=np.zeros((6, 1))):
    # define initial state and input

    # set the initial conditions of the optimization
    e0 = euler_to_quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [0],  # pd
                   [Va],  # u
                   [0.], # v
                   [0.], # w
                   [1],  # e0
                   [0],  # e1
                   [0],  # e2
                   [0],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    delta0 = np.array([[-0.2],  # elevator
                       [0],  # aileron
                       [0.005],  # rudder
                       [0.5]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma, wind),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)

    from tools.rotations import quaternion_to_euler
    # Extract quaternion from trim_state
    q_trim = trim_state[6:10].flatten()
    # Convert to Euler angles (phi, theta, psi)
    phi, theta, psi = quaternion_to_euler(q_trim)
    print(f"Trim Euler angles (rad): phi={phi:.6f}, theta={theta:.6f}, psi={psi:.6f}")
    print(f"Trim Euler angles (deg): phi={np.degrees(phi):.2f}, theta={np.degrees(theta):.2f}, psi={np.degrees(psi):.2f}")

    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma, wind):
    # objective function to be minimized
    state = x[0:13]
    delta = MsgDelta(elevator=x.item(13),
                    aileron=x.item(14),
                    rudder=x.item(15),
                    throttle=x.item(16))

    # Desired derivative: consistent with flight path angle gamma
    desired_trim_state_dot = np.array([
        0, 0, -Va * np.sin(gamma),  # p_dot_n, p_dot_e, p_dot_d
        0, 0, 0,                    # u_dot, v_dot, w_dot
        0, 0, 0,                    # phi_dot, theta_dot, psi_dot
        0, 0, 0                     # p_dot, q_dot, r_dot
    ])

    # Update MAV state
    mav.state = state
    mav._update_velocity_data(wind)
    forces_moments = mav._forces_moments(delta)
    f = mav._f(state, forces_moments)

    diff = desired_trim_state_dot - f
    J = np.linalg.norm(diff[2:12])**2 

    print(f"Current x: {x}")
    print("Final cost:", J)
    return J