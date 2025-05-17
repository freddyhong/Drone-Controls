"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion
from tools.rotations import quaternion_to_euler
from message_types.msg_delta import MsgDelta

def compute_trim(mav, Va, gamma, wind=np.zeros((6, 1))):
    # set the initial state of the optimization
    quat = euler_to_quaternion(0., gamma, 0.).flatten()
    state0 = np.array([[0.],  # pn
                   [0.],  # pe
                   [0.],  # pd
                   [Va],  # u
                   [0.],  # v
                   [0.],  # w
                   [quat[0]],  # e0
                   [quat[1]],  # e1
                   [quat[2]],  # e2
                   [quat[3]],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    # initial input guess
    delta0 = np.array([[0],  # elevator
                       [0.02],  # aileron
                       [0.005],  # rudder
                       [0.5]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    
    # constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                (x[3]-wind[0,0])**2 + (x[4]-wind[1,0])**2 + (x[5]-wind[2,0])**2 - Va**2, # Wind-relative velocity magnitude
                                x[4] - wind[1,0],   # v=0  no sideslip (wind relative)
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1., # unit quaternion norm
                                x[7],  # e1=0 no roll
                                x[9],  # e3=0 no yaw
                                x[10],  # p=0 no angular rates
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*(x[3]-wind[0,0]), 2*(x[4]-wind[1,0]), 2*(x[5]-wind[2,0]), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
   
    # solve the minimization to find the trim states and inputs
    res = minimize(trim_objective_fun, x0, method='SLSQP', 
                   args=(mav, Va, gamma, wind),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    
    # extract trim state and inputs
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    # print results
    final_cost = trim_objective_fun(res.x, mav, Va, gamma, wind)
    print(f'\nFinal cost: {final_cost:.6f}')

    trim_input.print()
    print('trim_state=', trim_state.T)

    q_trim = trim_state[6:10].flatten()
    phi, theta, psi = quaternion_to_euler(q_trim)   # convert to euler angles
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
                    0., 0., -Va * np.sin(gamma),  # p_dot_n, p_dot_e, p_dot_d
                    0., 0., 0.,                   # body accelerations
                    0., 0., 0.,                    # quaternion rates
                    0., 0., 0.                     # angular accelerations
    ])

    # Update MAV state
    mav.state = state
    mav._update_velocity_data(wind)
    # print(f"Computed Va: {mav._Va:.2f} m/s")
    forces_moments = mav._forces_moments(delta)
    actual = mav._f(state, forces_moments)

    diff = desired_trim_state_dot - actual
    J = np.linalg.norm(diff[2:13])**2

    return J