import numpy as np
from scipy.optimize import minimize

class MsgDelta:
    def __init__(self, elevator=0.0, aileron=0.0, rudder=0.0, throttle=0.5):
        self.elevator = elevator 
        self.aileron = aileron  
        self.rudder = rudder  
        self.throttle = throttle 

    def print(self):
        print(f"Elevator: {self.elevator}, Aileron: {self.aileron}, Rudder: {self.rudder}, Throttle: {self.throttle}")

def compute_trim(mav, Va, gamma):

    # set the initial conditions of the optimization
    phi0 = 0.0  
    theta0 = gamma  
    psi0 = 0.0  
    
    state0 = np.array([
            [0],  # pn
            [0],  # pe
            [0],  # pd
            [Va], # u (forward velocity)
            [0],  # v (no side velocity)
            [0.], # w (no vertical velocity)
            [phi0],  
            [theta0],  
            [psi0],  
            [0],  # p 
            [0],  # q 
            [0]   # r 
    ])
    delta0 = np.array([
        [0],  # elevator
        [0],  # aileron
        [0],  # rudder
        [0.5] # throttle
    ])

    x0 = np.concatenate((state0, delta0), axis=0).flatten()

    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                 x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # Ensure velocity magnitude is Va
                 x[4],  # Force v=0 (no side slip)
                 x[9],  # p=0 (zero roll rate)
                 x[10], # q=0 (zero pitch rate)
                 x[11]  # r=0 (zero yaw rate)
             ])
    })

    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma), constraints=cons, options={'ftol': 1e-10, 'disp': True})
    
    # extract trim state and input and return
    trim_state = np.array([res.x[0:12]]).T
    trim_input = MsgDelta(
                elevator=res.x[12],
                aileron=res.x[13],
                rudder=res.x[14],
                throttle=res.x[15])
    
    # print results
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized (so aircraft has steady flight)
    
    state = x[:12]
    delta = MsgDelta(elevator=x[12], aileron=x[13], rudder=x[14], throttle=x[15])

    desired_trim_state_dot = np.array([
        [0], [0], [-Va*np.sin(gamma)], [0], [0], [0],  # Position and velocity derivatives
        [0], [0], [0],  # Euler angles should remain steady
        [0], [0], [0]  # Angular rates (p, q, r)
    ])
    
    # Set MAV state and update dynamics
    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    f = mav._derivatives(state, forces_moments)

    # Compute cost function (minimizing error in state derivatives)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[2:12])**2.0

    return J