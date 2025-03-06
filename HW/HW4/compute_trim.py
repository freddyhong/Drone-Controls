import numpy as np
from scipy.optimize import minimize
import parameters as par

class MsgDelta:
    def __init__(self, elevator=-0.2, aileron=0.0, rudder=0.005, throttle=0.5):
        self.elevator = elevator 
        self.aileron = aileron  
        self.rudder = rudder  
        self.throttle = throttle 

    def print(self):
        print(f"Elevator: {self.elevator}, Aileron: {self.aileron}, Rudder: {self.rudder}, Throttle: {self.throttle}")


def compute_trim(mav, Va, gamma):
    state0 = par.initial_state
    delta0 = par.initial_delta
    x0 = np.concatenate((state0, delta0), axis=0)

    # have to do sth with this fkcing constrain
    cons = [
        {'type': 'eq',
        'fun': lambda x: np.array([
            x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # Velocity magnitude = Va
            x[4],  # v = 0 (no sideslip)
            x[6],  # no roll (phi = 0)
            #x[7],  # no pitch (theta = 0)
            x[8],  # no yaw (psi = 0)
            x[9],  # p = 0 (no roll rate)
            x[10], # q = 0 (no pitch rate)
            x[11], # r = 0 (no yaw rate)
        ])}
    ]
    
    

    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma), constraints=cons, options={'maxiter': 2000, 'ftol': 1e-8, 'disp': True, 'eps': 1e-8})


    # extract trim state and input
    trim_state = res.x[0:12].reshape((12, 1))
    trim_input = MsgDelta(
        elevator=res.x[12],
        aileron=res.x[13],
        rudder=res.x[14],
        throttle=res.x[15]
    )
    # print results
    trim_input.print()
    print('Trimmed State:\n', trim_state.T)
    print("Initial theta:", par.initial_state[7])
    print("Trimmed theta:", trim_state[7, 0])
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized (so aircraft has steady flight)

    state = x[:12]
    delta = MsgDelta(elevator=x[12], aileron=x[13], rudder=x[14], throttle=x[15])
    # Desired steady flight conditions
    desired_trim_state_dot = np.array([
        0, 0, -Va * np.sin(gamma),  # pn, pe, climb rate constant h
        0, 0, 0,  # Velocity derivatives
        0, 0, 0,  # Euler angles should remain steady
        0, 0, 0   # Angular rates (p, q, r)
    ])

    # set MAV state and update dynamics
    mav.state = state 
    mav._update_velocity_data(delta)
    forces, moments = mav.compute_forces_moments(delta, wind_enabled=False)  # no wind for trim for steady flight
    f = mav.equations_of_motion(state, forces, moments)
    
    # Compute cost function J (minimizing error in state derivatives)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[3:12])**2/10  # ignore first two position derivatives
    print(f"Current x: {x}")

    print("Final cost J:", J)

    return J

