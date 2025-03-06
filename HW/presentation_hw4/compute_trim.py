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
        print(f"Elevator: {self.elevator}, Aileron: {self.aileron}, "
              f"Rudder: {self.rudder}, Throttle: {self.throttle}")


def compute_trim(mav, Va, gamma):
    """
    Computes a trim condition for a given MAV, airspeed Va, and flight path angle gamma.
    """
    # Initial guesses for state (12) + control (4)
    state0 = par.initial_state
    delta0 = par.initial_delta
    x0 = np.concatenate((state0, delta0), axis=0)

    # Define constraints: zero sideslip, zero roll, zero yaw, p=q=r=0, speed=Va
    cons = [
        {
            'type': 'eq',
            'fun': lambda x: np.array([
                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # speed^2 = Va^2
                x[4],  # v=0  (no sideslip)
                x[6],  # phi=0 (no roll)
                x[8],  # psi=0 (no yaw)
                x[9],  # p=0
                x[10], # q=0
                x[11], # r=0
            ])
        }
    ]

    # Solve minimization
    res = minimize(
        trim_objective_fun,
        x0,
        method='SLSQP',
        args=(mav, Va, gamma),
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-8, 'disp': True, 'eps': 1e-8}
    )

    # Extract trim state and inputs
    trim_state = res.x[0:12].reshape((12, 1))
    trim_input = MsgDelta(
        elevator=res.x[12],
        aileron=res.x[13],
        rudder=res.x[14],
        throttle=res.x[15]
    )

    # Print results
    trim_input.print()
    print('Trimmed State:\n', trim_state.T)
    print("Initial theta:", par.initial_state[7])
    print("Trimmed theta:", trim_state[7, 0])

    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    """
    Objective function to minimize the difference between actual state derivatives
    and desired steady-flight conditions (for speed Va, flight path angle gamma).
    """
    # Parse state and controls from x
    state_vec = x[:12]
    delta_vals = x[12:]
    delta = MsgDelta(
        elevator=delta_vals[0],
        aileron=delta_vals[1],
        rudder=delta_vals[2],
        throttle=delta_vals[3]
    )

    # Desired derivative: consistent with flight path angle gamma
    desired_trim_state_dot = np.array([
        0, 0, -Va * np.sin(gamma),  # p_dot_n, p_dot_e, p_dot_d
        0, 0, 0,                    # u_dot, v_dot, w_dot
        0, 0, 0,                    # phi_dot, theta_dot, psi_dot
        0, 0, 0                     # p_dot, q_dot, r_dot
    ])

    # Update MAV state and compute the actual derivative
    mav.state = state_vec
    mav._update_velocity_data(delta)  # re-calc Va, alpha, beta, etc.
    forces, moments = mav.compute_forces_moments(delta)  # No wind_enabled param
    f = mav.equations_of_motion(state_vec, forces, moments)

    diff = desired_trim_state_dot - f
    cost = np.linalg.norm(diff[2:12])**2

    print(f"Current x: {x}")
    print("Final cost:", cost)
    return cost
