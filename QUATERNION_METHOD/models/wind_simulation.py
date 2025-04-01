"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in UAV book)
"""
import sys
import os

# Add the parent directory of 'models' to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools.transfer_function import TransferFunction


class WindSimulation:
    def __init__(self, Ts, gust_flag=True, steady_state=np.array([[0., 0., 0.]]).T):
        # Steady-state wind in inertial frame
        self._steady_state = steady_state
        self._Ts = Ts
        self._Va = 25.0 
        self.gust_flag = gust_flag

        # Dryden gust model parameters (low-altitude light turbulence typical)
        Lu, Lv, Lw = 200.0, 200.0, 50.0  
        sigma_u, sigma_v, sigma_w = 1.06, 1.06, 0.7

        # Dryden transfer functions for u, v, w gusts (from Section 4.4 UAV Book)
        self.u_w = TransferFunction(num=np.array([[sigma_u * np.sqrt(2 * self._Va / (np.pi * Lu))]]), 
                                    den=np.array([[1.0, self._Va / Lu]]), Ts=Ts)

        self.v_w = TransferFunction(num=np.array([[sigma_v * np.sqrt(3 * self._Va / (np.pi * Lv)), 
                                                sigma_v * np.sqrt(3 * self._Va / (np.pi * Lv)) * self._Va / (np.sqrt(3) * Lv)]]),
                                    den=np.array([[1.0, 2 * self._Va / Lv, (self._Va / Lv)**2]]), Ts=Ts)

        self.w_w = TransferFunction(num=np.array([[sigma_w * np.sqrt(3 * self._Va / (np.pi * Lw)), 
                                                sigma_w * np.sqrt(3 * self._Va / (np.pi * Lw)) * self._Va / (np.sqrt(3) * Lw)]]),
                                    den=np.array([[1.0, 2 * self._Va / Lw, (self._Va / Lw)**2]]), Ts=Ts)

    def update(self): # first 3 elements are steady state wind, last 3 are gusts
        if self.gust_flag:
            gust = np.array([
                [self.u_w.update(np.random.randn())],
                [self.v_w.update(np.random.randn())],
                [self.w_w.update(np.random.randn())]
            ])
        else:
            gust = np.zeros((3, 1))

        return np.concatenate((self._steady_state, gust), axis=0)
