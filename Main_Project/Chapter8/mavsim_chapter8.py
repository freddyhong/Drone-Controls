"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
        2/24/2020 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
        3/11/2024 - RWB
"""
import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_sensors import MavDynamics
from models.wind_simulation import WindSimulation
from controllers.autopilot_lqr import Autopilot
from estimators.observer import Observer
from viewers.view_manager import ViewManager
import time

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
observer = Observer(SIM.ts_simulation)
viewers = ViewManager(mav=True, 
                      data=True,
                      video=False, video_name='chap8.mp4')

# autopilot commands
from message_types.msg_autopilot import MsgAutopilot
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency = 0.01)
h_command = Signals(dc_offset=100.0,
                    amplitude=20.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = Signals(dc_offset=np.radians(0.0),
                      amplitude=np.radians(45.0),
                      start_time=10.0,
                      frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time
end_time = 110

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.polynomial(sim_time)
    commands.course_command = chi_command.polynomial(sim_time)
    commands.altitude_command = h_command.polynomial(sim_time)

    # -------- autopilot -------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = observer.update(measurements)  # estimate states from measurements
    delta, commanded_state = autopilot.update(commands, estimated_state)
    #delta, commanded_state = autopilot.update(commands, mav.true_state)

    # -------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------- update viewer -------------
    viewers.update(
        sim_time,
        true_state=mav.true_state,  # true states
        estimated_state=estimated_state,  # estimated states        
        commanded_state=commanded_state,  # commanded states
        delta=delta, # inputs to MAV
    )
        
    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    #time.sleep(0.001)  # make the sim run slower

# close viewers
viewers.close(dataplot_name="ch8_data_plot", 
              sensorplot_name="ch8_sensor_plot")