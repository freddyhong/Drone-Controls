import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from models.mav_dynamics import MavDynamics
from controllers.autopilot import Autopilot
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

def test_autopilot():
    # Initialize UAV and autopilot
    mav = MavDynamics(Ts=0.01)
    autopilot = Autopilot(ts_control=0.01)
    
    # Data storage
    time = []
    states = []
    controls = []
    commands = []
    
    # Test scenario (climb + turn)
    def run_test():
        cmd = MsgState()
        for t in np.arange(0, 30, 0.01):
            # Update commands
            if t < 10:
                cmd.altitude_command = 50.0  # Climb to 50m
                cmd.course_command = 0.0     # Maintain heading
                cmd.airspeed_command = 25.0  # Hold airspeed
            else:
                cmd.course_command = np.radians(90)  # Turn to 90°
            
            # Get state and update autopilot
            state = mav._state
            delta, cmd_state = autopilot.update(cmd, MsgState.from_array(state))

            print("\nCommanded State Verification:")
            print(f"Alt cmd: {cmd.altitude_command} → cmd_state: {cmd_state.altitude}")
            print(f"Course cmd: {np.degrees(cmd.course_command):.1f}deg → cmd_state: {np.degrees(cmd_state.chi):.1f}deg")
            print(f"Computed phi cmd: {np.degrees(cmd_state.phi):.1f}deg")
            print(f"Computed theta cmd: {np.degrees(cmd_state.theta):.1f}deg")
                    
            # Store data
            time.append(t)
            states.append(state.flatten())
            controls.append([delta.elevator, delta.aileron, 
                           delta.rudder, delta.throttle])
            commands.append([cmd.altitude_command, 
                            np.degrees(cmd.course_command),
                            cmd.airspeed_command])
            
            # Update UAV dynamics
            mav.update(delta, wind=np.zeros((6,1)))  # No wind for basic test

    run_test()

    # Convert to numpy arrays
    time = np.array(time)
    states = np.array(states)
    controls = np.array(controls)
    commands = np.array(commands)

    # Plotting
    plt.figure(figsize=(15, 12))

    # Position responses
    plt.subplot(4, 2, 1)
    plt.plot(time, states[:,2], label='Altitude')
    plt.plot(time, commands[:,0], 'r--', label='Command')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(time, np.degrees(states[:,8]), label='Heading')
    plt.plot(time, commands[:,1], 'r--', label='Command')
    plt.ylabel('Heading (deg)')
    plt.legend()
    plt.grid(True)

    # Attitude responses
    plt.subplot(4, 2, 3)
    plt.plot(time, np.degrees(states[:,6]), label='Roll')
    plt.ylabel('Roll (deg)')
    plt.grid(True)

    plt.subplot(4, 2, 4)
    plt.plot(time, np.degrees(states[:,7]), label='Pitch')
    plt.ylabel('Pitch (deg)')
    plt.grid(True)

    # Control surfaces
    plt.subplot(4, 2, 5)
    plt.plot(time, controls[:,0], label='Elevator')
    plt.ylim(-1.1, 1.1)
    plt.ylabel('Elevator')
    plt.grid(True)

    plt.subplot(4, 2, 6)
    plt.plot(time, controls[:,1], label='Aileron')
    plt.ylim(-1.1, 1.1)
    plt.ylabel('Aileron')
    plt.grid(True)

    plt.subplot(4, 2, 7)
    plt.plot(time, controls[:,2], label='Rudder')
    plt.ylim(-1.1, 1.1)
    plt.ylabel('Rudder')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.subplot(4, 2, 8)
    plt.plot(time, controls[:,3], label='Throttle')
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Throttle')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Check for saturation
    print("\nSaturation Analysis:")
    for i, surface in enumerate(['Elevator', 'Aileron', 'Rudder', 'Throttle']):
        saturated = np.where(np.abs(controls[:,i]) >= 0.95)[0]
        if len(saturated) > 0:
            print(f"{surface} saturated at {len(saturated)} points")
        else:
            print(f"{surface} never reached saturation")

if __name__ == "__main__":
    test_autopilot()