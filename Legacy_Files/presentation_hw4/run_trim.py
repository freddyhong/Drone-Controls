import numpy as np
from mav_model import MAV_wind, MAV_nowind
import parameters as par
from compute_trim import compute_trim
import matplotlib.pyplot as plt

# Initialize MAV with initial state
mav_with_wind = MAV_wind(
    ts=par.ts_simulation, 
    mass=par.mass, 
    Jx=par.Jx, 
    Jy=par.Jy, 
    Jz=par.Jz, 
    Jxz=par.Jxz, 
    initial_state=par.initial_state
)

mav_without_wind = MAV_nowind(
    ts=par.ts_simulation, 
    mass=par.mass, 
    Jx=par.Jx, 
    Jy=par.Jy, 
    Jz=par.Jz, 
    Jxz=par.Jxz, 
    initial_state=par.initial_state
)

mav_trim_no_wind = MAV_nowind(
    ts=par.ts_simulation, 
    mass=par.mass, 
    Jx=par.Jx, 
    Jy=par.Jy, 
    Jz=par.Jz, 
    Jxz=par.Jxz, 
    initial_state=par.initial_state
)

# Trim conditions
Va = 25.0  # m/s
gamma = 0 * np.pi / 180.0  # level flight

# Compute trim state and input
trim_state, trim_input = compute_trim(mav_with_wind, Va, gamma)

# Set the initial state of the MAV to the trim state for the wind case
mav_with_wind.state = trim_state
mav_trim_no_wind.state = trim_state

# Define a lambda function for delta
delta = lambda: None
delta = trim_input  # set input to constant trim input

# Simulation parameters
sim_time = par.start_time

# Store positions and velocities for all cases
positions_with_wind = []
positions_without_wind = []
positions_trim_no_wind = []
velocities_with_wind = []
velocities_without_wind = []
velocities_trim_no_wind = []
angular_velocities_with_wind = []
angular_velocities_without_wind = []
angular_velocities_trim_no_wind = []
times = []

# Simulation loop
while sim_time < par.end_time:
    mav_with_wind.update(delta)
    mav_without_wind.update(delta)
    mav_trim_no_wind.update(delta)

    positions_with_wind.append(mav_with_wind.state[:3])
    positions_without_wind.append(mav_without_wind.state[:3])
    positions_trim_no_wind.append(mav_trim_no_wind.state[:3])
    velocities_with_wind.append(mav_with_wind.state[3:6])
    velocities_without_wind.append(mav_without_wind.state[3:6])
    velocities_trim_no_wind.append(mav_trim_no_wind.state[3:6])
    angular_velocities_with_wind.append(mav_with_wind.state[9:12])
    angular_velocities_without_wind.append(mav_without_wind.state[9:12])
    angular_velocities_trim_no_wind.append(mav_trim_no_wind.state[9:12])
    times.append(sim_time)
    sim_time += par.ts_simulation

positions_with_wind = np.array(positions_with_wind)
positions_without_wind = np.array(positions_without_wind)
positions_trim_no_wind = np.array(positions_trim_no_wind)
velocities_with_wind = np.array(velocities_with_wind)
velocities_without_wind = np.array(velocities_without_wind)
velocities_trim_no_wind = np.array(velocities_trim_no_wind)
angular_velocities_with_wind = np.array(angular_velocities_with_wind)
angular_velocities_without_wind = np.array(angular_velocities_without_wind)
angular_velocities_trim_no_wind = np.array(angular_velocities_trim_no_wind)

# Plotting
# 3D Position
# 3D Position Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the three flight scenarios
ax.plot(positions_with_wind[:, 0], positions_with_wind[:, 1], positions_with_wind[:, 2], label='With Trim and Wind')
ax.plot(positions_without_wind[:, 0], positions_without_wind[:, 1], positions_without_wind[:, 2], label='Without Wind', linestyle='dashed')
#ax.plot(positions_trim_no_wind[:, 0], positions_trim_no_wind[:, 1], positions_trim_no_wind[:, 2], label='With Trim and No Wind', linestyle='dotted')
ax.set_title('3D Plot with and without wind')

# Set axis labels
ax.set_xlabel('North')
ax.set_ylabel('East')
ax.set_zlabel('Altitude')

# Check if East values are too small
east_min = min(positions_with_wind[:, 1].min(), positions_without_wind[:, 1].min(), positions_trim_no_wind[:, 1].min())
east_max = max(positions_with_wind[:, 1].max(), positions_without_wind[:, 1].max(), positions_trim_no_wind[:, 1].max())

if abs(east_max) < 1e-2 and abs(east_min) < 1e-2:  # If all values are near zero
    ax.set_ylim(-10, 10)  # Force reasonable limits
else:
    buffer = 0.1 * (east_max - east_min)  # Add a buffer for visualization
    ax.set_ylim(east_min - buffer, east_max + buffer)

# Add legend and show plot
ax.legend()
plt.show()


# Velocities (u, v, w)
plt.figure()
plt.plot(times, velocities_with_wind[:, 0], label='u with Trim and Wind')
plt.plot(times, velocities_without_wind[:, 0], label='u without Wind', linestyle='dashed')
#plt.plot(times, velocities_trim_no_wind[:, 0], label='u with Trim and No Wind', linestyle='dotted')

plt.plot(times, velocities_with_wind[:, 1], label='v with Trim and Wind')
plt.plot(times, velocities_without_wind[:, 1], label='v without Wind', linestyle='dashed')
#plt.plot(times, velocities_trim_no_wind[:, 1], label='v with Trim and No Wind', linestyle='dotted')

plt.plot(times, velocities_with_wind[:, 2], label='w with Trim and Wind')
plt.plot(times, velocities_without_wind[:, 2], label='w without Wind', linestyle='dashed')
#plt.plot(times, velocities_trim_no_wind[:, 2], label='w with Trim and No Wind', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Components (u, v, w) With and Without Wind')
plt.show()

# Angular Velocities (p, q, r)
plt.figure()
plt.plot(times, angular_velocities_with_wind[:, 0], label='p with Trim and Wind')
plt.plot(times, angular_velocities_without_wind[:, 0], label='p without Wind and Trim', linestyle='dashed')
#plt.plot(times, angular_velocities_trim_no_wind[:, 0], label='p with Trim and No Wind', linestyle='dotted')

plt.plot(times, angular_velocities_with_wind[:, 1], label='q with Trim and Wind')
plt.plot(times, angular_velocities_without_wind[:, 1], label='q without Wind', linestyle='dashed')
#plt.plot(times, angular_velocities_trim_no_wind[:, 1], label='q with Trim and No Wind', linestyle='dotted')

plt.plot(times, angular_velocities_with_wind[:, 2], label='r with Trim and Wind')
plt.plot(times, angular_velocities_without_wind[:, 2], label='r without Wind', linestyle='dashed')
#plt.plot(times, angular_velocities_trim_no_wind[:, 2], label='r with Trim and No Wind', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.legend()
plt.title('Angular Velocity Components (p, q, r) With and Without Wind')
plt.show()

print('Trimmed State:\n', trim_state.T)
print('Trim Input:\n', trim_input)