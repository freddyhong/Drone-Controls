import numpy as np
from mav_model import MAV  
import parameters as par
from compute_trim import compute_trim
import matplotlib.pyplot as plt


mav_with_wind = MAV(
        ts=par.ts_simulation, 
        mass=par.mass, 
        Jx=par.Jx, 
        Jy=par.Jy, 
        Jz=par.Jz, 
        Jxz=par.Jxz, 
        initial_state=par.initial_state
    )

mav_without_wind = MAV(
        ts=par.ts_simulation, 
        mass=par.mass, 
        Jx=par.Jx, 
        Jy=par.Jy, 
        Jz=par.Jz, 
        Jxz=par.Jxz, 
        initial_state=par.initial_state
    )

    # trim conditions
Va = 25.# m/s
gamma = 0*np.pi/180.  # level flight

trim_state, trim_input = compute_trim(mav_with_wind, Va, gamma)
mav_with_wind.state = trim_state  # set the initial state of the mav to the trim state
mav_without_wind.state = trim_state
delta = lambda: None
delta = trim_input  # set input to constant constant trim input

sim_time = par.start_time

# Store positions for both cases
positions_with_wind = []
positions_without_wind = []
velocities_with_wind = []
velocities_without_wind = []
angular_velocities_with_wind = []
angular_velocities_without_wind = []
times = []


while sim_time < par.end_time:
    mav_with_wind.update(delta, wind_enabled=True)
    mav_without_wind.update(delta, wind_enabled=False)

    positions_with_wind.append(mav_with_wind.state[:3])
    positions_without_wind.append(mav_without_wind.state[:3])
    velocities_with_wind.append(mav_with_wind.state[3:6])
    velocities_without_wind.append(mav_without_wind.state[3:6])
    angular_velocities_with_wind.append(mav_with_wind.state[9:12])
    angular_velocities_without_wind.append(mav_without_wind.state[9:12])
    times.append(sim_time)
    sim_time += par.ts_simulation

positions_with_wind = np.array(positions_with_wind)
positions_without_wind = np.array(positions_without_wind)
velocities_with_wind = np.array(velocities_with_wind)
velocities_without_wind = np.array(velocities_without_wind)
angular_velocities_with_wind = np.array(angular_velocities_with_wind)
angular_velocities_without_wind = np.array(angular_velocities_without_wind)

# Plotting
# 3D Position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_with_wind[:, 0], positions_with_wind[:, 1], positions_with_wind[:, 2], label='With Trim')
ax.plot(positions_without_wind[:, 0], positions_without_wind[:, 1], positions_without_wind[:, 2], label='Without Wind', linestyle='dashed')
ax.set_xlabel('North')
ax.set_ylabel('East')
ax.set_zlabel('Altitude')
ax.legend()
plt.show()

# Velocities (u, v, w)
plt.figure()
plt.plot(times, velocities_with_wind[:, 0], label='u with Trim')
plt.plot(times, velocities_without_wind[:, 0], label='u without Wind', linestyle='dashed')

plt.plot(times, velocities_with_wind[:, 1], label='v with Trim')
plt.plot(times, velocities_without_wind[:, 1], label='v without Wind', linestyle='dashed')

plt.plot(times, velocities_with_wind[:, 2], label='w with Wind')
plt.plot(times, velocities_without_wind[:, 2], label='w without Wind', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Components (u, v, w) With and Without Wind')
plt.show()

# Angular Velocities (p, q, r)
plt.figure()
plt.plot(times, angular_velocities_with_wind[:, 0], label='p with Wind')
plt.plot(times, angular_velocities_without_wind[:, 0], label='p without Wind', linestyle='dashed')
plt.plot(times, angular_velocities_with_wind[:, 1], label='q with Wind')
plt.plot(times, angular_velocities_without_wind[:, 1], label='q without Wind', linestyle='dashed')
plt.plot(times, angular_velocities_with_wind[:, 2], label='r with Wind')
plt.plot(times, angular_velocities_without_wind[:, 2], label='r without Wind', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Angular Velocity')
plt.legend()
plt.title('Angular Velocity Components (p, q, r) With and Without Wind')
plt.show()

print('Trimmed State:\n', trim_state.T)
trim_input.print()
