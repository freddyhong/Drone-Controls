import numpy as np
import parameters as P
from integrators import get_integrator
import matplotlib.pyplot as plt
from pid import PIDControl


class Controller:
    def __init__(self):
            self.pid = PIDControl(
            kp=P.kp, ki=P.ki, kd=P.kd, 
            limit=P.umax, sigma=P.sigma, Ts=P.Ts
        )
        
    def update(self, r, y):
        return self.pid.PID(r, y)
    
class System:
    def dynamics(self, t, y, u):
        # define dy/dt
        return (-1 /  P.tau) * y + (P.K/P.tau) * u  

    def __init__(self):
        self.y = 0 
        self.integrator = get_integrator(P.Ts, self.dynamics,  integrator="RK4") 
    
   
    def update(self, u):
        self.y = self.integrator.step(0, self.y, u)  
        return self.y

# Init system and feedback controller
system = System()
controller = Controller()

# Simulate step response
t_history = [0]
y_history = [0]
u_history = [0]

r = 1
y = 0
t = 0
for i in range(P.nsteps):
    u = controller.update(r, y) 
    y = system.update(u) 
    t += P.Ts

    t_history.append(t)
    y_history.append(y)
    u_history.append(u)

# Plot Response y due to Step Change in r
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(t_history, y_history, label="System Output (y)")
plt.axhline(r, color='r', linestyle='--', label="Reference (r)")
plt.xlabel("Time (s)")
plt.ylabel("Output y")
plt.title("Step Response of Controlled System")
plt.legend()
plt.grid()

# Plot Actuation Signal (Control Effort)
plt.subplot(2, 1, 2)
plt.plot(t_history, u_history, label="Control Signal (u)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Control Input u")
plt.title("Actuation Signal")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
