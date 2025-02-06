import numpy as np
import matplotlib.pyplot as plt
import integrators as intg
import control as ct


m = 1 
b = 0.25 
k = 1


# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input

def f(t, x, u):
    x1, x2 = x
    dx1 = x2
    dx2 = (-k * x1 - b * x2) / m
    return np.array([dx1, dx2])

t = 0
x0 = np.array([1, 0])
x = np.array([1, 0])
x2 = np.array([1, 0])    
x3 = np.array([1, 0])
u = 0
dt = 0.1  
n = 500  
T = np.linspace(0, 50, 500)

wd = np.sqrt( k/m - (b/2*m)**2 )
c1 = x[0]
c2 = x[1]/wd + b/(2*m*wd)*x[0]
phi = np.arctan(c2/c1)
A = np.sqrt(c1**2 + c2**2)
sol = A*np.exp(-b*T/(2*m))*np.cos(wd*T - phi)
sol_dot = np.gradient(sol, T)


integrator = intg.Euler(dt, f)
integrator2 = intg.Heun(dt, f)
integrator3 = intg.RungeKutta(dt, f)

t_history = [t]
x_history = [x]
x_history2 = [x]
x_history3 = [x]

for i in range(n):
    x = integrator.step(t, x, u)
    x2 = integrator2.step(t, x2, u)    
    t = (i + 1) * dt 
    t_history.append(t)
    x_history.append(x)
    x_history2.append(x2)
    x_history3.append(x3)

x_history = np.array(x_history)
x_history2 = np.array(x_history2)
x_history3 = np.array(x_history3)
t_history = np.array(t_history)


plt.figure(figsize=(10, 5))
plt.plot(t_history, x_history[:, 0], label="Position (x) Euler")
plt.plot(t_history, x_history2[:, 0], label="Position (x) Heun")
plt.plot(t_history, x_history3[:, 0], label="Position (x) Runge Kutta")
plt.plot(T, sol, label="Position (x) Analytical", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("State Variables")
plt.title("Mass-Spring-Damper System Simulation Position")
plt.legend()
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(t_history, x_history[:, 1], label="Velocity (dx/dt) Euler")
plt.plot(t_history, x_history2[:, 1], label="Velocity (dx/dt) Heun")
plt.plot(t_history, x_history3[:, 1], label="Velocity (dx/dt) Runge Kutta")
plt.plot(T, sol_dot, label="Velocity (dx/dt) Analytical", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("State Variables")
plt.title("Mass-Spring-Damper System Simulation Velocity")
plt.legend()
plt.grid()

num = [1]  
den = [m, b, k]  
system = ct.TransferFunction(num, den)
T, response = ct.step_response(system, T)

plt.figure(figsize=(10, 5))
plt.plot(T, response, label="Step Response", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (x)")
plt.title("Step Response of Mass-Spring-Damper System")
plt.legend()
plt.grid()
plt.show()