import sympy as sp

# Define symbolic variables
J1, J2, J3 = sp.symbols('J1 J2 J3')
p, r = sp.symbols('p r')
m1, m2, m3 = sp.symbols('m1 m2 m3')

# Define inertia matrix
J = sp.diag(J1, J2, J3)

# Define angular velocity vector (pitch velocity q = 0)
omega = sp.Matrix([p, 0, r])

# Define external torque vector
torque = sp.Matrix([m1, m2, m3])

# Compute angular momentum
h = J * omega

# Compute Euler's equations: J * d(omega)/dt + omega x (J * omega) = torque
omega_dot = J.inv() * (torque - omega.cross(h))

# Display the equations
print("Angular acceleration equations (dp/dt, dq/dt, dr/dt):")
sp.pprint(omega_dot)

