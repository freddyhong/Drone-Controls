import sympy as sp

p, q, r = sp.symbols('p q r')  #
dp, dq, dr = sp.symbols('dp dq dr')  #

J1, J2, J3 = sp.symbols('J1 J2 J3')
J = sp.Matrix([[J1, 0, 0], [0, J2, 0], [0, 0, J3]])

omega = sp.Matrix([p, q, r])

omega_dot = sp.Matrix([dp, dq, dr])

J_omega_dot = J * omega_dot

cross_product = omega.cross(J * omega)

m_b = sp.Matrix(sp.symbols('m1^b m2^b m3^b'))

euler_equation = sp.Eq(J_omega_dot + cross_product, m_b)

sp.pprint(euler_equation)
