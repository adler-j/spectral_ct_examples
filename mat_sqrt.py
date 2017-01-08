from __future__ import division
from sympy import *

x, y, z, a, b, c = symbols('x y z a b c', real=True)

solution = solve_poly_system([x*x + y*y - a, y*y + z*z - c, x*y + y*z - b],
                             x, y, z)
