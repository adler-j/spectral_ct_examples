"""This formulation solves the model

    min_x ||Ax - b||_2^2 + ||grad(x)||_2^2

where A is the ray transform and grad is the gradient.
"""

import odl
from util import load_data

data, geometry = load_data()

space = odl.uniform_discr([-129, -129], [129, 129], [600, 600])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

L = odl.Gradient(space)

op = ray_trafo.adjoint * ray_trafo + 20 * L.adjoint * L

x = ray_trafo.domain.zero()
rhs = ray_trafo.adjoint(data[1])
odl.solvers.conjugate_gradient(op, x, rhs, 100,
                               callback=odl.solvers.CallbackShow())
