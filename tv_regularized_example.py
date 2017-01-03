"""This formulation solves the model

    min_x ||Ax - b||_2^2 + ||grad(x)||_1

where A is the ray transform and grad is the gradient.
"""

from util import load_data
import odl

data = load_data()

space = odl.uniform_discr([-150, -150], [150, 150], [600, 600])

geometry = odl.tomo.parallel_beam_geometry(space,
                                           angles=data.shape[1],
                                           det_shape=data.shape[2])
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

grad = odl.Gradient(space)

b = ray_trafo.range.element(data[1])

l2_sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(b)
l1 = odl.solvers.GroupL1Norm(grad.range)
f = odl.solvers.IndicatorNonnegativity(space)

g = [l2_sq, l1]
L = [ray_trafo, grad]
tau = 1.0
sigma = [1 / odl.power_method_opnorm(ray_trafo)**2,
         1 / odl.power_method_opnorm(grad)**2]

x = space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, L, tau, sigma, niter=1000,
                                callback=odl.solvers.CallbackShow(display_step=20))
