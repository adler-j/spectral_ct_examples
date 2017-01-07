"""Example of reconstruction with the Huber norm.

Here, the huber norm is implemented using the moreau envelope, for convenience.

The problem is solved using the BFGS quasi-newton method.
"""

import odl
from util import load_data

data, geometry = load_data()

space = odl.uniform_discr([-150, -150], [150, 150], [600, 600])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Data discrepancy
b = ray_trafo.range.element(data[1])
l2 = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - b)

# Create huber norm
grad = odl.Gradient(space)
l1_norm = odl.solvers.GroupL1Norm(grad.range)
huber = odl.solvers.MoreauEnvelope(l1_norm, sigma=0.1)

func = l2 + 10 * huber * grad

callback = odl.solvers.CallbackShow()
x = func.domain.zero()
odl.solvers.bfgs_method(func, x, line_search=0.0005, maxiter=50, num_store=5,
                        callback=callback)
