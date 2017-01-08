"""Example of reconstruction with the Huber norm.

Here, the huber norm is implemented using the moreau envelope, for convenience.

The problem is solved using the BFGS quasi-newton method.
"""

import odl
from util import load_data, load_fan_data

data, geometry = load_fan_data()

space = odl.uniform_discr([-150, -150], [150, 150], [600, 600])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Data discrepancy
b = ray_trafo.range.element(data[1])
l2 = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - b)

# Create huber norm
grad = odl.Gradient(space)
l1_norm = odl.solvers.GroupL1Norm(grad.range)
huber = odl.solvers.MoreauEnvelope(l1_norm, sigma=0.1)

func = l2 + 3 * huber * grad

callback = (odl.solvers.CallbackShow() &
            odl.solvers.CallbackPrintIteration())

opnorm = odl.power_method_opnorm(ray_trafo)
hessinv_estimate = odl.ScalingOperator(func.domain, 1 / opnorm ** 2)

x = func.domain.zero()
odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=50, num_store=5,
                        callback=callback, hessinv_estimate=hessinv_estimate)
