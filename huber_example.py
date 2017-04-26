"""Example of reconstruction with the Huber norm.

Here, the huber norm is implemented using the moreau envelope, for convenience.

The problem is solved using the BFGS quasi-newton method.
"""

import odl
from util import load_data, load_fan_data

material = 'water'
lam = 100
sigma = 0.03

data, geometry = load_fan_data()

space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Data discrepancy
if material == 'water':
    b = ray_trafo.range.element(data[0])
elif material == 'bone':
    b = ray_trafo.range.element(data[1])

l2 = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - b)

# Create huber norm
grad = odl.Gradient(space)
l1_norm = odl.solvers.GroupL1Norm(grad.range)
huber = odl.solvers.MoreauEnvelope(l1_norm, sigma=sigma)

func = l2 + lam * huber * grad

callback = (odl.solvers.CallbackShow(display_step=50) &
            odl.solvers.CallbackShow(display_step=50, clim=[0.9, 1.1]) &
            odl.solvers.CallbackPrintIteration())

opnorm = odl.power_method_opnorm(ray_trafo)
reg_norm = odl.power_method_opnorm((lam * huber * grad).gradient)
hessinv_estimate = odl.ScalingOperator(func.domain, 1 / (opnorm + reg_norm) ** 2)

x = func.domain.zero()
odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=1000, num_store=10,
                        callback=callback, hessinv_estimate=hessinv_estimate)

import scipy.io as sio
mdict = {'result': x.asarray(), 'lam': lam, 'sigma': sigma}
sio.savemat('result_huber_{}'.format(material), mdict)