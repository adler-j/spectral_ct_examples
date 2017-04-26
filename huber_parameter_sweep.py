"""This formulation solves the model

    min_x ||Ax - b||_2^2 + ||grad(x)||_1

where A is the ray transform and grad is the gradient.
"""

import numpy as np
from util import load_fan_data
import odl

material = 'bone'
lam_values = np.linspace(0, 100, 51)

for lam in lam_values:
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

    callback = odl.solvers.CallbackPrintIteration('huber material {} lam {}'.format(material, lam))

    opnorm = odl.power_method_opnorm(ray_trafo)
    if lam > 0:
        reg_norm = odl.power_method_opnorm((lam * huber * grad).gradient)
    else:
        reg_norm = 0
    hessinv_estimate = odl.ScalingOperator(func.domain, 1 / (opnorm + reg_norm) ** 2)

    x = func.domain.zero()
    odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=1000, num_store=10,
                            callback=callback, hessinv_estimate=hessinv_estimate)

    import scipy.io as sio
    mdict = {'result': x.asarray(), 'lam': lam}
    sio.savemat('data/results/parameters/result_huber_{}_lam_{}'.format(material, lam), mdict)
