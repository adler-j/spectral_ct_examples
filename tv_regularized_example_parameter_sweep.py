"""This formulation solves the model

    min_x ||Ax - b||_2^2 + ||grad(x)||_1

where A is the ray transform and grad is the gradient.
"""

import numpy as np
from util import load_fan_data
import odl

data, geometry = load_fan_data()

material = 'water'

space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

grad = odl.Gradient(space)

if material == 'water':
    b = ray_trafo.range.element(data[0])
elif material == 'bone':
    b = ray_trafo.range.element(data[1])

l2_sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(b)
l1 = odl.solvers.GroupL1Norm(grad.range)
f = odl.solvers.ZeroFunctional(space)

fbp_op = odl.tomo.fbp_op(ray_trafo,
                         filter_type='Hann', frequency_scaling=1.0)
x_init = fbp_op(b)

lam_values = np.linspace(20, 100, 21)

for lam in lam_values:
    if 0:
        precon = 100.0
        g = [l2_sq, (lam / precon) * l1]
        L = [ray_trafo, precon * grad]
        tau = 1
        sigma = [1.0 / odl.power_method_opnorm(ray_trafo)**2,
                 1.0 / odl.power_method_opnorm(precon * grad)**2]

        callback = odl.solvers.CallbackPrintIteration()

        callback = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(clim=[0.9, 1.1], step=10))

        x = x_init.copy()
        odl.solvers.douglas_rachford_pd(x, f, g, L, tau, sigma, niter=5000,
                                        callback=callback)
    else:
        op = odl.BroadcastOperator(ray_trafo, 100 * grad)
        g = odl.solvers.SeparableSum(l2_sq, 0.01 * lam * l1)

        op_norm = odl.power_method_opnorm(op)

        # Accelerataion parameter
        gamma = 0.4

        # Step size for the proximal operator for the primal variable x
        tau = 1.0 / op_norm

        # Step size for the proximal operator for the dual variable y
        sigma = 1.0 / (op_norm ** 2 * tau)

        # Reconstruct
        callback = odl.solvers.CallbackPrintIteration('lam={}'.format(lam))
        #callback &= odl.solvers.CallbackShow(step=10)

        # Use the FBP as initial guess
        x = x_init.copy()

        niter = 2000
        odl.solvers.chambolle_pock_solver(x, g, f, op, tau=tau, sigma=sigma,
                                          niter=niter, gamma=gamma, callback=callback)

    x.show('lam={}'.format(lam))

    import scipy.io as sio
    mdict = {'result': x.asarray(), 'lam': lam}
    sio.savemat('data/results/parameters/result_tv_{}_lam_{}'.format(material, lam), mdict)
