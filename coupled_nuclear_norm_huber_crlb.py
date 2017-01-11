"""This formulation solves the model

    min_x ||Ax - b||_W^2 + ||grad(x)||_*

where A is the ray transform, || . ||_W is the weighted l2 norm,
grad is the gradient and || . ||_* is the nuclear norm.
"""

import odl
import numpy as np
import scipy.linalg as spl
from util import cov_matrix, load_data, load_fan_data, inverse_sqrt_matrix

data, geometry, crlb = load_fan_data(return_crlb=True)

space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
A = odl.DiagonalOperator(ray_trafo, 2)

grad = odl.Gradient(space)
L = odl.DiagonalOperator(grad, 2)

# Compute covariance matrix and matrix power, this is slow.
mat_sqrt_inv = inverse_sqrt_matrix(crlb)

re = ray_trafo.range.element
W = odl.ProductSpaceOperator([[odl.MultiplyOperator(re(mat_sqrt_inv[0, 0])), odl.MultiplyOperator(re(mat_sqrt_inv[0, 1]))],
                              [odl.MultiplyOperator(re(mat_sqrt_inv[1, 0])), odl.MultiplyOperator(re(mat_sqrt_inv[1, 1]))]])

op = W * A

rhs = W(data)

data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)
regularizer = odl.solvers.NuclearNorm(L.range)

fbp_op = odl.tomo.fbp_op(ray_trafo,
                         filter_type='Hann', frequency_scaling=0.7)
x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])

huber = 0.003 * odl.solvers.MoreauEnvelope(regularizer, sigma=0.001)

func = data_discrepancy * op + huber * L

callback = (odl.solvers.CallbackShow() &
            odl.solvers.CallbackShow(clim=[0.9, 1.1]) &
            odl.solvers.CallbackPrintIteration())

opnorm = odl.power_method_opnorm(op)
hessinv_estimate = odl.ScalingOperator(func.domain, 1 / opnorm ** 2)

odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=400, num_store=50,
                        callback=callback, hessinv_estimate=hessinv_estimate)
