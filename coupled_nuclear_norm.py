"""This formulation solves the model

    min_x ||Ax - b||_W^2 + ||grad(x)||_*

where A is the ray transform, || . ||_W is the weighted l2 norm,
grad is the gradient and || . ||_* is the nuclear norm.
"""

import odl
import numpy as np
import scipy.linalg as spl
from util import cov_matrix, load_data, load_fan_data

data, geometry = load_fan_data()

space = odl.uniform_discr([-129, -129], [129, 129], [200, 200])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
A = odl.DiagonalOperator(ray_trafo, 2)

grad = odl.Gradient(space)
L = odl.DiagonalOperator(grad, 2)

# Compute covariance matrix and matrix power
I = odl.IdentityOperator(ray_trafo.range)

cov_mat = cov_matrix(data)
w_mat = spl.fractional_matrix_power(cov_mat, -0.5)
# raise Exception
W = odl.ProductSpaceOperator(np.multiply(w_mat, I))
op = W * A

rhs = W(data)

data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)
regularizer = 0.01 * odl.solvers.NuclearNorm(L.range)

fbp_op = odl.tomo.fbp_op(ray_trafo)
x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])

f = odl.solvers.IndicatorBox(A.domain, 0, 1)
g = [data_discrepancy, regularizer]
lin_ops = [op, L]
tau = 1.0
sigma = [1.0/odl.power_method_opnorm(op)**2,
         1.0/odl.power_method_opnorm(grad)**2]
niter = 100

callback = (odl.solvers.CallbackShow(step=1) &
            odl.solvers.CallbackPrintIteration())

odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau, sigma, niter,
                                callback=callback)
