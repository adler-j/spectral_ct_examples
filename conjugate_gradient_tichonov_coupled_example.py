"""This formulation solves the model

    min_x ||Ax - b||_W^2 + ||grad(x)||_2^2

where A is the vectorial ray transform, W is the correlation matrix and grad
is the gradient.
"""

import odl
import numpy as np
import scipy.linalg as spl
from util import cov_matrix, load_data, load_fan_data

data, geometry = load_fan_data()

space = odl.uniform_discr([-150, -150], [150, 150], [200, 200])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
A = odl.DiagonalOperator(ray_trafo, ray_trafo)

# Compute covariance matrix and matrix power
cov_mat = cov_matrix(data)
w_mat = spl.fractional_matrix_power(cov_mat, -0.5)

I = odl.IdentityOperator(ray_trafo.range)
W = odl.ProductSpaceOperator(np.multiply(w_mat, I))
A_corr = W * A

grad = odl.Gradient(space)
L = odl.DiagonalOperator(grad, grad)

op = A_corr.adjoint * A_corr + 10 * L.adjoint * L

fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.7)

callback = (odl.solvers.CallbackShow(display_step=10) &
            odl.solvers.CallbackPrintIteration())

x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])
x.show('filtered back-projection')
rhs = A_corr.adjoint(W(data))
odl.solvers.conjugate_gradient(op, x, rhs, 100,
                               callback=callback)
