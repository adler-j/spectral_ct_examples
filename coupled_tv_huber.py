"""This formulation solves the model

    min_x ||Ax - b||_W^2 + ||grad(x)||_*

where A is the ray transform, || . ||_W is the weighted l2 norm,
grad is the gradient and || . ||_* is the nuclear norm.
"""

import odl
import numpy as np
import scipy.linalg as spl
from util import cov_matrix, load_data, load_fan_data, inverse_sqrt_matrix

"""
class MyOperator(odl.Operator):
    def _call(self, x):
        result = self.range.zero()
        for k in range(self.domain.size):
            for i in range(self.domain[0].size):
                for j in range(self.domain[0].size):
                    result += x[k][i] * x[k][j]
        return result.ufunc.sqrt()

    def derivative(self, x):
        result = sum(x, x[0].space.zero())
        result /= self(x)
        return odl.PointwiseSum(self.domain[0]) * odl.PointwiseSum(self.domain) * self.domain.element([result] * self.domain.size)


class MyOperatorTransp(odl.Operator):
    def _call(self, x):
        result = self.range.zero()
        for k in range(self.domain[0].size):
            for i in range(self.domain.size):
                for j in range(self.domain.size):
                    result += x[j][k] * x[j][k]
        return result.ufunc.sqrt()

    def derivative(self, x):
        result = self.domain.zero()

        sx = self(x)
        for i in range(self.domain[0].size):
            result[0] += x[0][i] / sx

        for i in range(1, self.domain.size):
            result[i].assign(result[0])

        return odl.PointwiseSum(self.domain[0]) * odl.PointwiseSum(self.domain) * result
"""

class MyOperatorTrace(odl.Operator):
    def _call(self, x):
        result = self.range.zero()
        for i in range(self.domain.size):
            for j in range(self.domain[0].size):
                result += x[i][j] * x[i][j]
        return result.ufunc.sqrt()

    def derivative(self, x):
        result = x.copy()
        sx = self(x)
        for xi in result:
            for xii in xi:
                xii /= sx
        return odl.PointwiseSum(self.domain[0]) * odl.PointwiseSum(self.domain) * result

data, geometry, crlb = load_fan_data(return_crlb=True)

space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
A = odl.DiagonalOperator(ray_trafo, 2)

grad = odl.Gradient(space)
grad_vec = odl.DiagonalOperator(grad, 2)

op = A
rhs = data

data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)

l1_norm = odl.solvers.L1Norm(space)
huber = 4.0 * odl.solvers.MoreauEnvelope(l1_norm, sigma=0.01)

my_op = MyOperatorTrace(domain=grad_vec.range, range=space, linear=False)
func = data_discrepancy * op + huber * my_op * grad_vec

fbp_op = odl.tomo.fbp_op(ray_trafo,
                         filter_type='Hann', frequency_scaling=0.7)
x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])

callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackShow(display_step=1, clim=[0.9, 1.1]) &
            odl.solvers.CallbackPrintIteration())

opnorm = odl.power_method_opnorm(op)
hessinv_estimate = odl.ScalingOperator(func.domain, 1 / opnorm ** 2)

odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=1000, num_store=10,
                        callback=callback, hessinv_estimate=hessinv_estimate)
