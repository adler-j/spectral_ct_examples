"""This formulation solves the model

    min_x ||Ax - b||_W^2 + ||grad(x)||_*

where A is the ray transform, || . ||_W is the weighted l2 norm,
grad is the gradient and || . ||_* is the nuclear norm.
"""

import odl
import numpy as np
import scipy.linalg as spl
from util import cov_matrix, load_data, load_fan_data, inverse_sqrt_matrix

class MyOperatorTrace(odl.Operator):
    def _call(self, x):
        result = self.range.zero()
        for i in range(self.domain.size):
            for j in range(self.domain[0].size):
                result += x[i][j] * x[i][j]
        return result.ufuncs.sqrt()

    def derivative(self, x):
        result = x.copy()
        sx = self(x)
        for xi in result:
            for xii in xi:
                xii /= sx
        return odl.PointwiseSum(self.domain[0]) * odl.PointwiseSum(self.domain) * result


class LamOp(odl.Operator):
    def __init__(self, space, arr):
        self.arr = arr
        odl.Operator.__init__(self, space, space, True)

    def _call(self, x):
        result = self.range.zero()
        for k in range(self.domain.size):
            for i in range(self.domain.size):
                for j in range(self.domain[0].size):
                    result[i][j] += self.arr[k][i] * x[k][j]

        return result

    @property
    def adjoint(self):
        return LamOp(self.domain, self.arr.T)


for cross_terms in [False, True]:
    for c in [0.0, 0.5]:
        data, geometry, crlb = load_fan_data(return_crlb=True)

        space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

        ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
        A = odl.DiagonalOperator(ray_trafo, 2)

        grad = odl.Gradient(space)
        grad_vec = odl.DiagonalOperator(grad, 2)

        if not cross_terms:
            crlb[1, 0, ...] = 0
            crlb[0, 1, ...] = 0

        mat_sqrt_inv = inverse_sqrt_matrix(crlb)

        re = ray_trafo.range.element
        W = odl.ProductSpaceOperator([[odl.MultiplyOperator(re(mat_sqrt_inv[0, 0])), odl.MultiplyOperator(re(mat_sqrt_inv[0, 1]))],
                                      [odl.MultiplyOperator(re(mat_sqrt_inv[1, 0])), odl.MultiplyOperator(re(mat_sqrt_inv[1, 1]))]])

        mat_sqrt_inv_hat = mat_sqrt_inv.mean(axis=(2,3))
        mat_sqrt_inv_hat_inv = np.linalg.inv(mat_sqrt_inv_hat)

        I = odl.IdentityOperator(space)
        precon = odl.ProductSpaceOperator(np.multiply(mat_sqrt_inv_hat_inv, I))
        precon_inv = odl.ProductSpaceOperator(np.multiply(mat_sqrt_inv_hat, I))

        op = W * A
        rhs = W(data)

        data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)

        l1_norm = odl.solvers.L1Norm(space)

        lam_values = np.linspace(0, 5, 51)

        for lam in lam_values:
            huber = lam * odl.solvers.MoreauEnvelope(l1_norm, sigma=0.005)

            my_op = MyOperatorTrace(domain=grad_vec.range, range=space, linear=False)

            spc_cov_matrix = [[1, -c],
                              [-c, 1]]
            spc_cov_matrix_inv_sqrt = inverse_sqrt_matrix(spc_cov_matrix)
            Lam = LamOp(grad_vec.range, arr=spc_cov_matrix_inv_sqrt)

            func = (data_discrepancy * op + huber * my_op * Lam * grad_vec) * precon

            fbp_op = odl.tomo.fbp_op(ray_trafo,
                                     filter_type='Hann', frequency_scaling=0.7)
            x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])
            x = precon_inv(x)

            opnorm = odl.power_method_opnorm(op * precon)
            hessinv_estimate = odl.ScalingOperator(func.domain, 1 / opnorm ** 2)

            callback = odl.solvers.CallbackPrintIteration('cross_terms={}, c={}, lam={}'.format(cross_terms, c, lam))

            odl.solvers.bfgs_method(func, x, line_search=1.0, maxiter=1000, num_store=10,
                                    callback=callback,
                                    hessinv_estimate=hessinv_estimate)

            result = precon(x)

            import scipy.io as sio
            mdict = {'water': result[0].asarray(), 'bone': result[1].asarray(),
                     'c': c, 'cross_terms': cross_terms}
            sio.savemat('data/results/parameters/result_crlb_correlated_c_{}_cross_terms_{}_lam_{}'.format(c, cross_terms, lam), mdict)
