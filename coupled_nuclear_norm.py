import scipy.io as sio
import odl
import numpy as np

data_mat = sio.loadmat('E:/Data/spectral_ct/aux_corr_in_real_ct_image.mat')
data = data_mat['decomposedBasisProjectionsmmObj']
data = data.swapaxes(0, 2)

reco_space = odl.uniform_discr([-150, -150], [150, 150], [200, 200])

angle_interval = odl.uniform_partition(0, np.pi, 180)
detector_partition = odl.uniform_partition(-150 * np.sqrt(2),
                                           150 * np.sqrt(2),
                                           853)

geometry = odl.tomo.Parallel2dGeometry(angle_interval, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

grad = odl.Gradient(reco_space, method='forward')

I = odl.IdentityOperator(ray_trafo.range)
c = -0.5

A = odl.DiagonalOperator(ray_trafo, ray_trafo)
W_sqrt = odl.ProductSpaceOperator([[(np.sqrt(1 - c) + np.sqrt(c + 1)) * I, (np.sqrt(c + 1) - np.sqrt(1 - c)) * I],
                                   [(np.sqrt(c + 1) - np.sqrt(1 - c)) * I, (np.sqrt(1 - c) + np.sqrt(c + 1)) * I]])
L = odl.DiagonalOperator(grad, grad)

op = W_sqrt * A
rhs = W_sqrt(data)

data_discrepancy = odl.solvers.L2Norm(A.range).translated(rhs)
regularizer = 0.1 * odl.solvers.NuclearNorm(L.range, singular_vector_exp=1)

fbp_op = odl.tomo.fbp_op(ray_trafo)
x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])

f = odl.solvers.ZeroFunctional(A.domain)
#f = odl.solvers.IndicatorNonnegativity(A.domain)
g = [data_discrepancy, regularizer]
L = [op, L]
tau = 1.0
sigma = [0.0003, 1]
niter = 1000

callback = odl.solvers.CallbackShow(display_step=1)

odl.solvers.douglas_rachford_pd(x, f, g, L, tau, sigma, niter,
                                callback=callback)
