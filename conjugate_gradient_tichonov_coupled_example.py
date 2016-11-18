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
c = -0.8

A = odl.DiagonalOperator(ray_trafo, ray_trafo)
W = odl.ProductSpaceOperator([[I, c * I],
                              [c * I, I]])
L = odl.DiagonalOperator(grad, grad)

op = A.adjoint * W * A + 3 * L.adjoint * L

fbp_op = odl.tomo.fbp_op(ray_trafo)

x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])
rhs = A.adjoint(W(data))
odl.solvers.conjugate_gradient_normal(op, x, rhs, 1000,
                                      callback=odl.solvers.CallbackShow(display_step=10))
