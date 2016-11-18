import scipy.io as sio
import odl
import numpy as np

data_mat = sio.loadmat('E:/Data/spectral_ct/aux_corr_in_real_ct_image.mat')
data = data_mat['decomposedBasisProjectionsmmObj']
data = data.swapaxes(0, 2)

reco_space = odl.uniform_discr([-150, -150], [150, 150], [600, 600])

angle_interval = odl.uniform_partition(0, np.pi, 180)
detector_partition = odl.uniform_partition(-150 * np.sqrt(2),
                                           150 * np.sqrt(2),
                                           853)

geometry = odl.tomo.Parallel2dGeometry(angle_interval, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

L = odl.Gradient(reco_space, method='forward')

op = ray_trafo.adjoint * ray_trafo + 10 * L.adjoint * L

x = ray_trafo.domain.zero()
rhs = ray_trafo.adjoint(data[1])
odl.solvers.conjugate_gradient_normal(op, x, rhs, 100,
                                      callback=odl.solvers.CallbackShow())
