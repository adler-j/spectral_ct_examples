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

x = ray_trafo.domain.zero()
rhs = ray_trafo.range.element(data[1])
odl.solvers.conjugate_gradient_normal(ray_trafo, x, rhs, 100)

x.show('result')
