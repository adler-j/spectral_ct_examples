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

back_proj = ray_trafo.adjoint(data[1])
back_proj.show('back_proj')

fbp_operator = odl.tomo.fbp_op(ray_trafo)

fbp_reconstruction = fbp_operator(data[0])
fbp_reconstruction.show('fbp_reconstruction 0')
fbp_reconstruction = fbp_operator(data[1])
fbp_reconstruction.show('fbp_reconstruction 1')
