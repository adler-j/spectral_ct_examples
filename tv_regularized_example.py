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
grad = odl.Gradient(reco_space, method='forward')

b = ray_trafo.range.element(data[1])

l2_sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(b)
l1 = odl.solvers.L1Norm(grad.range)
f = odl.solvers.IndicatorNonnegativity(reco_space)

g = [l2_sq, l1]
L = [ray_trafo, grad]
tau = 1.0
sigma = [1 / odl.power_method_opnorm(ray_trafo)**2,
         1 / odl.power_method_opnorm(grad)**2]

x = reco_space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, L, tau, sigma, niter=1000,
                                callback=odl.solvers.CallbackShow(display_step=20))
