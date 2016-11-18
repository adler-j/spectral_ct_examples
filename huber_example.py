import scipy.io as sio
import odl
import numpy as np

class HuberNorm(odl.solvers.Functional):
    def __init__(self, space):
        odl.solvers.Functional.__init__(self, space, linear=False)

    def _call(self, x):
        result = np.where(np.less_equal(np.abs(x), 1),
                          x**2 / 2.0,
                          np.abs(x) - 0.5)

        return np.sum(result)

    def gradient(self, x):
        result = np.where(np.less_equal(np.abs(x), 1),
                          x,
                          np.sign(x))

        return result

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
d1 = odl.PartialDerivative(reco_space, axis=0, method='forward')
d2 = odl.PartialDerivative(reco_space, axis=1, method='forward')

b = ray_trafo.range.element(data[1])

l2 = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - b)
g1 = HuberNorm(d1.range) * d1
g2 = HuberNorm(d2.range) * d2

func = l2 + 2 * (g1 * 10 + g2 * 10)

x = reco_space.zero()
odl.solvers.smooth.steepest_descent(func, x, line_search=0.0001,
                                    callback=odl.solvers.CallbackShow())
