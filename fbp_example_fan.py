"""Example of Filtered-back projection reconstruction using ODL."""

import odl
from util import load_fan_data
import numpy as np

data, geometry = load_fan_data()

space = odl.uniform_discr([-129, -129], [129, 129], [400, 400])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

filter_type = 'Hann'
frequency_scaling = 0.8
fbp_operator = odl.tomo.fbp_op(ray_trafo,
                               filter_type=filter_type,
                               frequency_scaling=frequency_scaling)

fbp_reconstruction_water = fbp_operator(data[0])
fbp_reconstruction_water.show('fbp_reconstruction water', clim=[0.9, 1.1])
fbp_reconstruction_bone = fbp_operator(data[1])
fbp_reconstruction_bone.show('fbp_reconstruction bone')

import scipy.io as sio
mdict = {'water': fbp_reconstruction_water.asarray(), 'bone': fbp_reconstruction_bone.asarray(),
         'filter_type': filter_type, 'frequency_scaling': frequency_scaling}
sio.savemat('result_fbp', mdict)
