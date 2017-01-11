"""Example of Filtered-back projection reconstruction using ODL."""

import odl
from util import load_fan_data
import numpy as np

data, geometry = load_fan_data()

space = odl.uniform_discr([-129, -129], [129, 129], [500, 500])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

fbp_operator = odl.tomo.fbp_op(ray_trafo,
                               filter_type='Hann', frequency_scaling=0.3)

fbp_reconstruction = fbp_operator(data[0])
fbp_reconstruction.show('fbp_reconstruction 0', clim=[0.9, 1.1])
fbp_reconstruction = fbp_operator(data[1])
fbp_reconstruction.show('fbp_reconstruction 1')
