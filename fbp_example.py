"""Example of Filtered-back projection reconstruction using ODL."""

import odl
from util import load_data

data, geometry = load_data()

space = odl.uniform_discr([-129, -129], [129, 129], [600, 600])

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

fbp_operator = odl.tomo.fbp_op(ray_trafo,
                               filter_type='Hann', frequency_scaling=0.7)

fbp_reconstruction = fbp_operator(data[0])
fbp_reconstruction.show('fbp_reconstruction 0')
fbp_reconstruction = fbp_operator(data[1])
fbp_reconstruction.show('fbp_reconstruction 1')
