import odl
import numpy as np
import scipy.io as sio


def get_indicator(phantom_array, name):
    if name == 'bone':
        indicator = (phantom_array > 1.75)
    elif name == 'eye':
        indicator = (phantom_array > 1.058) & (phantom_array < 1.062)
    elif name == 'blood':
        indicator = (phantom_array > 1.053) & (phantom_array < 1.056)
    elif name == 'denser_sphere':
        indicator = (phantom_array > 1.052) & (phantom_array < 1.053)
    elif name == 'brain':
        indicator = (phantom_array > 1.048) & (phantom_array < 1.052)
    elif name == 'less_dense_sphere':
        indicator = (phantom_array > 1.047) & (phantom_array < 1.048)
    elif name == 'csf':
        indicator = (phantom_array > 1.043) & (phantom_array < 1.047)
    else:
        assert 0
    return indicator

space = odl.uniform_discr([-129, -129], [129, 129], [2017, 2017])

det_size = 853
n_angles = 360
n_super_sample = 10

angle_partition = odl.uniform_partition(0.0, 2.0 * np.pi, n_angles * n_super_sample)
detector_partition = odl.uniform_partition(-det_size / 2.0,
                                           det_size / 2.0,
                                           det_size * n_super_sample)

geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500,
                                    det_radius=500)

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')  # change to astra_cpu

phantom = odl.phantom.forbild(space, resolution=True, ear=True)
phantom.show('phantom')

phantom_array = np.asarray(phantom)

all_names = ['bone', 'eye', 'blood', 'denser_sphere',
             'brain', 'less_dense_sphere', 'csf']

mdict = {}
phantom_dict = {'phantom': phantom.asarray()}

for name in all_names:
    indicator = get_indicator(phantom_array, name)

    sinogram = ray_trafo(indicator)
    sinogram.show(name)

    sinogram_array = np.asarray(sinogram)

    # reduce sinogram size
    sinogram_array_downsample = sinogram_array.reshape((n_angles, n_super_sample, det_size, n_super_sample)).mean(axis=(1, 3)).reshape((n_angles, det_size))

    mdict[name] = sinogram_array_downsample
    phantom_dict[name] = indicator

sio.savemat('data/material_proj_data', mdict, do_compression=True)
sio.savemat('raw_phantom', phantom_dict)

# Try the data with a more coarse projector
if False:
    space_coarse = odl.uniform_discr([-129, -129], [129, 129], [600, 600])

    angle_partition_coarse = odl.uniform_partition(0.0, 2.0 * np.pi, 360)
    detector_partition_coarse = odl.uniform_partition(-det_size / 2.0,
                                               det_size / 2.0,
                                               det_size)

    geometry_coarse = odl.tomo.FanFlatGeometry(angle_partition_coarse, detector_partition_coarse,
                                               src_radius=500,
                                               det_radius=500)

    ray_trafo_coarse = odl.tomo.RayTransform(space_coarse, geometry_coarse, impl='astra_cuda')  # change to astra_cpu

    fbp_op = odl.tomo.fbp_op(ray_trafo_coarse, filter_type='Hann')

    fbp = fbp_op(sinogram_array_downsample)
    fbp.show('fbp')
