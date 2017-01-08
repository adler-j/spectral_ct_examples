import numpy as np
from scipy import signal
import scipy.io as sio
import os
import odl


def load_data():
    """Get the data from disk.

    Returns
    -------
    mat1_sino, mat2_sino : numpy.ndarray
        projection of material 1 and 2
    geometry : odl.tomo.Geometry
        Geometry of the data
    """
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_path,
                             'data',
                             'aux_corr_in_real_ct_image.mat')

    try:
        data_mat = sio.loadmat(data_path)
    except IOError:
        raise IOError('data/aux_corr_in_real_ct_image.mat missing, contact '
                      'developers for a copy of the data or use another data '
                      'source.')

    data = data_mat['decomposedBasisProjectionsmmObj']
    data = data.swapaxes(0, 2)

    angle_partition = odl.uniform_partition(0, np.pi, 180)
    detector_partition = odl.uniform_partition(-150 * np.sqrt(2),
                                               150 * np.sqrt(2),
                                               853)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    return data, geometry


def fan_to_fan_flat(geometry, data):
    tmp_space = odl.uniform_discr_frompartition(geometry.partition,
                                                interp='linear')
    rot_angles = tmp_space.meshgrid[0]
    fan_angles = tmp_space.meshgrid[1]
    data = tmp_space.element(data)

    source_to_detector = geometry.src_radius + geometry.det_radius

    fan_dist = source_to_detector * np.arctan(fan_angles / source_to_detector)
    data = data.interpolation((rot_angles, fan_dist),
                              bounds_check=False)
    data = data[::-1]
    return data


def load_fan_data(return_crlb=False, fan_flat_data=True):
    if fan_flat_data:
        file_name = 'runs_2017_01_07_lineardet'
    else:
        file_name = 'simulated_images_2017_01_06'

    current_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_path,
                             'data', file_name,
                             'head_image.mat')

    try:
        data_mat = sio.loadmat(data_path)
    except IOError:
        raise IOError('data/simulated_images_2017_01_06/head_image.mat missing, '
                      'contact '
                      'developers for a copy of the data or use another data '
                      'source.')

    # print(sorted(data_mat.keys()))
    data = data_mat['decomposedbasisProjectionsmm']
    data = data.swapaxes(0, 2)

    if fan_flat_data:
        det_size = 853

        angle_partition = odl.uniform_partition(0.5 * np.pi, 2.5 * np.pi, 360)
        detector_partition = odl.uniform_partition(-det_size / 2.0,
                                                   det_size / 2.0,
                                                   853)

        geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                            src_radius=500,
                                            det_radius=500)

        data[:] = data[:, ::-1]

        if not return_crlb:
            return data, geometry
        else:
            crlb = data_mat['CRLB']
            crlb = crlb.swapaxes(0, 1)
            crlb[:] = crlb[::-1]
            crlb = np.moveaxis(crlb, [-2, -1], [0, 1])

            # Negative correlation
            crlb[0, 1] *= -1
            crlb[1, 0] *= -1

            return data, geometry, crlb
    else:
        # Create approximate fan flat geometry.
        det_size = 883 * (500 + 500)

        angle_partition = odl.uniform_partition(0.5 * np.pi, 2.5 * np.pi, 360)
        detector_partition = odl.uniform_partition(-det_size / 2.0,
                                                   det_size / 2.0,
                                                   883)

        geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                            src_radius=500,
                                            det_radius=500)

        # Convert to true fan flat geometry
        data[0][:] = fan_to_fan_flat(geometry, data[0])
        data[1][:] = fan_to_fan_flat(geometry, data[1])

        if not return_crlb:
            return data, geometry
        else:
            crlb = data_mat['CRLB']
            crlb = crlb.swapaxes(0, 1)
            crlb = np.moveaxis(crlb, [-2, -1], [0, 1])

            crlb[0, 0][:] = fan_to_fan_flat(geometry, crlb[0, 0])
            crlb[0, 1][:] = fan_to_fan_flat(geometry, crlb[0, 1])
            crlb[1, 0][:] = fan_to_fan_flat(geometry, crlb[1, 0])
            crlb[1, 1][:] = fan_to_fan_flat(geometry, crlb[1, 1])

            # Negative correlation
            crlb[0, 1] *= -1
            crlb[1, 0] *= -1

            return data, geometry, crlb


def estimate_cov(I1, I2):
    """Estiamte the covariance of I1 and I2."""
    assert I1.shape == I2.shape

    H, W = I1.shape

    M = np.array([[1, -2, 1],
                  [-2, 4., -2],
                  [1, -2, 1]])

    sigma = np.sum(signal.convolve2d(I1, M) * signal.convolve2d(I2, M))
    sigma /= (W * H - 1)

    return sigma / 36.0  # unknown factor, too lazy to solve


def inverse_sqrt_matrix(mat):
    """Compute pointwise inverse square root of matri(ces).

    See formula from wikipedia:
    https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 1]

    tau = a + c
    delta = a * c - b * b

    s = np.sqrt(delta)
    t = np.sqrt(tau + 2 * s)

    mat_sqrt = np.zeros(mat.shape)
    mat_sqrt[0, 0] = a + s
    mat_sqrt[0, 1] = b
    mat_sqrt[1, 0] = b
    mat_sqrt[1, 1] = c + s
    mat_sqrt /= t[None, None]

    # compute inverse, need to move 2x2 matrix to last
    mat_sqrt = np.moveaxis(mat_sqrt, [0, 1], [-2, -1])
    mat_sqrt_inv = np.linalg.inv(mat_sqrt)
    mat_sqrt_inv = np.moveaxis(mat_sqrt_inv, [-2, -1], [0, 1])

    return mat_sqrt_inv


def cov_matrix(data):
    """Estimate the covariance matrix from data.

    Parameters
    ----------
    data : kxnxm `numpy.ndarray`
        Estimates the covariance along the first dimension.

    Returns
    -------
    cov_mat : kxk `numpy.ndarray`
        Covariance matrix.
    """
    n = len(data)

    cov_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            cov_mat[i, j] = estimate_cov(data[i], data[j])

    return cov_mat


if __name__ == '__main__':
    # Example
    I1 = np.random.randn(50, 50)
    I2 = 3 * np.random.randn(50, 50)
    corr_variable = (I1 + I2)

    print(estimate_cov(I1, I1))  # should be 1
    print(estimate_cov(I1, I2))  # should be 0
    print(estimate_cov(I2, I1))  # should be 0
    print(estimate_cov(I2, I2))  # should be 9
    print(estimate_cov(I1, corr_variable))  # should be 1
