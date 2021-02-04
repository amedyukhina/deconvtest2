from typing import Union

import numpy as np
from deconvtest2_core.shapes import shapes
from deconvtest2_core.utils.conversion import convert_size


def gaussian(sigma: float, aspect: float = 1., voxel_size: Union[float, list, np.ndarray] = 1.):
    """
    Generates a Point Spread Function (PSF) as a 3D Gaussian with given standard deviation and aspect ratio.

    Parameters
    ----------
    sigma : float
        Standard deviation in xy in micrometers of the Gaussian function that is used to approximate the PSF.
    aspect : float, optional
        Ratio between the Gaussian standard deviations in z and xy.
        Default is 1.
    voxel_size : scalar or sequence
        Voxel size in z, y and x used to generate the PSF image.
        If one value is provided, the voxel size is assumed to be equal along all axes.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Output 3D image of the PSF.
    """

    if not type(sigma) in [float, int]:
        raise TypeError("'sigma' must be int or float, '{}' provided!".format(type(sigma).__name__))
    if not type(aspect) in [float, int]:
        raise TypeError("'aspect' must be int or float, '{}' provided!".format(type(aspect).__name__))

    voxel_size = convert_size(voxel_size)
    sigmas = np.array([aspect * sigma, sigma, sigma]) / voxel_size
    psf = shapes.gaussian(sigma=sigmas, scale=8)
    return psf
