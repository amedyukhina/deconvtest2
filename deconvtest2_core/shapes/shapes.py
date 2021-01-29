from typing import Union

import numpy as np
from scipy import ndimage


def gaussian(sigma: Union[list, np.ndarray], scale: int = 4):
    """
    Generates a Point Spread Function (PSF) as a 3D Gaussian with given standard deviation and aspect ratio.

    Parameters
    ----------
    sigma : np.ndarray
        Standard deviations in pixels of the Gaussian kernel to generate.
    scale : int, optional
        Multiplier to specify the dimensions of the output kernel.
        Default is 8.

    Returns
    -------
    numpy.ndarray
        Output Gaussian kernel with the number of dimensions specified by the length of `sigma`.
    """
    if not type(sigma) in [float, int, list, np.ndarray]:
        raise TypeError(
            "'sigma' must be int, float, list, or numpy.ndarray '{}' provided!".format(type(sigma).__name__))
    if not type(scale) is int:
        raise TypeError("'scale' must be integer, '{}' provided!".format(type(scale).__name__))
    sigma = np.array([sigma]).flatten()

    # calculate the size of the output array based on sigma and scale; make sure it is non-zero
    size = np.int_(np.round_(sigma))
    size[np.where(size < 1)] = 1
    size = size * 2 * scale + 1

    kernel = np.zeros(size)  # create an empty array
    kernel[tuple(np.int_(size / 2))] = 255.  # create a peak in the center of the kernel
    kernel = ndimage.gaussian_filter(kernel, sigma)  # smooth the peak with a Gaussian
    kernel = kernel / np.max(kernel)  # normalize the kernel
    return kernel


def ellipsoid(axis_sizes: list, phi: float = 0, theta: float = 0, cval: float = 255.):
    """

    Parameters
    ----------
    axis_sizes : list of scalars
        Sizes of the ellipsoid axes in pixels.
        Must be a list or array of size 2 or 3.
    phi : float, optional
        Azimuthal rotation angle in the range from 0 to 2 pi.
        If 0, no azimuthal rotation is done.
        Default is 0.
    theta : float, optional
        Polar rotation angle in the range from 0 to pi.
        If 0, no polar rotation is done.
        Default is 0.
    cval : float, optional
        Value to fill the ellipsoid.
        Default is 255.

    Returns
    -------
    numpy.ndarray
        2- or 3-dimensional binary image of ellipsoid
    """
    if not type(axis_sizes) in [list, np.ndarray]:
        raise TypeError(
            "'axis_sizes' must be a list or numpy.ndarray; '{}' provided!".format(type(axis_sizes).__name__))
    if not type(phi) in [float, int]:
        raise TypeError(
            "'phi' must be a int or float; '{}' provided!".format(type(phi).__name__))
    if not type(theta) in [float, int]:
        raise TypeError(
            "'theta' must be a int or float; '{}' provided!".format(type(theta).__name__))
    if not type(cval) in [float, int]:
        raise TypeError(
            "'cval' must be a int or float; '{}' provided!".format(type(cval).__name__))

    # convert axis sizes
    axis_sizes = np.array([axis_sizes]).flatten()

    # convert the angles from radians to degrees
    phi = phi * 180 / np.pi
    theta = theta * 180 / np.pi

    # calculate target ellipsoid volume to adjust the threshold accordingly
    if len(axis_sizes) == 2:
        target_volume = np.pi * np.prod(axis_sizes / 2.)
    elif len(axis_sizes) == 3:
        target_volume = 4. / 3 * np.pi * np.prod(axis_sizes / 2.)
    else:
        raise NotImplementedError('Implemented for 2 or 3 dimensions. '
                                  'The number of dimensions provided was {}'.format(len(axis_sizes)))

    # generate a Gaussian kernel
    gauss_kernel = gaussian(axis_sizes / np.min(axis_sizes), scale=int(round(np.min(axis_sizes))))

    # Rotate the Gaussian kernel according to the input angles
    if theta > 0:  # rotate the image, if rotation angles are not 0
        gauss_kernel = ndimage.interpolation.rotate(gauss_kernel, theta, axes=(0, 1),
                                                    reshape=True, mode='constant', cval=0)
        if phi > 0:
            gauss_kernel = ndimage.interpolation.rotate(gauss_kernel, phi, axes=(1, 2),
                                                        reshape=True, mode='constant', cval=0)

    # compute the intensity percentile (for the thresholding) that corresponds to the target volume of the cells
    percentile = 100 - target_volume * 100. / np.prod(gauss_kernel.shape)

    # threshold the gaussian kernel
    gauss_kernel = (gauss_kernel >= np.percentile(gauss_kernel, percentile)) * cval
    return gauss_kernel
