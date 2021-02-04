from typing import Union

import numpy as np
from scipy import ndimage

from ..utils.measure import bounding_box


def gaussian(sigma: Union[list, np.ndarray], scale: int = 4):
    """
    Generates a Gaussian kernel of a given size.

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


def ellipsoid(axis_sizes: Union[list, np.ndarray],
              phi: float = 0,
              theta: float = 0,
              cval: float = 255.,
              margin: int = 3):
    """
    Generate an ellipsoid mask of a given axis size.

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
    margin : int, optional
        Margin between ellipsoid shape and image border in pixels.
        Default is 3.

    Returns
    -------
    numpy.ndarray
        2- or 3-dimensional binary image of ellipsoid
    """

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
    gauss_kernel = gaussian(axis_sizes / np.min(axis_sizes) * 3, scale=int(round(np.min(axis_sizes)/3)))

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
    indmin, indmax = bounding_box(gauss_kernel)
    slc = tuple([slice(max(0, indmin[i] - margin),
                       min(gauss_kernel.shape[i], indmax[i] + 1 + margin))
                 for i in range(len(indmin))])
    gauss_kernel = gauss_kernel[slc]

    return gauss_kernel
