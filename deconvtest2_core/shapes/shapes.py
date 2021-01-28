import numpy as np
from scipy import ndimage


def gaussian(sigma: list, scale: int = 8):
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
    size = np.int_(np.round_(sigma)) * scale + 1

    kernel = np.zeros(size)  # create an empty array
    kernel[np.int_(size / 2)] = 255.  # create a peak in the center of the kernel
    kernel = ndimage.gaussian_filter(kernel, sigma)  # smooth the peak with a Gaussian
    kernel = kernel / np.max(kernel)  # normalize the kernel
    return kernel
