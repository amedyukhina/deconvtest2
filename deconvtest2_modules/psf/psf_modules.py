from __future__ import division
import numpy as np
from scipy import ndimage


def gaussian(sigma: float, aspect: float = 1., scale: int = 8, **kwargs_to_ignore: dict):
    if not type(sigma) in [float, int]:
        raise TypeError("'sigma' must be int or float, '{}' provided!".format(type(sigma).__name__))
    if not type(aspect) in [float, int]:
        raise TypeError("'aspect' must be int or float, '{}' provided!".format(type(aspect).__name__))
    if not type(scale) is int:
        raise TypeError("'scale' must be integer, '{}' provided!".format(type(scale).__name__))

    zsize = int(round((sigma + 1) * aspect)) * scale + 1
    xsize = int(round(sigma + 1)) * scale + 1

    psf = np.zeros([zsize, xsize, xsize])  # create an empty array
    psf[int(zsize / 2), int(xsize / 2), int(xsize / 2)] = 255.  # create a peak in the center of the image
    psf = ndimage.gaussian_filter(psf, [sigma * aspect, sigma, sigma])  # smooth the peak with a Gaussian
    psf = psf / np.max(psf)
    return psf


