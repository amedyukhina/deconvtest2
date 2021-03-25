import numpy as np
from scipy.signal import fftconvolve

from ...core.utils.utils import check_type


def convolve(img: np.ndarray, psf: np.ndarray):
    """
    Convolve input image with a PSF.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    psf : numpy.ndarray
        PSF image with the same number of dimensions as the input image

    Returns
    -------
    numpy.ndarray:
        Convolved image.
    """
    check_type(['img', 'psf'],
               [img, psf],
               [np.ndarray] * 2)
    if len(img.shape) != len(psf.shape):
        raise ValueError('Input image and PSF must have the same dimension. '
                         'Provided array were {}D for the input and {}D for the PSF'.format(len(img.shape),
                                                                                            len(psf.shape)))
    return fftconvolve(img, psf, mode='full')
