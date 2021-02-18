import warnings

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


def poisson_noise(img: np.ndarray, snr: float):
    """
    Add Poisson noise to the input image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    snr : float
        Signal-to-noise ratio for the Poisson noise.
    Returns
    -------
    numpy.ndarray:
        Noisy image of the same shape as the input image.
    """
    check_type(['img', 'snr'],
               [img, snr],
               [np.ndarray, [float, int]])
    if snr is None:
        warnings.warn("SNR is None, returning the input image")
        return img
    else:
        img = img.astype(np.float32)
        imgmax = snr ** 2  # new image maximum to generate the right level of Poisson noise
        ratio = imgmax / img.max()  # keep the ratio of the new and old maximum to recover the old dynamic range
        img = img * ratio
        img = np.random.poisson(img)
        if ratio > 0:
            img = img / ratio
        return img
