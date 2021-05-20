import warnings
from typing import Union

import numpy as np

from ...core.utils.utils import check_type


def poisson_noise(img: np.ndarray, snr: Union[int, float]):
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
        img[np.where(img < 0)] = 0
        img = img.astype(np.float32)
        imgmax = snr ** 2  # new image maximum to generate the right level of Poisson noise
        ratio = imgmax / img.max()  # keep the ratio of the new and old maximum to recover the old dynamic range
        img = img * ratio
        img = np.random.poisson(img)
        if ratio > 0:
            img = img / ratio
        return img
