import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the peak signal to noise ratio (PSNR) for a test image

    Parameters
    ----------
    img1 : ndarray
        Ground truth image
    img2 : numpy.ndarray
        Test image

    Returns
    -------
    float:
        PSNR for the test image
    """
    return peak_signal_noise_ratio(img1 * 1., img2 * 1., data_range=np.max(img1) - np.min(img1))
