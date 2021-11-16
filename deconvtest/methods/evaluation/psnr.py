import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt: np.ndarray, img: np.ndarray) -> float:
    """
    Compute the peak signal to noise ratio (PSNR) for a test image

    Parameters
    ----------
    gt : ndarray
        Ground truth image
    img : numpy.ndarray
        Test image

    Returns
    -------
    float:
        PSNR for the test image
    """
    return peak_signal_noise_ratio(gt * 1., img * 1., data_range=np.max(gt) - np.min(gt))
