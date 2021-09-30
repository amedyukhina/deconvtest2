import numpy as np

from .rmse import rmse


def nrmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Square Error (NRMSE) between two input images.

    Parameters
    ----------
    img1 : ndarray
        Input ground truth image.
        This image's volume and intensity range are used to normalize the output NRMSE
    img2 : numpy.ndarray
        Second input image.

    Returns
    -------
    float:
        NRMSE between the two input images
    """
    err = rmse(img1, img2)
    return err / (np.max(img1) - np.min(img1))
