import numpy as np

from .rmse import rmse


def nrmse(gt: np.ndarray, img: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Square Error (NRMSE) between two input images.

    Parameters
    ----------
    gt : ndarray
        Input ground truth image.
        This image's volume and intensity range are used to normalize the output NRMSE
    img : numpy.ndarray
        Second input image.

    Returns
    -------
    float:
        NRMSE between the two input images
    """
    err = rmse(gt, img)
    return err / (np.max(gt) - np.min(gt))
