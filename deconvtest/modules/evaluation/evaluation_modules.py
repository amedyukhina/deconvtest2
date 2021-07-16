import numpy as np

from ...core.utils.conversion import unify_shape
from ...core.utils.utils import check_type


def rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Root Mean Square Error (RMSE) between two input images.

    Parameters
    ----------
    img1 : ndarray
        Input ground truth image.
        This image's volume is used to normalize the output RMSE
    img2 : numpy.ndarray
        Second input image.

    Returns
    -------
    float:
        RMSE between the two input images
    """
    volume = np.prod(img1.shape)
    check_type(['img1', 'img2'],
               [img1, img2],
               [np.ndarray] * 2)
    img1, img2 = unify_shape(img1, img2)
    return np.sqrt(np.sum((img1 - img2) ** 2) / volume)


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

