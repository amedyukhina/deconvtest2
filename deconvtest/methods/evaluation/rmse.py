import numpy as np

from ...core.utils.conversion import unify_shape
from ...core.utils.utils import check_type


def rmse(gt: np.ndarray, img: np.ndarray) -> float:
    """
    Compute Root Mean Square Error (RMSE) between two input images.

    Parameters
    ----------
    gt : ndarray
        Input ground truth image.
        This image's volume is used to normalize the output RMSE
    img : numpy.ndarray
        Second input image.

    Returns
    -------
    float:
        RMSE between the two input images
    """
    volume = np.prod(gt.shape)
    check_type(['gt', 'img'],
               [gt, img],
               [np.ndarray] * 2)
    gt, img = unify_shape(gt, img)
    return np.sqrt(np.sum((gt - img) ** 2) / volume)
