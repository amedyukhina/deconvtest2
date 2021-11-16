import numpy as np
from skimage.metrics import structural_similarity

from ...core.utils.conversion import unify_shape


def ssim(gt: np.ndarray, img: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two input images.

    Parameters
    ----------
    gt, img : ndarray
        Input images of the same shape

    Returns
    -------
    float:
        SSIM between the two input images
    """
    gt, img = unify_shape(gt, img)
    return structural_similarity(gt * 1., img * 1., full=False)
