import numpy as np
from skimage.metrics import structural_similarity

from ...core.utils.conversion import unify_shape


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two input images.

    Parameters
    ----------
    img1, img2 : ndarray
        Input images of the same shape

    Returns
    -------
    float:
        SSIM between the two input images
    """
    img1, img2 = unify_shape(img1, img2)
    return structural_similarity(img1 * 1., img2 * 1., full=False)
