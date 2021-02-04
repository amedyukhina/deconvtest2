import numpy as np


def bounding_box(arr: np.ndarray):
    ind = np.array(np.where(arr > 0))
    if len(ind[0]) == 0:
        return [None] * len(arr.shape), [None] * len(arr.shape)
    return ind.min(1), ind.max(1)
