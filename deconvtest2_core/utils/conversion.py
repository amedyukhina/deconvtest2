from typing import Union

import numpy as np


def convert_size(size: Union[float, list, np.ndarray]):
    """
    Convert voxel size or object dimension into a uniform format of 3-number array.

    Parameters
    ----------
    size : scalar or sequence
        Voxel size or object dimensions in z, y and x.
        If one value is provided, the size is assumed to be equal along all axes.

    Returns
    -------
    numpy.array
        Formatted voxel size
    """
    size = np.array([size]).flatten()
    if len(size) == 1:
        size = np.array([size[0]] * 3)
    elif len(size) == 2:
        size = np.array([size[0], size[1], size[1]])
    elif len(size) == 3:
        size = size
    else:
        raise ValueError('Size must be a number of a sequence of length 2 or 3!')
    return size
