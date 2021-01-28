import numpy as np
from typing import Union


def convert_voxel_size(voxel_size: Union[float, list, np.ndarray]):
    """
    Convert voxel size into a uniform format of 3-number array.

    Parameters
    ----------
    voxel_size : scalar or sequence
        Voxel size in z, y and x used to generate the PSF image.
        If one value is provided, the voxel size is assumed to be equal along all axes.

    Returns
    -------
    numpy.array
        Formatted voxel size
    """
    if not type(voxel_size) in [float, int, list, np.ndarray]:
        raise TypeError("'voxel_size' must be number or sequence, '{}' provided!".format(type(voxel_size).__name__))
    voxel_size = np.array([voxel_size]).flatten()
    if len(voxel_size) == 1:
        voxel_size = np.array([voxel_size[0]]*3)
    elif len(voxel_size) == 2:
        voxel_size = np.array([voxel_size[0], voxel_size[1], voxel_size[1]])
    elif len(voxel_size) == 3:
        voxel_size = voxel_size
    else:
        raise ValueError('voxel size must be a number of a sequence of length 2 or 3!')
    return voxel_size

