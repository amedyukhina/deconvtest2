from typing import Union

import numpy as np

from ...core.shapes import shapes
from ...core.utils.conversion import convert_size
from ...core.utils.utils import check_type


def ellipsoid(size: Union[float, list, np.ndarray],
              voxel_size: Union[float, list, np.ndarray] = 1.,
              theta: float = 0,
              phi: float = 0):
    """
    Generates a synthetic object of ellipsoidal shape.

    Parameters
    ----------
    size : scalar or sequence of scalars
        Size of the cell in micrometers.
        If only one value is provided, the size along all axes is assume to be equal (spherical cell).
        To specify individual size for each axis (ellipsoid cells), 3 values should be provided ([z, y, x]).
    voxel_size : scalar or sequence
        Voxel size in z, y and x used to generate the PSF image.
        If one value is provided, the voxel size is assumed to be equal along all axes.
        Default is 1.
    phi : float, optional
        Azimuthal rotation angle in the range from 0 to 2 pi.
        If 0, no azimuthal rotation is done.
        Default is 0.
    theta : float, optional
        Polar rotation angle in the range from 0 to pi.
        If 0, no polar rotation is done.
        Default is 0.
    """

    check_type(['size', 'voxel_size', 'theta', 'phi'],
               [size, voxel_size, theta, phi],
               [[int, float, list, np.ndarray], [float, int, list, np.ndarray], [float, int], [float, int]])

    voxel_size = convert_size(voxel_size)
    size = convert_size(size) / voxel_size
    ell = shapes.ellipsoid(axis_sizes=size, phi=phi, theta=theta, cval=255, margin=3)
    return ell
