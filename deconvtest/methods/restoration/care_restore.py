from __future__ import print_function, unicode_literals, absolute_import, division

import os

import numpy as np
from csbdeep.models import CARE
from csbdeep.utils.tf import limit_gpu_memory


def care_restore(img: np.ndarray, model: str, limit_gpu: float = 0.5, axes: str = 'ZYX', **kwargs):
    """

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    model : str
        Model name (full path).
    limit_gpu : float, optional
        Fraction of the GPU memory to use.
        Default: 0.5
    axes : str
        Axes of the input ``img``.
    kwargs : dict
        Configuration attributes (see below).

    Attributes
    ----------
    normalizer : :class:`csbdeep.data.Normalizer` or None
        Normalization of input image before prediction and (potentially) transformation back after prediction.
    resizer : :class:`csbdeep.data.Resizer` or None
        If necessary, input image is resized to enable neural network prediction and result is (possibly)
        resized to yield original image size.
    n_tiles : iterable or None
        Out of memory (OOM) errors can occur if the input image is too large.
        To avoid this problem, the input image is broken up into (overlapping) tiles
        that can then be processed independently and re-assembled to yield the restored image.
        This parameter denotes a tuple of the number of tiles for every image axis.
        Note that if the number of tiles is too low, it is adaptively increased until
        OOM errors are avoided, albeit at the expense of runtime.
        A value of ``None`` denotes that no tiling should initially be used.

    """
    limit_gpu_memory(fraction=limit_gpu)
    model = CARE(config=None, name=model.split('/')[-1], basedir=os.path.dirname(model))
    restored = model.predict(img, axes=axes, **kwargs)
    return restored
