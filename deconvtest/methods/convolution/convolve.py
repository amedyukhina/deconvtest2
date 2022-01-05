import numpy as np
from scipy.signal import fftconvolve

from ...core.utils.utils import check_type


def convolve(img: np.ndarray, psf: np.ndarray, conv_mode: str = 'full'):
    """
    Convolve input image with a PSF.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    psf : numpy.ndarray
        PSF image with the same number of dimensions as the input image
    conv_mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    numpy.ndarray:
        Convolved image.
    """
    check_type(['img', 'psf'],
               [img, psf],
               [np.ndarray] * 2)
    if len(img.shape) != len(psf.shape):
        raise ValueError('Input image and PSF must have the same dimension. '
                         'Provided array were {}D for the input and {}D for the PSF'.format(len(img.shape),
                                                                                            len(psf.shape)))
    return fftconvolve(img, psf, mode=conv_mode)
