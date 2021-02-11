import unittest

import numpy as np
import os
from ddt import ddt, data
from skimage import io

from ...deconvolution.deconvolutionlab2 import regularized_inverse_filter
from ...ground_truth.ground_truth_modules import ellipsoid
from ...psf.psf_modules import gaussian
from ...transforms.transform_modules import convolve


@ddt
class TestRIF(unittest.TestCase):

    def test_rif(self):
        plugin_path = '/Applications/Fiji.app/plugins/DeconvolutionLab_2.jar'
        data_path = '/Users/amedyukh/Documents/StJude/DeconvTestScratch/'
        psf = gaussian(0.1, 0.3)
        x = ellipsoid(10, 0.3)
        xconv = convolve(x, psf)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        io.imsave(data_path + 'psf.tif', (psf*255).astype(np.uint8))
        io.imsave(data_path + 'conv.tif', xconv.astype(np.uint16))
        regularized_inverse_filter(0.1, data_path, plugin_path)
        self.assertTrue(os.path.exists(data_path + 'deconv.tif'))

    if __name__ == '__main__':
        unittest.main()
