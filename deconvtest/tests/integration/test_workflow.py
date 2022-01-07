import os
import shutil
import unittest

import numpy as np
from ddt import ddt

from ...core.utils.constants import *
from ...framework.workflow.workflow import Workflow


@ddt
class TestWorkflow(unittest.TestCase):

    def test_workflow(self):
        path = 'test_workflow'
        w = Workflow(name='test workflow', path=path)

        w.add_step('GroundTruth', 'ellipsoid', parmeter_mode='align',
                   size=[[10, 6, 6], 10], voxel_size=[[0.5, 0.2, 0.2]],
                   theta=[0, np.pi / 2], phi=[np.pi, np.pi * 4 / 3])
        w.add_step('PSF', 'gaussian', parmeter_mode='align', sigma=[1, 2], aspect=[2, 4])
        w.add_step('Convolution', 'convolve', input_step=[0, 1], img='pipeline', psf='pipeline', conv_mode='same')
        w.add_step('Transform', 'poisson_noise', img='pipeline', snr=[2, 5])
        w.add_step('Evaluation', ['rmse', 'nrmse'], input_step=[0, 3], gt='pipeline', img='pipeline')

        w.run(verbose=False)
        files = os.listdir(os.path.join(path, DATA_FOLDER_NAME))
        shutil.rmtree(path)
        self.assertEqual(len(files), 24)


if __name__ == '__main__':
    unittest.main()
