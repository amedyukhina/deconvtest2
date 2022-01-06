import os
import shutil
import unittest

import numpy as np
from ddt import ddt

from ...core.utils.constants import *
from ...framework.workflow.step import Step
from ...framework.workflow.workflow import Workflow


@ddt
class TestWorkflow(unittest.TestCase):

    def test_workflow(self):
        path = 'test_workflow'
        w = Workflow(name='test workflow', path=path)

        s = Step('GroundTruth', 'ellipsoid')
        s.specify_parameters(size=[[10, 6, 6], 10], voxel_size=[[0.5, 0.2, 0.2]],
                             theta=[0, np.pi / 2], phi=[np.pi, np.pi * 4 / 3], mode='align', base_name='GT')
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        s.specify_parameters(sigma=[1, 2], aspect=[2, 4], mode='align')
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        s.specify_parameters(img='pipeline', psf='pipeline', conv_mode='same')
        w.add_step(s, input_step=[0, 1])

        s = Step('Transform', 'poisson_noise')
        s.specify_parameters(img='pipeline', snr=[2, 5], base_name='noise')
        w.add_step(s)

        s = Step('Evaluation', ['rmse', 'nrmse'])
        s.specify_parameters(gt='pipeline', img='pipeline')
        w.add_step(s, input_step=[0, 3])

        w.save()
        w.run(verbose=False)
        files = os.listdir(os.path.join(path, DATA_FOLDER_NAME))
        shutil.rmtree(path)
        self.assertEqual(len(files), 24)


if __name__ == '__main__':
    unittest.main()
