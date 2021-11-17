import os
import unittest
import shutil

import numpy as np
from ddt import ddt

from ...framework.workflow.step import Step
from ...framework.workflow.workflow import Workflow


@ddt
class TestWorkflow(unittest.TestCase):

    def test_workflow(self):
        path = 'test_workflow'
        w = Workflow(name='test workflow', output_path=os.path.join(path, 'data'))

        s = Step('GroundTruth', 'ellipsoid')
        path_gt = 'params_ellipsoid.csv'
        s.specify_parameters(size=[[10, 6, 6], 10], voxel_size=[[0.5, 0.2, 0.2]],
                             theta=[0, np.pi / 2], phi=[np.pi, np.pi * 4 / 3], mode='align', base_name='GT')
        s.save_parameters(os.path.join(path, path_gt))
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        path_psf = 'params_psf.csv'
        s.specify_parameters(sigma=[1, 2], aspect=[2, 4], mode='align')
        s.save_parameters(os.path.join(path, path_psf))
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        s.specify_parameters(img='pipeline', psf='pipeline')
        w.add_step(s, input_step=[0, 1])

        s = Step('Transform', 'poisson_noise')
        s.specify_parameters(img='pipeline', snr=[2, 5], base_name='noise')
        path_noise = 'params_noise.csv'
        s.save_parameters(os.path.join(path, path_noise))
        w.add_step(s)

        # s = Step('Evaluation', ['rmse', 'nrmse'])
        # s.specify_parameters(gt='pipeline', img='pipeline')
        # w.add_step(s, input_step=[0, 3])

        s = Step('Organize', 'care_prep')
        s.specify_parameters(img_high='pipeline', img_low='pipeline')
        w.add_step(s, input_step=[0, 3])

        wpath = 'workflow.json'
        w.save(os.path.join(path, wpath))
        print(w.get_workflow_graph())

        w.run(verbose=False)
        files = os.listdir(os.path.join(path, 'data'))
        shutil.rmtree(path)
        # self.assertEqual(len(files), 24)


if __name__ == '__main__':
    unittest.main()
