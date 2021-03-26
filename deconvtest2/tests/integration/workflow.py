import unittest

import numpy as np
from ddt import ddt

from ...framework.workflow.step import Step
from ...framework.workflow.workflow import Workflow


@ddt
class TestWorkflow(unittest.TestCase):

    def test_workflow(self):
        w = Workflow(name='test workflow')

        s = Step('GroundTruth', 'ellipsoid')
        path_gt = 'params_ellipsoid.csv'
        s.specify_parameters(size=[10], voxel_size=0.5,
                             theta=[0, np.pi / 2], phi=[np.pi, np.pi * 4 / 3], mode='permute', base_name='GT')
        s.save_parameters(path_gt)
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        path_psf = 'params_psf.csv'
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.save_parameters(path_psf)
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        s.specify_parameters(img='pipeline', psf='pipeline')
        w.add_step(s, input_step=[0, 1])

        s = Step('Transform', 'poisson_noise')
        s.specify_parameters(img='pipeline', snr=[2, 5, 10], base_name='noise')
        path_noise = 'params_noise.csv'
        s.save_parameters(path_noise)
        w.add_step(s)

        s = Step('Evaluation', 'rmse')
        s.specify_parameters(img1='pipeline', img2='pipeline')
        w.add_step(s, input_step=[0, 3])

        path = 'workflow.json'
        w.save(path)
        w2 = Workflow()
        w2.load(path)
        path_graph = 'workflow_graph.json'
        w2.save_workflow_graph(path_graph)


if __name__ == '__main__':
    unittest.main()
