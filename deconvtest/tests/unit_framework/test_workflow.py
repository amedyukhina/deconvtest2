import os
import shutil
import unittest

import numpy as np
import pandas as pd
from ddt import ddt

from ...core.utils.constants import *
from ...framework.workflow.utils import generate_id_table
from ...framework.workflow.workflow import Workflow


@ddt
class TestWorkflow(unittest.TestCase):

    def test_list_workflow_steps(self):
        w = Workflow()
        steps = [st.name for st in w.available_steps]
        self.assertIn('PSF', steps)

    def test_add_step(self):
        w = Workflow()
        w.add_step('PSF', 'gaussian')
        self.assertEqual(len(w.steps), 1)

    def test_export_import(self):
        w = Workflow()
        w.add_step('PSF', 'gaussian', sigma=[1, 2, 3], aspect=[3, 2, 4], parmeter_mode='align')

        path = 'test.json'
        w.save(path)

        w2 = Workflow()
        w2.load(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        shutil.rmtree(w2.path)
        self.assertEqual(str(w), str(w2))

    def test_missing_method(self):
        w = Workflow()
        self.assertRaises(ValueError, w.add_step, 'PSF')

    def test_wrong_step_order(self):
        w = Workflow()
        self.assertRaises(IndexError, w.add_step, 'Convolution', 'convolve')

    def test_wrong_step_number(self):
        w = Workflow(name='test workflow')
        w.add_step('GroundTruth', 'ellipsoid')
        w.add_step('PSF', 'gaussian')

        self.assertRaises(ValueError, w.add_step, 'Convolution', 'convolve', input_step=[1, 0, 0])

    def test_wrong_step_values(self):
        w = Workflow(name='test workflow')
        w.add_step('GroundTruth', 'ellipsoid')
        w.add_step('PSF', 'gaussian')
        self.assertRaises(IndexError, w.add_step, 'Convolution', 'convolve', input_step=[1, 2])

    def test_multiple_evaluation_methods(self):
        w = Workflow(name='test workflow')

        w.add_step('GroundTruth', 'ellipsoid',
                   size=[[10, 6, 6], 10], voxel_size=[[0.5, 0.2, 0.2]], parmeter_mode='align')
        w.add_step('Transform', 'poisson_noise', img='pipeline', snr=[2, 5])
        w.add_step('Evaluation', method=['rmse', 'nrmse'], input_step=[0, 1], gt='pipeline', img='pipeline')

    def test_id_table(self):
        w = Workflow()
        w.add_step('GroundTruth', 'ellipsoid',
                   size=[[10, 6, 6], 10], voxel_size=[[0.5, 0.2, 0.2]],
                   theta=[0, np.pi / 2], phi=[np.pi, np.pi * 4 / 3],
                   parmeter_mode='permute')
        w.run(verbose=False)

        generate_id_table(os.path.join(w.path, DATA_FOLDER_NAME), os.path.join(w.path, 'id_table.csv'))
        ids = pd.read_csv(os.path.join(w.path, 'id_table.csv'))
        self.assertEqual(len(ids), 8)
        shutil.rmtree(w.path)


if __name__ == '__main__':
    unittest.main()
