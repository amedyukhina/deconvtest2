import os
import shutil
import unittest

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

        s = Step('Evaluation', ['rmse', 'nrmse'])
        s.specify_parameters(gt='pipeline', img='pipeline')
        w.add_step(s, input_step=[0, 3])

        wpath = 'workflow.json'
        w.save(os.path.join(path, wpath))

        w.run(verbose=False)
        files = os.listdir(os.path.join(path, 'data'))
        shutil.rmtree(path)
        self.assertEqual(len(files), 24)

    def test_care_workflow(self):
        path = 'test_workflow'
        w = Workflow(name='test workflow', output_path=os.path.join(path, 'data'))

        s = Step('GroundTruth', 'ellipsoid')
        path_gt = 'params_ellipsoid.csv'
        s.specify_parameters(size=[10], voxel_size=[1],
                             mode='align', base_name='GT')
        s.save_parameters(os.path.join(path, path_gt))
        w.add_step(s)

        s = Step('Transform', 'poisson_noise')
        s.specify_parameters(img='pipeline', snr=[2, 5], base_name='noise')
        path_noise = 'params_noise.csv'
        s.save_parameters(os.path.join(path, path_noise))
        w.add_step(s)

        s = Step('Organize', 'care_prep')
        s.specify_parameters(img_high='pipeline', img_low='pipeline')
        w.add_step(s, input_step=[0, 1])

        s = Step('DataGen', 'care_datagen')
        s.specify_parameters(base_dir='pipeline', patch_size=[[4, 4, 4]],
                             n_patches_per_image=[5], verbose=False)
        path_datagen = 'params_datagen.csv'
        s.save_parameters(os.path.join(path, path_datagen))
        w.add_step(s)

        s = Step('Training', 'care_train')
        s.specify_parameters(data_file='pipeline', limit_gpu=0.2, train_batch_size=2,
                             train_epochs=1, train_steps_per_epoch=10, validation_split=0.5)
        w.add_step(s)

        s = Step('Restoration', 'care_restore')
        s.specify_parameters(img='pipeline', model='pipeline')
        w.add_step(s, input_step=[1, 4])

        s = Step('Evaluation', ['rmse', 'nrmse', 'ssim'])
        s.specify_parameters(gt='pipeline', img='pipeline')
        w.add_step(s, input_step=[0, 5])

        wpath = 'workflow.json'
        w.save(os.path.join(path, wpath))
        # gr = w.get_workflow_graph()
        # for item in gr['items']:
        #     print('Item:', item, '\nSteps:\n')
        #     for m in item['modules']:
        #         print(m)

        w.run(verbose=False, nsteps=4)
        w.run(verbose=False, njobs=1, nsteps=6)
        w.run(verbose=False)
        files = os.listdir(os.path.join(path, 'data'))
        shutil.rmtree(path)
        self.assertEqual(len(files), 17)


if __name__ == '__main__':
    unittest.main()
