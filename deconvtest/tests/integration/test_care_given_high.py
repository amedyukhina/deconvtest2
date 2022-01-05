import os
import shutil
import unittest

from ddt import ddt
from skimage import io
import numpy as np
from am_utils.utils import imsave

from ...framework.workflow.step import Step
from ...framework.workflow.workflow import Workflow
from ...framework.workflow.utils import generate_id_table
from ...methods.transforms.poisson_noise import poisson_noise


@ddt
class TestWorkflow(unittest.TestCase):

    def test_care_workflow(self):
        path = 'test_workflow_care'
        w = Workflow(name='test workflow', output_path=os.path.join(path, 'data/gt'))

        s = Step('GroundTruth', 'ellipsoid')
        s.specify_parameters(size=[10, 12], voxel_size=[1],
                             mode='align', base_name='GT')
        w.add_step(s)
        w.run(verbose=False)
        for fn in os.listdir(os.path.join(path, 'data/gt')):
            img = io.imread(os.path.join(path, 'data/gt', fn))
            img = poisson_noise(img, 5)
            imsave(os.path.join(path, 'data/noise', fn), img.astype(np.uint8))

        gt_path = os.path.join(path, 'input_data.csv')
        generate_id_table(os.path.join(path, 'data/gt'), gt_path)
        noise_path = os.path.join(path, 'noisy_data.csv')
        generate_id_table(os.path.join(path, 'data/noise'), noise_path)
        #
        # w = Workflow(name='test workflow2', output_path=os.path.join(path, 'data'))
        #
        # s = Step('GroundTruth', 'ellipsoid')
        # s.load_parameters(gt_path)
        # w.add_step(s)
        #
        # s = Step('DataGen', 'care_datagen')
        # s.specify_parameters(base_dir='pipeline', patch_size=[[4, 4, 4]],
        #                      n_patches_per_image=[5], verbose=False)
        # path_datagen = 'params_datagen.csv'
        # s.save_parameters(os.path.join(path, path_datagen))
        # w.add_step(s)
        #
        # s = Step('Training', 'care_train')
        # s.specify_parameters(data_file='pipeline', limit_gpu=0.2, train_batch_size=2,
        #                      train_epochs=1, train_steps_per_epoch=10, validation_split=0.5)
        # w.add_step(s)
        #
        # s = Step('Restoration', 'care_restore')
        # s.specify_parameters(img='pipeline', model='pipeline')
        # w.add_step(s, input_step=[1, 4])
        #
        # s = Step('Evaluation', ['rmse', 'nrmse', 'ssim'])
        # s.specify_parameters(gt='pipeline', img='pipeline')
        # w.add_step(s, input_step=[0, 5])
        #
        # w.run(verbose=False, nsteps=4)
        # w.run(verbose=False, njobs=1, nsteps=6)
        # w.run(verbose=False)


if __name__ == '__main__':
    unittest.main()
