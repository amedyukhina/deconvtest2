import os
import shutil
import unittest

import numpy as np
import pandas as pd
from am_utils.utils import imsave
from ddt import ddt
from skimage import io

from ...core.utils.constants import *
from ...framework.workflow.workflow import Workflow
from ...methods.evaluation.nrmse import nrmse
from ...methods.evaluation.ssim import ssim
from ...methods.transforms.poisson_noise import poisson_noise


@ddt
class TestWorkflow(unittest.TestCase):

    def test_care_workflow(self):

        w = Workflow(name='data_for_care')
        w.add_step('GroundTruth', 'ellipsoid',
                   size=[10, 11], voxel_size=[1],
                   parmeter_mode='align')  # 0
        w.run(verbose=False)

        path = w.path
        for fn in os.listdir(os.path.join(path, 'data')):
            img = io.imread(os.path.join(path, 'data', fn))
            img = poisson_noise(img, 5)
            imsave(os.path.join(path, 'noise', fn), img.astype(np.uint8))

        w = Workflow(name='test workflow')
        w.add_step('ImageList', 'symlink_to_workflow_folder',
                   input_dir=os.path.join(path, 'data'))  # 0
        w.add_step('ImageList', 'symlink_to_workflow_folder',
                   input_dir=os.path.join(path, 'noise'))  # 1
        w.add_step('Organize', 'care_prep', input_step=[0, 1],
                   img_high='pipeline', img_low='pipeline')  # 2
        w.add_step('DataGen', 'care_datagen',
                   base_dir='pipeline', patch_size=[[4, 4, 4]],
                   n_patches_per_image=[5], verbose=False)  # 3
        w.add_step('Training', 'care_train',
                   data_file='pipeline', limit_gpu=0.2, train_batch_size=2,
                   train_epochs=1, train_steps_per_epoch=10, validation_split=0.5)  # 4
        w.add_step('Restoration', 'care_restore', input_step=[1, 4], img='pipeline', model='pipeline')  # 5
        w.add_step('Evaluation', ['rmse', 'nrmse', 'ssim'], input_step=[0, 5], gt='pipeline', img='pipeline')  # 6

        w.save()

        w.run(verbose=False, nsteps=4)
        w.run(verbose=False, njobs=1, nsteps=6)
        w.run(verbose=False)
        files = os.listdir(os.path.join(w.path, DATA_FOLDER_NAME))
        self.assertEqual(len(files), 11)

        for fn1, fn2 in [('00_00_0000.tif', '02_0000/high/00_00_0000.tif'),
                         ('00_00_0001.tif', '02_0000/high/00_00_0001.tif'),
                         ('01_00_0000.tif', '02_0000/low/00_00_0000.tif'),
                         ('01_00_0001.tif', '02_0000/low/00_00_0001.tif')
                         ]:
            img1 = io.imread(os.path.join(w.path, DATA_FOLDER_NAME, fn1))
            img2 = io.imread(os.path.join(w.path, DATA_FOLDER_NAME, fn2))
            self.assertAlmostEqual(np.sum(abs(img1 - img2)), 0, 4)

        for fn1, fn2, fn3 in [('00_00_0000.tif', '05_00000000000000000000.tif', '06_000000000000000000000000.csv'),
                              ('00_00_0001.tif', '05_00010000000000000000.tif', '06_000100000000000000000000.csv')
                              ]:
            img1 = io.imread(os.path.join(w.path, DATA_FOLDER_NAME, fn1))
            img2 = io.imread(os.path.join(w.path, DATA_FOLDER_NAME, fn2))
            df = pd.read_csv(os.path.join(w.path, DATA_FOLDER_NAME, fn3)).iloc[0]
            self.assertAlmostEqual(ssim(img1, img2), df['ssim'], 4)
            self.assertAlmostEqual(nrmse(img1, img2), df['nrmse'], 4)

        shutil.rmtree(w.path)
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
