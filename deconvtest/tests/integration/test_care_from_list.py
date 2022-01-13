import os
import unittest
import shutil
import numpy as np
from am_utils.utils import imsave
from ddt import ddt
from skimage import io

from ...core.utils.constants import *
from ...framework.workflow.workflow import Workflow
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
                   input_dir=os.path.join(path, 'data'))
        w.add_step('ImageList', 'symlink_to_workflow_folder',
                   input_dir=os.path.join(path, 'noise'))
        w.add_step('Organize', 'care_prep', input_step=[0, 1],
                   img_high='pipeline', img_low='pipeline')

        w.save()
        w.save_workflow_graph()
        w.run()
        files = os.listdir(os.path.join(w.path, DATA_FOLDER_NAME))
        shutil.rmtree(w.path)
        self.assertEqual(len(files), 5)


if __name__ == '__main__':
    unittest.main()
