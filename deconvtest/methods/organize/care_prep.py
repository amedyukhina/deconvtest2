import os

from skimage import io

from ...core.utils.conversion import unify_shape


def care_prep(img_high: str, img_low: str,
              name_high: str = 'high', name_low: str = 'low',
              output_dir: str = 'test', img_name: str = 'test.tif'):
    img_high, img_low = unify_shape(img_high, img_low)
    os.makedirs(os.path.join(output_dir, name_low), exist_ok=True)
    os.makedirs(os.path.join(output_dir, name_high), exist_ok=True)
    io.imsave(os.path.join(output_dir, name_high, img_name), img_high)
    io.imsave(os.path.join(output_dir, name_low, img_name), img_low)
