import os
from time import sleep
from typing import Union

from csbdeep.data import RawData, create_patches


def care_datagen(base_dir: str, n_patches_per_image: int, name_high: str = 'high', name_low: str = 'low',
                 min_inputs: int = 0, axes: str = 'CZYX', fn_output: str = 'test',
                 patch_size: Union[int, list, tuple] = 10, verbose: bool = True):
    while len(os.listdir(os.path.join(base_dir, name_high))) < min_inputs or \
            len(os.listdir(os.path.join(base_dir, name_low))) < min_inputs:
        sleep(1)

    raw_data = RawData.from_folder(
        basepath=base_dir,
        source_dirs=[name_high],
        target_dir=name_low,
        axes=axes,
    )

    create_patches(raw_data, n_patches_per_image=n_patches_per_image,
                   patch_size=patch_size, verbose=verbose, save_file=fn_output)
