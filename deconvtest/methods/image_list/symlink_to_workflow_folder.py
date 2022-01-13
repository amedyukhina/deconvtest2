import os

import pandas as pd
from am_utils.utils import walk_dir


def symlink_to_workflow_folder(input_dir: str, **kwargs):
    output_dir = kwargs.get('output_dir')
    base_name = kwargs.get('base_name')
    files = walk_dir(input_dir)
    ids = [base_name + fn[len(input_dir) + 1:].replace('/', '_') for fn in files]
    for fn, fn_out in zip(files, ids):
        fn_out = os.path.join(output_dir, fn_out)
        if not os.path.exists(fn_out):
            os.makedirs(os.path.dirname(fn_out), exist_ok=True)
            os.symlink(fn, fn_out)
    ids = [fn.split('.')[0] for fn in ids]
    table = pd.DataFrame({'ID': ids})
    return table
