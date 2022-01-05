import os

import pandas as pd
from am_utils.utils import walk_dir


def generate_id_table(input_dir: str, output_file: str):
    files = walk_dir(input_dir)
    ids = [fn[len(input_dir) + 1:].replace('/', '_')[:-len(fn.split('.')[-1]) - 1] for fn in files]
    table = pd.DataFrame({'ID': ids,
                          'Filename': files})
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    table.to_csv(output_file, index=False)


def generate_id_table_with_folders(input_dir: str, output_file: str):
    ids = [fn for fn in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, fn))]
    files = [os.path.join(input_dir, fn) for fn in ids]
    table = pd.DataFrame({'ID': ids,
                          'Filename': files})
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    table.to_csv(output_file, index=False)
