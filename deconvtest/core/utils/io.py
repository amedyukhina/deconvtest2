import pandas as pd
import numpy as np
from csbdeep.io import save_training_data
from skimage import io


def read_img(fn):
    return io.imread(fn)


def read_stat(fn):
    return pd.read_csv(fn)


def read_file(fn):
    return fn


def write_img(fn, output, dtype=np.uint16):
    io.imsave(fn, output.astype(dtype))


def write_stat(fn, output):
    output.to_csv(fn, index=False)


def write_file(fn, output):
    return


READ_FN = dict(image=read_img,
               stat=read_stat,
               folder=read_file,
               file=read_file,
               data=read_file,
               model=read_file)

WRITE_FN = dict(image=write_img,
                stat=write_stat,
                folder=write_file,
                file=write_file,
                data=write_file,
                model=write_file)


def read(fn, type_input):
    return READ_FN[type_input](fn)


def write(fn, output, type_output, **kwargs):
    WRITE_FN[type_output](fn, output, **kwargs)
