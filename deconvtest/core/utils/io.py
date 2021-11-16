import pandas as pd
from skimage import io


def read_img(fn):
    return io.imread(fn)


def read_stat(fn):
    return pd.read_csv(fn)


def read_folder(fn):
    return fn


def write_img(fn, output):
    io.imsave(fn, output)


def write_stat(fn, output):
    output.to_csv(fn, index=False)


def write_folder(fn, output):
    return


READ_FN = dict(image=read_img,
               stat=read_stat,
               folder=read_folder)

WRITE_FN = dict(image=write_img,
                stat=write_stat,
                folder=write_folder)


def read(fn, type_input):
    return READ_FN[type_input](fn)


def write(fn, output, type_output):
    WRITE_FN[type_output](fn, output)
