import re
from typing import Union

import numpy as np
import pandas as pd

from ..utils.utils import check_type


def convert_size(size: Union[float, list, np.ndarray]):
    """
    Convert voxel size or object dimension into a uniform format of 3-number array.

    Parameters
    ----------
    size : scalar or sequence
        Voxel size or object dimensions in z, y and x.
        If one value is provided, the size is assumed to be equal along all axes.

    Returns
    -------
    numpy.array
        Formatted voxel size
    """
    size = np.array([size]).flatten()
    if len(size) == 1:
        size = np.array([size[0]] * 3)
    elif len(size) == 2:
        size = np.array([size[0], size[1], size[1]])
    elif len(size) == 3:
        size = size
    else:
        raise ValueError('Size must be a number of a sequence of length 2 or 3!')
    return size


def unify_shape(x: np.ndarray, y: np.ndarray):
    """
    Pads the two given arrays to bring them to the same shape.

    Parameters
    ----------
    x, y : ndarray
        Input arrays
    Returns
    -------
    x, y : ndarray
        Modified input arrays having the same shape.
    """
    check_type(['x', 'y'],
               [x, y],
               [np.ndarray] * 2)
    if len(x.shape) != len(y.shape):
        raise ValueError("The number of dimensions in the two arrays must be equal!")
    nshape = np.array([x.shape, y.shape]).max(0)
    out = []
    for arr in [x, y]:
        shape_diff = (nshape - np.array(arr.shape))
        pad_width = [(int(shape_diff[i] / 2), shape_diff[i] - int(shape_diff[i] / 2)) for i in range(len(shape_diff))]
        out.append(np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0))

    return out


def list_to_keys(params: dict, sep: str = '_'):
    """
    Convert list values in a dictionary to individual dictionary entries.

    Parameters
    ----------
    params : dict
        Dictionary to convert
    sep : str, optional
        Separator to separate indices.
        Default is '_'

    Returns
    -------
    dict:
        Converted dictionary

    """
    params_converted = dict()
    for key in params.keys():
        if type(params[key]) in [list, np.array]:
            for i, value in enumerate(params[key]):
                params_converted[key + sep + str(i)] = value
        else:
            params_converted[key] = params[key]
    return params_converted


def list_to_columns(params: pd.DataFrame, sep: str = '_'):
    """
    Convert list values in a table to individual column entries.

    Parameters
    ----------
    params : dict
        Table to convert
    sep : str, optional
        Separator to separate indices.
        Default is '_'

    Returns
    -------
    dict:
        Converted table

    """
    list_length = dict()
    for key in params.columns:
        list_length[key] = 0
        for ind in range(len(params)):
            if type(params[key].iloc[ind]) in [list, np.array]:
                list_length[key] = max(len(params[key].iloc[ind]), list_length[key])

    params_converted = pd.DataFrame()
    for key in params.columns:
        if list_length[key] > 0:
            df_tmp = pd.DataFrame()
            for ind in range(len(params)):
                df_tmp_row = pd.DataFrame()
                if type(params[key].iloc[ind]) in [list, np.array]:
                    for i, value in enumerate(params[key].iloc[ind]):
                        df_tmp_row[key + sep + str(i)] = [value]
                else:
                    for i in range(list_length[key]):
                        df_tmp_row[key + sep + str(i)] = [params[key].iloc[ind]]
                df_tmp = pd.concat([df_tmp, df_tmp_row], ignore_index=True)
            for col in df_tmp.columns:
                params_converted[col] = df_tmp[col]

        else:
            params_converted[key] = params[key]

    return params_converted


def keys_to_list(params: dict, sep: str = '_'):
    """
    Convert key values in a dictionary that have a common stem to one key with a list value.

    Parameters
    ----------
    params : dict
        Dictionary to convert
    sep : str, optional
        Separator that separates indices.
        Default is '_'

    Returns
    -------
    dict:
        Converted dictionary

    """
    params_converted = dict()
    p = re.compile(rf'(.+){sep}\d')
    keys = []
    values = []
    for key in params.keys():
        if len(p.findall(key)) > 0:
            keys.append(p.findall(key)[0])
            values.append(params[key])
        else:
            params_converted[key] = params[key]
    if len(keys) > 1:
        keys = np.array(keys)
        values = np.array(values)
        unique_keys = np.unique(keys)
        for unique_key in unique_keys:
            params_converted[unique_key] = list(values[np.where(keys == unique_key)])
    return params_converted
