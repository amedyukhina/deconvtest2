import numpy as np


def check_type(names, variables, types):
    for name, var, t in zip(names, variables, types):
        t = np.array([t]).flatten()
        if not type(var) in t:
            raise TypeError("Type of '{}' must be one of: {}; '{}' provided!".format(name, t, type(var).__name__))