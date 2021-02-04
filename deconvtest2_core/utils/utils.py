import numpy as np
import importlib.util
import inspect
import os
from pathlib import Path
MODULE_EXTENSIONS = '.py'


def check_type(names, variables, types):
    for name, var, t in zip(names, variables, types):
        t = list(np.array([t]).flatten())
        if not type(var) in t:
            raise TypeError("Type of '{}' must be one of: {}; '{}' provided!".format(name, t, type(var).__name__))


def __list_modules(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return set()

    pathname = Path(spec.origin).parent
    ret = set()
    with os.scandir(pathname) as entries:
        for entry in entries:
            if entry.name.startswith('__'):
                continue
            current = '.'.join((package_name, entry.name.partition('.')[0]))
            if entry.is_file():
                if entry.name.endswith(MODULE_EXTENSIONS):
                    ret.add(current)
            elif entry.is_dir():
                ret.add(current)
                ret |= __list_modules(current)

    return ret


def list_modules(package):
    package_name = package.__name__
    modules = list(__list_modules(package_name))
    functions = []
    for module in modules:
        if len(module.split('tests')) == 1:
            m = importlib.import_module(module)
            module_functions = inspect.getmembers(m, inspect.isfunction)
            for function in module_functions:
                func_info = inspect.getfullargspec(function[1])
                if function[1].__module__.startswith(package_name):
                    functions.append((function[1], func_info))
    return functions