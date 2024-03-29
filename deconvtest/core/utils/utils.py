import importlib
import inspect
import os
from pathlib import Path

import numpy as np

from .errors import raise_not_valid_type_error

MODULE_EXTENSIONS = '.py'


def check_type(names, variables, types):
    for name, var, t in zip(names, variables, types):
        t = list(np.array([t]).flatten())
        if not type(var) in t:
            raise_not_valid_type_error(type(var).__name__, name, t)


def is_valid_type(variable, valid_type):
    if type(valid_type) is type:
        valid_types = [valid_type]
    else:
        valid_types = list(valid_type.__args__)

    return type(variable) in valid_types


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


def list_modules(package, module_type=inspect.isfunction):
    package_name = package.__name__
    modules = list(__list_modules(package_name))
    functions = []
    for module in modules:
        if len(module.split('tests')) == 1:
            m = importlib.import_module(module)
            module_functions = inspect.getmembers(m, module_type)
            for function in module_functions:
                func_info = inspect.getfullargspec(function[1])
                if function[1].__module__.startswith(package_name):
                    functions.append((function[1], func_info))
    return functions


def modules_to_json(modules):
    functions = []
    for module in modules:
        function = dict({'name': module[0].__name__,
                         'module': module[0].__module__,
                         })
        meta = module[1]._asdict()
        for arg in meta['annotations'].keys():
            if type(meta['annotations'][arg]) is type:
                meta['annotations'][arg] = meta['annotations'][arg].__name__
            else:
                meta['annotations'][arg] = [t.__name__ for t in meta['annotations'][arg].__args__]
        for key in meta.keys():
            function[key] = meta[key]
        functions.append(function)
    return functions
