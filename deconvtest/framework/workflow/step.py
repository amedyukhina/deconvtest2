import inspect
import itertools
import json
import os
import warnings
from typing import Union

import numpy as np
import pandas as pd

from ...core.utils.constants import DEFAULT_PIPELINE_PARAM
from ...core.utils.conversion import list_to_keys, list_to_columns
from ...core.utils.errors import raise_mandatary_param_error, raise_not_valid_type_pipeline_error
from ...core.utils.errors import raise_not_valid_step_error, raise_not_valid_method_error
from ...core.utils.errors import warn_param_not_in_list
from ...core.utils.utils import list_modules, is_valid_type
from ...framework import module as available_steps


class Step:
    """
    class for a workflow step
    """

    def __init__(self, step_name: str, method: Union[str, list] = None):
        self.name = step_name
        self.parameters = pd.DataFrame()
        self.path = None
        self.n_inputs = None
        self.n_outputs = None
        self.type_input = None
        self.type_output = None
        self.align = False
        self.method = None
        self.methods = []  # the same as `method` but for the case of multiple methods (e.g. for evaluation)
        self.module = None
        self.input_step = None
        self.valid_parameters = None
        self.add_id = True

        self.available_modules = list_modules(available_steps, module_type=inspect.isclass)
        self.set_module()
        self.available_methods = self.list_available_methods()
        self.add_method(method)

    def list_available_methods(self):
        module = self.module()
        return module.list_available_methods_names()

    def set_module(self):
        for av_module in self.available_modules:
            if av_module[0].__name__ == self.name:
                self.module = av_module[0]
                self.n_inputs = av_module[0]().n_inputs
                self.n_outputs = av_module[0]().n_outputs
                self.type_input = av_module[0]().type_input
                self.type_output = av_module[0]().type_output
                self.align = av_module[0]().align
                self.add_id = av_module[0]().add_id
        if self.module is None:
            raise_not_valid_step_error(self.name, self.available_modules)

    def add_method(self, method: str):

        if method is not None:
            if type(method) is list:
                for m in method:
                    self.__add_method(m, append=True)
                self.method = self.methods
            else:
                self.__add_method(method, append=False)

    def __add_method(self, method, append=False):
        if method in self.available_methods:
            if append:
                self.methods.append(method)
            else:
                self.method = method
            module = self.module(method)
            self.n_inputs = module.n_inputs
            self.n_outputs = module.n_outputs
            self.valid_parameters = self.list_parameters()
        else:
            raise_not_valid_method_error(method, self.name, self.list_available_methods())

    def list_parameters(self):
        if self.method is None:
            warnings.warn(rf'No method is defined for step {self.name} to list parameters!')
            return None
        else:
            module = self.module(self.method)
            return module.parameters

    def specify_parameters(self, mode: str = 'permute', overwrite: bool = True,
                           base_name: str = None, sep: str = '', pos: int = 4,
                           **parameters):
        """
        Specify the list of parameters for the step.

        Parameters
        ----------
        mode : str, optional
            'permute' or 'align'
            If 'align', the module_base values for each module_base will be aligned.
            If 'permute', the combination of all possible module_base values will be generated.
            For 'align', the list of values for each module_base must have the same length.
            Default is 'permute'.
        overwrite : bool, optional
            If True, a new table will be created.
            If False, the generate table will be appended to the existing table.
            Default is True.
        base_name : str, optional
            Base name to label step items.
            If None, the step name is used as the `base_name`
            Default is None.
        sep : str, optional
            Separator between the base_name and the numeric ID.
            Default is '' (no separator).
        pos : int, optional
            Number of digit positions used for numeric ID.
            Default is 4.
        parameters : key value
            Parameter names and values.
            For values, provide one value or list.
            If `mode` is set to 'align', the length of all provided lists must be equal.

        Returns
        -------
        pandas.DataFrame()
            Table with module_base values
        """
        if mode not in ['align', 'permute']:
            raise ValueError(rf'{mode} is not a valid mode; must be "align" or "permute"')

        df_parameters = self.__get_param_table(parameters, mode)
        df_parameters = self.__add_ids(df_parameters, base=base_name, pos=pos, sep=sep)

        if overwrite:
            self.parameters = pd.DataFrame()

        self.parameters = pd.concat([self.parameters, df_parameters], ignore_index=True)
        return df_parameters

    def __get_param_table(self, parameters, mode):
        param_values_list, param_values_single = self.__get_parameter_lists(parameters)

        if mode == 'align':
            df_parameters = pd.DataFrame()
            length = len(param_values_list[list(param_values_list.keys())[0]])
            for key in param_values_list.keys():
                if not len(param_values_list[key]) == length:
                    raise ValueError(rf'{length}!={len(param_values_list[key])}. '
                                     'Lengths of module_base lists for mode "align" must be equal!')
                df_parameters[key] = param_values_list[key]
        else:
            values = np.array(list(itertools.product(*list(param_values_list.values()))))
            df_parameters = pd.DataFrame(values, columns=param_values_list.keys())

        param_values_single = list_to_keys(param_values_single)
        df_parameters = list_to_columns(df_parameters)
        for key in param_values_single.keys():
            if len(df_parameters) > 0:
                df_parameters[key] = param_values_single[key]
            else:
                df_parameters[key] = [param_values_single[key]]

        return df_parameters

    def __get_parameter_lists(self, parameters):
        method, method_param_names = self.get_method()

        for key in parameters.keys():
            if key not in method_param_names:
                warn_param_not_in_list(key, self.method)

        param_values_list = dict()
        param_values_single = dict()
        param_values_single['input_vars'] = []
        for param in method.parameters:
            if param.name not in parameters.keys():
                if param.optional:
                    param_values_single[param.name] = param.default_value
                else:
                    raise_mandatary_param_error(param.name)

            elif type(parameters[param.name]) is str and \
                    parameters[param.name] == DEFAULT_PIPELINE_PARAM:
                continue
            else:
                if type(parameters[param.name]) in [list, np.ndarray]:
                    is_list = True
                    for param_value in parameters[param.name]:
                        if not is_valid_type(param_value, param.type):
                            raise_not_valid_type_pipeline_error(type(param_value), param.name, param.type)

                    if len(parameters[param.name]) == 1:
                        is_list = False
                else:
                    is_list = False
                    if is_valid_type(parameters[param.name], param.type):
                        parameters[param.name] = [parameters[param.name]]
                    else:
                        raise_not_valid_type_pipeline_error(type(parameters[param.name]),
                                                            param.name, param.type)

                if is_list:
                    param_values_list[param.name] = parameters[param.name]
                else:
                    param_values_single[param.name] = parameters[param.name][0]

        return param_values_list, param_values_single

    def get_method(self):
        if self.method is None:
            raise ModuleNotFoundError(rf'No method is defined for step {self.name} to specify parameters!')
        else:
            if type(self.method) is list:
                method = self.module(self.method[0])
            else:
                method = self.module(self.method)
            method_param_names = [param.name for param in method.parameters]
        return method, method_param_names

    def __add_ids(self, df_parameters, base=None, pos=4, sep=''):
        if base is None:
            base = self.name
        if df_parameters is not None:
            names = [rf"{base}{sep}" + str(i).zfill(pos) for i in range(len(df_parameters))]
            df_parameters['ID'] = names
        return df_parameters

    def save_parameters(self, path: str = None):
        if path is not None:
            self.path = path

        if self.path is None:
            raise ValueError('Path must be provided!')
        else:
            if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != '':
                os.makedirs(os.path.dirname(path))
            self.parameters.to_csv(path, index=False)

    def load_parameters(self, path: str):
        self.parameters = pd.read_csv(path)
        return self.parameters

    def to_dict(self):
        step = dict()
        step['name'] = self.name
        step['number of inputs'] = self.n_inputs
        step['number of outputs'] = self.n_outputs
        if self.method is not None:
            step['method'] = self.method
            step['parameter_path'] = self.path
            step['number of parameter combinations'] = len(self.parameters)
            step['input step'] = self.input_step
            if len(self.parameters) == 0 and self.valid_parameters is not None:
                step['valid parameters'] = [str(p) for p in self.valid_parameters]
        else:
            step['available methods'] = self.available_methods
        return step

    def from_dict(self, step_dict):
        self.name = step_dict['name']
        self.method = step_dict['method']
        self.path = step_dict['parameter_path']
        self.n_inputs = step_dict['number of inputs']
        self.n_outputs = step_dict['number of outputs']
        self.input_step = step_dict['input step']
        if step_dict['parameter_path'] is not None:
            self.parameters = self.load_parameters(step_dict['parameter_path'])

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)
