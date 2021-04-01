import inspect
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd

from ...core.utils.conversion import list_to_keys
from ...core.utils.utils import list_modules, is_valid_type
from ...framework import step as workflow_steps


class Step:
    """
    class for a workflow step
    """

    def __init__(self, step_name: str, method: str = None):
        self.name = step_name
        self.parameters = pd.DataFrame()
        self.path = None
        self.n_inputs = None
        self.n_outputs = None
        self.method = None
        self.step = None
        self.input_step = None

        steps = list_modules(workflow_steps, module_type=inspect.isclass)
        for st in steps:
            if st[0].__name__ == step_name:
                self.step = st[0]

        if self.step is None:
            raise ValueError(
                rf'{step_name} is not a valid step! Valid steps are: {[st[0].__name__ for st in steps]}')

        if method is not None:
            self.add_method(method)

    def to_dict(self):
        step = dict()
        step['name'] = self.name
        step['method'] = self.method
        step['parameter_path'] = self.path
        step['number of parameter combinations'] = len(self.parameters)
        step['number of inputs'] = self.n_inputs
        step['number of outputs'] = self.n_outputs
        step['input step'] = self.input_step
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

    def list_available_methods(self):
        module = self.step()
        return module.list_available_modules_names()

    def add_method(self, method: str):
        if method in self.list_available_methods():
            self.method = method
            module = self.step(self.method)
            self.n_inputs = module.n_inputs
            self.n_outputs = module.n_outputs
        else:
            raise ValueError(rf'{method} is not a valid method! Valid methods are: {self.list_available_methods()}')

    def list_parameters(self):
        if self.method is None:
            warnings.warn(rf'No method is defined for step {self.name} to list parameters!')
            return None
        else:
            module = self.step(self.method)
            return module.parameters

    def get_method(self):
        if self.method is None:
            raise ModuleNotFoundError(rf'No method is defined for step {self.name} to specify parameters!')
        else:
            module = self.step(self.method)
            module_param_names = [param.name for param in module.parameters]
        return module, module_param_names

    def __get_parameter_lists(self, parameters):
        module, module_param_names = self.get_method()

        for key in parameters.keys():
            if key not in module_param_names:
                warnings.warn(rf'Parameter "{key}" is not in the list of parameters for module "{self.method}"'
                              rf'and will not be included!')

        param_values_list = dict()
        param_values_single = dict()
        for param in module.parameters:
            if param.name not in parameters.keys():
                if param.optional:
                    param_values_single[param.name] = param.default_value
                else:
                    raise ValueError(rf'Parameter {param.name} is mandatory! '
                                     '\nIf the parameter is an input image that will be '
                                     'generated in another step of the pipeline, specify "pipeline"')

            elif is_valid_type(np.ones([2, 2, 2]), param.type) and type(parameters[param.name]) is str and \
                    parameters[param.name] == 'pipeline':
                continue
            else:
                if type(parameters[param.name]) in [list, np.ndarray]:
                    is_list = True
                    for param_value in parameters[param.name]:
                        if not is_valid_type(param_value, param.type):
                            raise ValueError(rf'{type(param_value)} is not a valid type for {param.name}; '
                                             f'valid types are: {param.type}.'
                                             '\nIf the parameter is an input image that will be '
                                             'generated in another step of the pipeline, specify "pipeline"'
                                             )
                else:
                    is_list = False
                    if is_valid_type(parameters[param.name], param.type):
                        parameters[param.name] = [parameters[param.name]]
                    else:
                        raise ValueError(rf'{type(parameters[param.name])} is not a valid type for {param.name}; '
                                         f'valid types are: {param.type}.'
                                         '\nIf the parameter is an input image that will be '
                                         'generated in another step of the pipeline, specify "pipeline"'
                                         )

                if is_valid_type([], param.type) and len(parameters[param.name]) <= 3:
                    warnings.warn(rf'Since list is a valid type for parameter {param.name} and '
                                  rf'the number of provided values is <= 3, '
                                  rf'values {parameters[param.name]} will be assumed as '
                                  rf'{len(parameters[param.name])} different values.'
                                  rf'To specify one value for different dimensions, wrap them up in another list, e.g:'
                                  rf'[[value_dim1, value_dim2, value_dim2]]')
                if is_list:
                    param_values_list[param.name] = parameters[param.name]
                else:
                    param_values_single[param.name] = parameters[param.name][0]

        return param_values_list, param_values_single

    def __get_param_table(self, parameters, mode):
        param_values_list, param_values_single = self.__get_parameter_lists(parameters)

        if mode == 'align':
            df_parameters = pd.DataFrame()
            length = len(param_values_list[list(param_values_list.keys())[0]])
            for key in param_values_list.keys():
                if not len(param_values_list[key]) == length:
                    raise ValueError(rf'{length}!={len(param_values_list[key])}. '
                                     'Lengths of parameter lists for mode "align" must be equal!')
                df_parameters[key] = param_values_list[key]
        else:
            values = np.array(list(itertools.product(*list(param_values_list.values()))))
            df_parameters = pd.DataFrame(values, columns=param_values_list.keys())

        param_values_single = list_to_keys(param_values_single)
        for key in param_values_single.keys():
            df_parameters[key] = param_values_single[key]

        return df_parameters

    def __add_ids(self, df_parameters, base=None, pos=4, sep=''):
        if base is None:
            base = self.name
        if df_parameters is not None:
            names = [rf"{base}{sep}" + str(i).zfill(pos) for i in range(len(df_parameters))]
            df_parameters['ID'] = names
        return df_parameters

    def specify_parameters(self, mode: str = 'permute', overwrite: bool = True,
                           base_name: str = None, sep: str = '', pos: int = 4,
                           **parameters):
        """
        Specify the list of parameters for the step.

        Parameters
        ----------
        mode : str, optional
            'permute' or 'align'
            If 'align', the parameter values for each parameter will be aligned.
            If 'permute', the combination of all possible parameter values will be generated.
            For 'align', the list of values for each parameter must have the same length.
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
            Table with parameter values
        """
        if mode not in ['align', 'permute']:
            raise ValueError(rf'{mode} is not a valid mode; must be "align" or "permute"')

        df_parameters = self.__get_param_table(parameters, mode)
        df_parameters = self.__add_ids(df_parameters, base=base_name, pos=pos, sep=sep)

        if overwrite:
            self.parameters = pd.DataFrame()

        self.parameters = pd.concat([self.parameters, df_parameters], ignore_index=True)
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
