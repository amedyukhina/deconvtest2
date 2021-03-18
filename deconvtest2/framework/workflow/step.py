import inspect
import warnings
import itertools
import json

import pandas as pd
import numpy as np
import os

from ...core.utils.utils import list_modules, is_valid_type
from ...core.utils.conversion import list_to_keys
from ...framework import step as workflow_steps


class Step:
    """
    class for a workflow step
    """

    def __init__(self, step_name: str, method: str = None):
        self.name = step_name
        self.method = method
        self.parameters = pd.DataFrame()
        self.path = None

        steps = list_modules(workflow_steps, module_type=inspect.isclass)
        self.step = None
        for st in steps:
            if st[0].__name__ == step_name:
                self.step = st[0]

        if self.step is None:
            raise ValueError(
                rf'{step_name} is not a valid step! Valid steps are: {[st[0].__name__ for st in steps]}')

    def to_dict(self):
        step = dict()
        step['name'] = self.name
        step['method'] = self.method
        step['parameter_path'] = self.path
        step['number of parameter combinations'] = len(self.parameters)
        return step

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def list_available_methods(self):
        module = self.step()
        return module.list_available_modules_names()

    def add_method(self, method: str):
        if method in self.list_available_methods():
            self.method = method
        else:
            raise ValueError(rf'{method} is not a valid method! Valid methods are: {self.list_available_methods()}')

    def list_parameters(self):
        if self.method is None:
            warnings.warn(rf'No method is defined for step {self.name} to list parameters!')
            return None
        else:
            module = self.step(self.method)
            return module.parameters

    def specify_parameters(self, mode: str = 'permute', overwrite: bool = True, **parameters):
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
        if self.method is None:
            raise ModuleNotFoundError(rf'No method is defined for step {self.name} to specify parameters!')
        else:
            module = self.step(self.method)
            module_param_names = [param.name for param in module.parameters]
        if overwrite:
            self.parameters = pd.DataFrame()

        for key in parameters.keys():
            if key not in module_param_names:
                warnings.warn(rf'Parameter "{key}" is not in the list of parameters for module "{self.method}"'
                              rf'and will not be included!')

        param_values_list = dict()
        param_values_single = dict()
        for param in module.parameters:
            if param.name not in parameters.keys():
                if param.optional is False:
                    raise ValueError(rf'Parameter {param.name} is mandatory!')
                else:
                    param_values_single[param.name] = param.default_value
            else:
                if type(parameters[param.name]) in [list, np.ndarray]:
                    is_list = True
                    for param_value in parameters[param.name]:
                        if not is_valid_type(param_value, param.type):
                            raise ValueError(rf'{type(param_value)} is not a valid type for {param.name}; '
                                             f'valid types are: {param.type}')
                else:
                    is_list = False
                    if not is_valid_type(parameters[param.name], param.type):
                        raise ValueError(rf'{type(parameters[param.name])} is not a valid type for {param.name}; '
                                         f'valid types are: {param.type}')

                if is_valid_type([], param.type) and len(parameters[param.name]) <= 3:
                    warnings.warn(rf'Since list is a valid type for parameter {param.name} and '
                                  rf'the number of provided values is <= 3, '
                                  rf'values {parameters[param.name]} will be assumed as'
                                  rf'{len(parameters[param.name])} different values.'
                                  rf'To specify one value for different dimensions, wrap them up in another list, e.g:'
                                  rf'[[value_dim1, value_dim2, value_dim2]]')
                if is_list:
                    param_values_list[param.name] = parameters[param.name]
                else:
                    param_values_single[param.name] = parameters[param.name]

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

        self.parameters = pd.concat([self.parameters, df_parameters], ignore_index=True)
        return df_parameters

    def add_to_parameter_table(self, **parameters):
        """
        Add parameter values to existing parameter table.

        Parameters
        ----------
        parameters : key value
            Parameter names and values.

        Returns
        -------
        pandas.DataFrame:
            Updated parameter table
        """
        if self.method is None:
            raise ModuleNotFoundError(rf'No method is defined for step {self.name} to specify parameters!')
        else:
            module = self.step(self.method)
            module_param_names = [param.name for param in module.parameters]

        for key in parameters.keys():
            if key not in module_param_names:
                warnings.warn(rf'Parameter "{key}" is not in the list of parameters for module "{self.method}"'
                              rf'and will not be included!')
        for param in module.parameters:
            if param.name not in parameters.keys():
                if param.optional is False:
                    raise ValueError(rf'Parameter {param.name} is mandatory!')
                else:
                    parameters[param.name] = param.default_value
            else:
                if not is_valid_type(parameters[param.name], param.type):
                    raise ValueError(rf'{type(parameters[param.name])} is not a valid type for {param.name}; '
                                     f'valid types are: {param.type}')

        parameters = list_to_keys(parameters)
        df_parameters = pd.DataFrame()
        for key in parameters.keys():
            df_parameters[key] = [parameters[key]]

        self.parameters = pd.concat([self.parameters, df_parameters], ignore_index=True)
        return self.parameters

    def save_parameters(self, path: str = None):
        if path is not None:
            self.path = path

        if self.path is None:
            raise ValueError('Path must be provided!')
        else:
            if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != '':
                os.makedirs(os.path.dirname(path))
            self.parameters.to_csv(path, index=False)



