from .constants import DEFAULT_PIPELINE_PARAM
import warnings


def raise_not_valid_method_error(method, module_name, available_methods):
    raise ValueError(rf'{method} is not a valid {module_name} method; available methods are: {available_methods}')


def raise_not_valid_type_error(provided_type, var_name, valid_types):
    raise TypeError(
        rf'{provided_type} is not a valid type for {var_name}; valid types are: {valid_types}')


def raise_not_valid_step_error(step_name, available_modules):
    raise ValueError(
        rf'{step_name} is not a valid step! Valid steps are: {[st[0].__name__ for st in available_modules]}')


def raise_mandatary_param_error(param_name):
    raise ValueError(rf'Parameter {param_name} is mandatory! '
                     f'\nIf the parameter is an input image that will be '
                     rf'generated in another step of the pipeline, specify "{DEFAULT_PIPELINE_PARAM}"')


def warn_param_not_in_list(param_name, method_name):
    warnings.warn(rf'Parameter "{param_name}" is not in the list of parameters for module "{method_name}"'
                  rf'and will not be included!')


def raise_not_valid_type_pipeline_error(provided_type, var_name, valid_types):
    raise TypeError(
        rf'{provided_type} is not a valid type for {var_name}; valid types are: {valid_types}'
        rf'\nIf the parameter is an input image that will be '
        rf'generated in another step of the pipeline, specify "{DEFAULT_PIPELINE_PARAM}"')

