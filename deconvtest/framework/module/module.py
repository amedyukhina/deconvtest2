import importlib

from ..module.parameter import Parameter
from ...core.utils.utils import list_modules, is_valid_type


class Module:
    """
    Abstract module class
    """

    def __init__(self, method: str = None, parameters: dict = None, parent_name: str = 'deconvtest.modules'):
        if parameters is None:
            self.parameter_values = dict()
        else:
            self.parameter_values = parameters

        self.parent_name = parent_name
        self.method = None
        self.arg_spec = None
        self.parameters = []
        self.result = None
        self.n_inputs = None
        self.n_outputs = None
        self.inputs = None
        self.align = False

        if method is not None:
            self.import_method(method)
        if self.arg_spec is not None:
            self.add_parameters(self.arg_spec)

    def list_available_modules(self):
        parent_module = importlib.import_module(self.parent_name)
        available_modules = list_modules(parent_module)
        return available_modules

    def list_available_modules_names(self):
        available_modules = self.list_available_modules()
        return [module[0].__name__ for module in available_modules]

    def import_method(self, method):
        available_modules = self.list_available_modules()
        for module in available_modules:  # find a module with a matching name
            if module[0].__name__ == method:
                self.method = module[0]
                self.arg_spec = module[1]

        if self.method is None:  # raise an error if no matching module found
            modules = self.list_available_modules_names()
            raise ValueError(rf'{method} is not a valid {self.parent_name} module; available modules are: {modules}')

    def add_parameters(self, arg_spec):
        names = arg_spec.args
        defaults = arg_spec.defaults
        if defaults is None:
            defaults = []
        types = arg_spec.annotations
        n_non_optional_parameters = len(names) - len(defaults)
        self.parameters = []
        for i in range(len(names)):
            if i < n_non_optional_parameters:
                optional = False
                default = None
            else:
                optional = True
                default = defaults[i - n_non_optional_parameters]
            parameter_type = None
            if names[i] in types.keys():
                parameter_type = types[names[i]]
            self.parameters.append(Parameter(name=names[i],
                                             default_value=default,
                                             optional=optional,
                                             parameter_type=parameter_type))

    def verify_parameters(self):
        missing_param = 0
        for parameter in self.parameters:
            if parameter.name in self.parameter_values.keys():
                if not is_valid_type(self.parameter_values[parameter.name], parameter.type):
                    raise ValueError(
                        rf'{type(self.parameter_values[parameter.name])} is not a valid type for {parameter.name}; '
                        f'valid types are: {parameter.type}')

            else:
                # add default value if available, otherwise raise error
                if parameter.optional is True:
                    self.parameter_values[parameter.name] = parameter.default_value
                elif self.inputs is not None and len(self.inputs) > 0:
                    missing_param += 1
                else:
                    raise ValueError(rf'Parameter `{parameter.name}` is mandatory, please provide a value!')
        if missing_param > 0 and missing_param != len(self.inputs):
            raise ValueError(rf'Number of inputs to {self.method} must be {missing_param}, '
                             rf'{len(self.inputs)} provided.')

    def run(self, *inputs, **parameters):
        self.parameter_values = parameters
        self.inputs = inputs
        self.verify_parameters()
        self.result = self.method(*self.inputs, **self.parameter_values)
        return self.result
