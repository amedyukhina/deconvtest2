import importlib
from deconvtest2.core.utils.utils import list_modules
from ..module.parameter import Parameter


class Module:
    """
    Abstract module class
    """
    def __init__(self, method: str, parameters: dict = None, parent_name: str = 'deconvtest2.modules'):
        if parameters is None:
            self.parameter_values = dict()
        else:
            self.parameter_values = parameters

        self.parent_name = parent_name
        self.method = None
        self.arg_spec = None
        self.parameters = []

        self.import_method(method)
        if self.arg_spec is not None:
            self.add_parameters(self.arg_spec)

    def import_method(self, method):
        parent_module = importlib.import_module(self.parent_name)
        available_modules = list_modules(parent_module)
        for module in available_modules:  # find a module with a matching name
            if module[0].__name__ == method:
                self.method = module[0]
                self.arg_spec = module[1]

        if self.method is None:  # raise an error if no matching module found
            modules = [module[0].__name__ for module in available_modules]
            raise ValueError('{} is not a valid {} module; available modules are: {}'.format(method,
                                                                                             self.parent_name,
                                                                                             modules))

    def add_parameters(self, arg_spec):
        names = arg_spec.args
        defaults = arg_spec.defaults
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
        for parameter in self.parameters:
            if parameter.name in self.parameter_values.keys():
                if type(parameter.type) is type:
                    valid_types = [parameter.type]
                else:
                    valid_types = list(parameter.type.__args__)
                if not type(self.parameter_values[parameter.name]) in valid_types:
                    raise ValueError('{} is not a valid type for {}; '
                                     'valid types are: {}'.format(type(self.parameter_values[parameter.name]),
                                                                  parameter.name,
                                                                  valid_types))

            else:
                # add default value if available, otherwise raise error
                if parameter.optional is True:
                    self.parameter_values[parameter.name] = parameter.default_value
                else:
                    raise ValueError('Parameter `{}` is mandatory, please provide a value!'.format(parameter.name))

    def run(self, **parameters):
        self.parameter_values = parameters
        self.verify_parameters()
        return self.method(**self.parameter_values)

