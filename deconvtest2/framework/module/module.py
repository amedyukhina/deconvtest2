import importlib
from deconvtest2.core.utils.utils import list_modules
from ..module.parameter import Parameter


class Module:
    """
    Abstract module class
    """
    def __init__(self, method: str, parameters: dict = None, parent_name: str = 'deconvtest2.modules'):
        if parameters is None:
            self.parameters = dict()
        else:
            self.parameters = parameters

        self.parent_name = parent_name
        self.method = None
        self.arg_spec = None

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


