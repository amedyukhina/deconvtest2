import importlib
from deconvtest2.core.utils.utils import list_modules


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

