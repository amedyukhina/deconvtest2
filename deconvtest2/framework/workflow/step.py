import inspect
import warnings

from ...core.utils.utils import list_modules
from ...framework import step as workflow_steps


class Step:
    """
    class for a workflow step
    """

    def __init__(self, step_name: str, method: str = None):
        self.name = step_name
        self.method = method

        steps = list_modules(workflow_steps, module_type=inspect.isclass)
        self.step = None
        for st in steps:
            if st[0].__name__ == step_name:
                self.step = st[0]

        if self.step is None:
            raise ValueError(
                rf'{step_name} is not a valid step! Valid steps are: {[st[0].__name__ for st in steps]}')

    def __repr__(self):
        string = self.name
        if self.method is not None:
            string = string + ', ' + self.method
        return string

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
