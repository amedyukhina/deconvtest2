import inspect

from ...core.utils.utils import list_modules
from ...framework import step


class Step:
    """
    class for a workflow step
    """

    def __init__(self, step_name: str, method: str = None):
        self.step_name = step_name
        self.method = method

        steps = list_modules(step, module_type=inspect.isclass)
        self.step = None
        for st in steps:
            if st[0].__name__ == step_name:
                self.step = st[0]
        if self.step is None:
            raise ValueError(
                rf'{step_name} is not a valid step! Valid steps are: {[st[0].__name__ for st in steps]}')

    def __repr__(self):
        string = self.step_name
        if self.method is not None:
            string = string + ', ' + self.method
        return string
