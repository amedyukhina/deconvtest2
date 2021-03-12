import inspect

from ...core.utils.utils import list_modules
from ...framework import step
from .step import Step


def list_available_steps():
    steps = list_modules(step, module_type=inspect.isclass)
    steps = [Step(st[0].__name__) for st in steps]
    return steps


class Workflow:
    """
    Workflow class
    """
    def __init__(self, name : str = 'New Workflow'):
        self.name = name
        self.available_steps = list_available_steps()
        self.steps = []

    def add_step(self, name, method=None):
        step = None
        for st in self.available_steps:
            if st.name == name:
                step = st
        if step is None:
            raise ValueError(rf'{name} is not a valid step! Valid steps are {self.available_steps}')
        if method is not None:
            step.add_method(method)
        self.steps.append(step)



