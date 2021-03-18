import inspect

from ...core.utils.utils import list_modules
from ...framework import step as workflow_steps
from .step import Step


def list_available_steps():
    steps = list_modules(workflow_steps, module_type=inspect.isclass)
    steps = [Step(st[0].__name__) for st in steps]
    return steps


class Workflow:
    """
    Workflow class
    """
    def __init__(self, name: str = 'New Workflow'):
        self.name = name
        self.available_steps = list_available_steps()
        self.steps = []

    def add_step(self, step: Step):
        self.steps.append(step)



