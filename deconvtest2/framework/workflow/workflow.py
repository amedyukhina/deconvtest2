import inspect
import json
import os

from .step import Step
from ...core.utils.utils import list_modules
from ...framework import step as workflow_steps


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
        self.path = None

    def add_step(self, step: Step):
        self.steps.append(step)

    def to_dict(self):
        workflow = dict()
        workflow['name'] = self.name
        workflow['path'] = self.path
        workflow['steps'] = [step.to_dict() for step in self.steps]
        return workflow

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def save(self, path: str = None):
        if path is not None:
            self.path = path

        if self.path is None:
            raise ValueError('Path must be provided!')
        else:
            if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != '':
                os.makedirs(os.path.dirname(path))
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f)

    def load(self, path: str):
        with open(path) as f:
            workflow = json.load(f)
        self.name = workflow['name']
        self.path = path
        self.steps = workflow['steps']
