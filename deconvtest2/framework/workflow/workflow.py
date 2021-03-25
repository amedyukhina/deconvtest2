import inspect
import json
import os
import numpy as np
from typing import Union

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

    def add_step(self, step: Step, input_step: Union[int, list] = None):
        if step.method is None:
            raise ValueError(rf"Step {step.name} does not have a method. "
                             "First specify the method by `step.add_method(method)`")
        if step.n_inputs > len(self.steps):
            raise IndexError(rf"Not enough previous steps in the workflow for step {step.name}."
                             rf"{len(self.steps)} were added; {step.n_inputs} are required.")

        if input_step is None:
            input_step = list(map(int, np.arange(len(self.steps)-step.n_inputs, len(self.steps))))
        else:
            if len(input_step) != step.n_inputs:
                raise ValueError(rf"Number of input steps must be {step.n_inputs}, {len(input_step)} provided")
            for st in input_step:
                if st >= len(self.steps):
                    raise IndexError(rf"{st} is invalid value for step index; must be < {len(self.steps)}")

        step.input_step = input_step
        # if step.n_inputs == 0:
        #     input_step = None
        # else:

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
