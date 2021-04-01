import copy
import inspect
import itertools
import json
import os
from typing import Union

import numpy as np

from .step import Step
from ...core.utils.conversion import keys_to_list
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
        self.workflow = None

    def add_step(self, step: Step, input_step: Union[int, list] = None):
        if step.method is None:
            raise ValueError(rf"Step {step.name} does not have a method. "
                             "First specify the method by `step.add_method(method)`")
        if step.n_inputs > len(self.steps):
            raise IndexError(rf"Not enough previous steps in the workflow for step {step.name}."
                             rf"{len(self.steps)} were added; {step.n_inputs} are required.")

        if input_step is None:
            input_step = list(map(int, np.arange(len(self.steps) - step.n_inputs, len(self.steps))))
        else:
            if len(input_step) != step.n_inputs:
                raise ValueError(rf"Number of input steps must be {step.n_inputs}, {len(input_step)} provided")
            for st in input_step:
                if st >= len(self.steps):
                    raise IndexError(rf"{st} is invalid value for step index; must be < {len(self.steps)}")

        step.input_step = input_step
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
        self.steps = []
        for step in workflow['steps']:
            s = Step(step['name'])
            s.from_dict(step)
            self.steps.append(s)

    def get_workflow_graph(self, to_json=False):
        blocks = []
        for master_step in self.steps:
            if master_step.name != 'Evaluation':
                block = dict(name=rf'block{len(blocks):02d}')
                block['items'] = []
                for i in range(len(master_step.parameters)):
                    item = dict(name=rf'item{i:02d}')
                    item['steps'] = []
                    step = dict(name=master_step.name, method=master_step.method)
                    params = dict(master_step.parameters.iloc[i])
                    params = keys_to_list(params)
                    for key in params.keys():
                        try:
                            step[key] = params[key].item()
                        except AttributeError:
                            step[key] = params[key]
                    step['outputID'] = step.pop('ID')
                    item['steps'].append(step)
                    block['items'].append(item)
                if len(master_step.parameters) == 0:
                    item = dict(name=rf'item00')
                    item['steps'] = []
                    step = dict(name=master_step.name, method=master_step.method)
                    step['outputID'] = ''
                    item['steps'].append(step)
                    block['items'].append(item)

                if master_step.n_inputs > 0:
                    lists = []
                    for input_step in master_step.input_step:
                        lists.append(blocks[input_step]['items'])
                    if len(block['items']) > 0:
                        lists.append(block['items'])

                    nblock = dict(name=rf'updated_block{len(blocks):02d}')
                    nblock['items'] = []
                    for i, items in enumerate(itertools.product(*lists)):
                        item = dict(name=rf'item{i:03d}')
                        item['steps'] = []
                        outputID = ''
                        for iter_item in items:
                            for st in iter_item['steps']:
                                item['steps'].append(copy.deepcopy(st))
                            outputID = outputID + iter_item['steps'][-1]['outputID'] + '_'
                        nblock['items'].append(item)
                        nblock['items'][i]['steps'][-1]['outputID'] = outputID.rstrip('_')
                    blocks.append(nblock)
                else:
                    blocks.append(block)
        self.workflow = blocks[-1]
        if to_json:
            return json.dumps(self.workflow, indent=4)
        else:
            return self.workflow

    def save_workflow_graph(self, path):
        if path is not None:
            self.path = path

        if self.path is None:
            raise ValueError('Path must be provided!')
        else:
            workflow = self.get_workflow_graph()
            with open(path, 'w') as f:
                json.dump(workflow, f)
