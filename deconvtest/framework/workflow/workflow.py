import copy
import inspect
import itertools
import json
import os
from time import sleep
from typing import Union

import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel

from .step import Step
from ...core.utils import io
from ...core.utils.constants import EXTENSIONS
from ...core.utils.conversion import keys_to_list
from ...core.utils.utils import list_modules
from ...framework import module as available_steps


def list_available_steps():
    steps = list_modules(available_steps, module_type=inspect.isclass)
    steps = np.unique([st[0].__name__ for st in steps])
    steps = [Step(st) for st in steps]
    return steps


class Workflow:
    """
    Workflow class
    """

    def __init__(self, name: str = 'New Workflow', output_path: str = None):
        self.name = name
        self.available_steps = list_available_steps()
        self.steps = []
        self.filename = None
        self.workflow = None
        self.output_path = name.replace(' ', '_')
        if output_path is not None:
            self.output_path = output_path

    def add_step(self, step: Step, input_step: Union[int, list] = None):
        if step.method is None and len(step.methods) == 0:
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
        workflow['filename'] = self.filename
        workflow['output path'] = self.output_path
        workflow['steps'] = [step.to_dict() for step in self.steps]
        return workflow

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def save(self, filename: str = None):
        if filename is not None:
            self.filename = filename

        if self.filename is None:
            raise ValueError('Path must be provided!')
        else:
            if not os.path.exists(os.path.dirname(filename)) and os.path.dirname(filename) != '':
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f)

    def load(self, filename: str):
        with open(filename) as f:
            workflow = json.load(f)
        self.name = workflow['name']
        self.filename = filename
        self.output_path = workflow['output path']
        self.steps = []
        for step in workflow['steps']:
            s = Step(step['name'])
            s.from_dict(step)
            self.steps.append(s)

    def get_workflow_graph(self, to_json=False):
        blocks = []
        for step in self.steps:
            block = dict(name=rf'block{len(blocks):02d}', items=[])
            block = self.__add_items_to_block(step, block)

            if step.n_inputs > 0:
                if step.n_inputs == 2 and step.align is True:
                    blocks = self.__align_items(step, block, blocks)
                else:
                    blocks = self.__permute_items(step, block, blocks)
            else:
                blocks.append(block)
        self.workflow = blocks[-1]
        self.workflow['name'] = 'workflow_graph'
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

    def run(self, njobs=8, verbose=True):
        if self.workflow is None:
            self.get_workflow_graph()
        os.makedirs(self.output_path, exist_ok=True)

        # run the workflow in parallel
        run_parallel(
            process=run_item,
            process_name='Running the workflow',
            print_progress=verbose,
            items=self.workflow['items'],
            max_threads=njobs,
            output_path=self.output_path
        )

        stats = pd.DataFrame()
        for fn in os.listdir(self.output_path):
            if fn.endswith('csv'):
                stats = pd.concat([stats, pd.read_csv(os.path.join(self.output_path, fn))], ignore_index=True)
        stats.to_csv(os.path.join(self.output_path, '..', self.name + '.csv'), index=False)

    def __add_params_to_module(self, params, module):
        params = dict(params)
        params = keys_to_list(params)
        for key in params.keys():
            try:
                module[key] = params[key].item()
            except AttributeError:
                module[key] = params[key]
        return module

    def __add_items_to_block(self, step, block):
        for i in range(len(step.parameters)):
            module = dict(name=step.name, method=step.method)
            module = self.__add_params_to_module(step.parameters.iloc[i], module)
            module['outputID'] = module.pop('ID')
            module['type_output'] = step.type_output
            module['type_input'] = step.type_input
            item = dict(name=rf'item{i:02d}', modules=[module])
            block['items'].append(item)

        if len(step.parameters) == 0:
            module = dict(name=step.name, method=step.method)
            module['outputID'] = step.name + '0000'
            module['type_output'] = step.type_output
            module['type_input'] = step.type_input
            item = dict(name=rf'item00', modules=[module])
            block['items'].append(item)
        return block

    def __permute_items(self, step, new_block, blocks):
        lists = []
        for input_step in step.input_step:
            lists.append(blocks[input_step]['items'])
        if len(new_block['items']) > 0:
            lists.append(new_block['items'])

        combined_block = dict(name=rf'updated_block{len(blocks):02d}')
        combined_block['items'] = []
        for i, items in enumerate(itertools.product(*lists)):
            item = dict(name=rf'item{i:03d}')
            item['modules'] = []
            outputID = ''
            inputIDs = []
            for iter_item in items:
                for st in iter_item['modules']:
                    item['modules'].append(copy.deepcopy(st))
                outputID = outputID + iter_item['modules'][-1]['outputID'] + '_'
                inputIDs.append(iter_item['modules'][-1]['outputID'])
            combined_block = self.__add_ids(combined_block, i, item, inputIDs, outputID, step)
        blocks.append(combined_block)
        return blocks

    def __align_items(self, step, new_block, blocks):
        lists = []
        for input_step in step.input_step:
            lists.append(blocks[input_step]['items'])
        if len(new_block['items']) > 0:
            lists.append(new_block['items'])
        i = 0
        combined_block = dict(name=rf'updated_block{len(blocks):02d}')
        combined_block['items'] = []
        for items in itertools.product(*lists):
            if items[0]['modules'][0]['outputID'] == items[1]['modules'][0]['outputID']:
                item = dict(name=rf'item{i:03d}')
                item['modules'] = []
                for st in items[1]['modules']:
                    item['modules'].append(copy.deepcopy(st))
                item['modules'].append(copy.deepcopy(items[2]['modules'][0]))

                outputID = items[1]['modules'][-1]['outputID'] + '_' + items[2]['modules'][-1]['outputID']
                inputIDs = [items[0]['modules'][-1]['outputID'], items[1]['modules'][-1]['outputID']]
                combined_block = self.__add_ids(combined_block, i, item, inputIDs, outputID, step)

                i += 1
        blocks.append(combined_block)
        return blocks

    def __add_ids(self, block, i, item, inputIDs, outputID, step):
        block['items'].append(item)
        block['items'][i]['modules'][-1]['inputIDs'] = inputIDs[:len(step.input_step)]
        block['items'][i]['modules'][-1]['outputID'] = outputID.rstrip('_')
        return block


def run_item(item, output_path):
    steps = item['modules']
    for step_kwargs in steps:
        name = step_kwargs.pop('name')
        method = step_kwargs.pop('method')
        outputID = step_kwargs.pop('outputID')
        type_output = step_kwargs.pop('type_output')
        if 'type_input' in step_kwargs:
            type_input = step_kwargs.pop('type_input')
            if not type(type_input) is list:
                type_input = [type_input]
        else:
            type_input = []
        output_name = os.path.join(output_path, outputID + EXTENSIONS[type_output])

        # if output exists, wait until file size stops increasing
        if os.path.exists(output_name):
            size1 = os.path.getsize(output_name)
            sleep(0.1)
            size2 = os.path.getsize(output_name)
            while size2 > size1:
                size1 = size2
                sleep(0.1)
                size2 = os.path.getsize(output_name)
        else:
            if "inputIDs" in step_kwargs:
                inputIDs = step_kwargs.pop('inputIDs')
                input_fns = [os.path.join(output_path, inputIDs[i] + EXTENSIONS[type_input[i]])
                             for i in range(len(inputIDs))]
                inputs = [io.read(input_fns[i], type_input[i]) for i in range(len(input_fns))]
            else:
                inputs = []

            if name == 'Evaluation' and type(method) is list:
                output = pd.DataFrame({'OutputID': [outputID]})
                for m in method:
                    module = Step(name, m).module(method=m)
                    output[m] = module.run(*inputs, **step_kwargs)
            else:
                module = Step(name, method).module(method=method)
                output = module.run(*inputs, **step_kwargs)

            io.write(output_name, output, type_output)
