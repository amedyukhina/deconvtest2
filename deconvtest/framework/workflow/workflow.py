import copy
import inspect
import itertools
import json
import os
import warnings
from time import sleep
from typing import Union

import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel
from skimage import io

from .step import Step
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
            # print(step.name, step.n_inputs, step.align)
            block = dict(name=rf'block{len(blocks):02d}', items=[])
            block = self.__add_items_to_block(step, block)

            if step.n_inputs > 0:
                if step.n_inputs == 2 and step.align is True:
                    pass
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
        img_filename_pattern = os.path.join(self.output_path, '%s.tif')
        stat_filename_pattern = os.path.join(self.output_path, '%s.csv')

        # store filenames for the outputs
        outputs = dict()
        for item in self.workflow['items']:
            steps = item['modules']
            for step_kwargs in steps:
                outputs[step_kwargs['outputID']] = img_filename_pattern % step_kwargs['outputID']

        # run the workflow in parallel
        run_parallel(
            process=run_item,
            print_progress=verbose,
            items=self.workflow['items'],
            max_threads=njobs,
            img_filename_pattern=img_filename_pattern,
            stat_filename_pattern=stat_filename_pattern,
            outputs=outputs
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
            item = dict(name=rf'item{i:02d}', modules=[module])
            block['items'].append(item)

        if len(step.parameters) == 0:
            module = dict(name=step.name, method=step.method)
            module['outputID'] = step.name + '0000'
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
                inputIDs.append(iter_item['steps'][-1]['outputID'])
            combined_block['items'].append(item)
            combined_block['items'][i]['modules'][-1]['inputIDs'] = inputIDs[:len(step.input_step)]
            combined_block['items'][i]['modules'][-1]['outputID'] = outputID.rstrip('_')
            # if step.name == 'Evaluation':
            #     nblock['items'][i]['steps'][-1]['outputID'] = inputIDs[0] + '_vs_' + inputIDs[1]
        blocks.append(combined_block)
        return blocks


def run_item(item, img_filename_pattern, stat_filename_pattern, outputs):
    steps = item['modules']
    for step_kwargs in steps:
        name = step_kwargs.pop('name')
        method = step_kwargs.pop('method')
        outputID = step_kwargs.pop('outputID')
        output_name = img_filename_pattern % outputID

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
                inputs = [outputs[inputID] for inputID in inputIDs]
                inputs = [io.imread(fn) for fn in inputs]
            else:
                inputs = []

            if name == 'Evaluation' and type(method) is list:
                output = []
                for m in method:
                    module = Step(name, m).module(method=m)
                    output.append(module.run(*inputs, **step_kwargs))
            else:
                module = Step(name, method).module(method=method)
                output = module.run(*inputs, **step_kwargs)

            if name == 'Evaluation':
                stat = pd.DataFrame({'OutputID': [outputID]})
                if type(method) is list:
                    for m, o in zip(method, output):
                        stat[m] = o
                else:
                    stat[method] = output
                stat.to_csv(stat_filename_pattern % outputID, index=False)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(output_name, output)
