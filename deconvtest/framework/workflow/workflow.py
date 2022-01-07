import copy
import inspect
import itertools
import json
import os
import time
import warnings
from time import sleep
from typing import Union

import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel

from .step import Step
from ...core.utils import io
from ...core.utils.constants import *
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

    def __init__(self, name: str = None, path: str = None):
        if name is None:
            name = rf"{DEFAULT_WORKFLOW_NAME}_{int(time.time())}"
        self.name = name
        self.available_steps = list_available_steps()
        self.steps = []
        self.workflow_graph = None
        if path is not None:
            self.path = path
        else:
            self.path = self.name.replace(' ', '_')
        self.path = os.path.abspath(self.path)

    def add_step(self, step_name: str, method: Union[str, list] = None,
                 input_step: Union[int, list] = None, parmeter_mode: str = 'permute',
                 **parameters):
        step = Step(step_name, method)
        if step.method is None and len(step.methods) == 0:
            raise ValueError(rf"Step {step.name} does not have a method. "
                             "First specify the method by `step.add_method(method)`")
        if step.n_inputs > len(self.steps):
            raise IndexError(rf"Not enough previous steps in the workflow for step {step.name}."
                             rf"{len(self.steps)} were added; {step.n_inputs} are required.")
        if len(parameters.keys()) > 0:
            step.specify_parameters(mode=parmeter_mode, base_name=rf"{len(self.steps):02d}_", **parameters)
        else:
            step.parameters = pd.DataFrame()
            warnings.warn(rf'No parameters were defined for this step! Adding empty list')

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
        if step.wait_complete:
            previous_step = self.steps[input_step[0]]
            limiting_step = self.steps[previous_step.input_step[0]]
            step.parameters['min_inputs'] = len(limiting_step.parameters)
        self.steps.append(step)

    def to_dict(self):
        workflow = dict()
        workflow['name'] = self.name
        workflow['path'] = self.path
        workflow['steps'] = [step.to_dict() for step in self.steps]
        return workflow

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.path, WORKFLOW_FN)
        for i, step in enumerate(self.steps):
            step.save_parameters(os.path.join(self.path,
                                              PARAMETER_FOLDER_NAME,
                                              rf"{i:02d}_{step.name}_{step.method}.csv"))

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def load(self, filename: str):
        with open(filename) as f:
            workflow = json.load(f)
        self.name = workflow['name']
        self.path = workflow['path']
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
        self.workflow_graph = blocks[-1]
        self.workflow_graph['name'] = self.name
        # self.__add_module_ids()

        if to_json:
            return json.dumps(self.workflow_graph, indent=4)
        else:
            return self.workflow_graph

    def save_workflow_graph(self, path=None):
        if path is None:
            path = os.path.join(self.path, WORKFLOW_GRAPH_FN)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        workflow = self.get_workflow_graph()
        with open(path, 'w') as f:
            json.dump(workflow, f)

    def run(self, njobs=8, verbose=True, nsteps=None):
        if self.workflow_graph is None:
            self.get_workflow_graph()
        self.save_workflow_graph()
        output_path = os.path.join(self.path, DATA_FOLDER_NAME)
        os.makedirs(output_path, exist_ok=True)

        # run the workflow in parallel
        run_parallel(
            process=self.__run_item,
            process_name='Running the workflow',
            print_progress=verbose,
            items=self.workflow_graph['items'],
            max_threads=njobs,
            **dict(output_path=output_path, nsteps=nsteps)
        )

        stats = pd.DataFrame()
        for fn in os.listdir(output_path):
            if fn.endswith('csv'):
                stats = pd.concat([stats, pd.read_csv(os.path.join(output_path, fn))], ignore_index=True)
        stats.to_csv(os.path.join(self.path, self.name + '_results.csv'), index=False)

    def __run_item(self, item, output_path, nsteps=None):
        steps = item['item_steps']
        workflow_steps = [step.name for step in self.steps]
        if nsteps is not None:
            workflow_steps = workflow_steps[:nsteps]
        steps = [step for step in steps if step['name'] in workflow_steps]
        for step_kwargs in steps:
            step = self.steps[int(step_kwargs['ID'].split('_')[0])]
            name = step_kwargs.pop('name')
            method = step_kwargs.pop('method')
            parameters = step.parameters
            step_kwargs = self.__add_params_to_module(parameters[parameters['ID'] == step_kwargs['ID']].iloc[0],
                                                      step_kwargs)
            step_kwargs.pop('ID')
            outputID = step_kwargs.pop('outputID')
            type_output = step.type_output
            type_input = step.type_input
            if not type(type_input) is list:
                type_input = [type_input]
            output_name = os.path.join(output_path, outputID + EXTENSIONS[type_output])

            lock_file = outputID
            if 'inputIDs' in step_kwargs:
                lock_file = '_'.join(step_kwargs['inputIDs']) + '_' + lock_file
            lock_file = os.path.join(output_path, lock_file + '.lock')
            np.random.seed()
            sleep(np.random.rand())
            if os.path.exists(lock_file) or \
                    (os.path.exists(
                        output_name) and name != 'Organize'):  # check if the step is in process or completed
                while os.path.exists(lock_file):  # wait if the step is in process
                    sleep(1)
            else:
                with open(lock_file, 'w') as f:
                    pass
                if "inputIDs" in step_kwargs:
                    inputIDs = step_kwargs.pop('inputIDs')
                    input_fns = [os.path.join(output_path, inputIDs[i] + EXTENSIONS[type_input[i]])
                                 for i in range(len(inputIDs))]
                    inputs = [io.read(input_fns[i], type_input[i]) for i in range(len(input_fns))]
                else:
                    inputs = []
                    input_fns = []

                if name == 'Organize':
                    step_kwargs['img_name'] = input_fns[0][len(output_path) + 1:]

                if io.WRITE_FN[type_output] == io.write_file:
                    step_kwargs['fn_output'] = os.path.join(output_path, outputID)

                if name == 'Evaluation' and type(method) is list:
                    output = pd.DataFrame({'OutputID': [outputID]})
                    for m in method:
                        module = Step(name, m).module(method=m)
                        output[m] = module.run(*inputs, **step_kwargs)
                else:
                    module = Step(name, method).module(method=method)
                    output = module.run(*inputs, **step_kwargs)

                io.write(output_name, output, type_output)
                os.remove(lock_file)

    def __add_params_to_module(self, params, module):
        params = dict(params)
        params = keys_to_list(params)
        for key in params.keys():
            try:
                if type(params[key]) in [list, np.array]:
                    module[key] = [k.item() for k in params[key]]
                else:
                    module[key] = params[key].item()
            except AttributeError:
                module[key] = params[key]
        return module

    def __add_items_to_block(self, step, block):
        for i in range(len(step.parameters)):
            module = dict(name=step.name, method=step.method)
            module['ID'] = module['outputID'] = step.parameters.iloc[i]['ID']

            item = dict(name=rf'item{i:02d}', item_steps=[module])
            block['items'].append(item)

        if len(step.parameters) == 0:
            module = dict(name=step.name, method=step.method)
            item = dict(name=rf'item00', item_steps=[module])
            block['items'].append(item)
        return block

    def __gen_lists(self, step, blocks, new_block):
        lists = []
        for input_step in step.input_step:
            lists.append(blocks[input_step]['items'])
        if len(new_block['items']) > 0:
            lists.append(new_block['items'])
        return lists

    def __permute_items(self, step, new_block, blocks):
        lists = self.__gen_lists(step, blocks, new_block)

        combined_block = dict(name=rf'updated_block{len(blocks):02d}')
        combined_block['items'] = []
        for i, items in enumerate(itertools.product(*lists)):
            item = dict(name=rf'item{i:03d}')
            item['item_steps'] = []
            inputIDs = []
            outputID = rf"{items[-1]['item_steps'][-1]['outputID'].split('_')[0]}_"  # the step number
            for iter_item in items:
                for st in iter_item['item_steps']:
                    item['item_steps'].append(copy.deepcopy(st))
                outputID = outputID + iter_item['item_steps'][-1]['outputID'].split('_')[-1]
                inputIDs.append(iter_item['item_steps'][-1]['outputID'])
            # outputID = '_'.join(inputIDs)
            combined_block = self.__add_ids(combined_block, item,
                                            inputIDs[:len(step.input_step)], outputID)
        blocks.append(combined_block)
        return blocks

    def __align_items(self, step, new_block, blocks):
        lists = self.__gen_lists(step, blocks, new_block)
        i = 0
        combined_block = dict(name=rf'updated_block{len(blocks):02d}')
        combined_block['items'] = []
        for items in itertools.product(*lists):
            if items[0]['item_steps'][0]['ID'] == items[1]['item_steps'][0]['ID']:
                item = dict(name=rf'item{i:03d}')
                item['item_steps'] = []
                for st in items[1]['item_steps']:
                    item['item_steps'].append(copy.deepcopy(st))
                item['item_steps'].append(copy.deepcopy(items[2]['item_steps'][0]))

                outputID = items[2]['item_steps'][-1]['outputID'].split('_')[0] + '_' + \
                           items[1]['item_steps'][-1]['outputID'].split('_')[-1] + \
                           items[2]['item_steps'][-1]['outputID'].split('_')[-1]
                inputIDs = [items[0]['item_steps'][-1]['outputID'],
                            items[1]['item_steps'][-1]['outputID']]
                if step.name == 'Organize':
                    outputID = outputID.replace(inputIDs[0], '')
                combined_block = self.__add_ids(combined_block, item, inputIDs, outputID)

                i += 1
        blocks.append(combined_block)
        return blocks

    def __add_ids(self, block, item, inputIDs, outputID):
        block['items'].append(item)
        block['items'][-1]['item_steps'][-1]['inputIDs'] = inputIDs
        block['items'][-1]['item_steps'][-1]['outputID'] = outputID.rstrip('_').strip('_')
        return block

    def __add_module_ids(self):
        # add unique module ids
        items = []
        for item in self.workflow_graph['items']:
            for module in item['modules']:
                module_id = module['name'] + '_' + module['outputID']
                if 'inputIDs' in module:
                    module_id += '_' + '_'.join(module['inputIDs'])
                module['module_id'] = module_id

            items.append(item)
        self.workflow_graph['items'] = items

    def __remove_repeating_steps(self):
        for i in range(len(self.workflow_graph['items'])):
            module_ids = []
            modules = []
            for module in self.workflow_graph['items'][i]['modules']:
                if not module['module_id'] in module_ids:
                    modules.append(module)
                    module_ids.append(module['module_id'])

            self.workflow_graph['items'][i]['modules'] = modules
