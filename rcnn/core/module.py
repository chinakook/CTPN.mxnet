# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying and dynamic graph creating with training iterations. If shapes vary
and graph change, executors will rebind, using shared arrays from the initial
module binded with maximum shape.
"""

import logging

from mxnet import context as ctx
from mxnet.initializer import Uniform
from mxnet.module.base_module import BaseModule
from mxnet.module.module import Module

from mxnet.module.executor_group import DataParallelExecutorGroup
from mxnet.module.base_module import _parse_data_desc
from mxnet.ndarray.utils import zeros

class OldModule(Module):
    def bind(self, data_shapes, label_shapes=None, for_training=True,
            inputs_need_grad=False, force_rebind=False, shared_module=None,
            grad_req='write'):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
        for_training : bool
            Default is ``True``. Whether the executors should be bound for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. This is used in bucketing. When not ``None``, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already bound, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        self._data_shapes, self._label_shapes = _parse_data_desc(
            self.data_names, self.label_names, data_shapes, label_shapes)

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                    shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
            assert len(shared_group.execs) >= len(self._context)
        else:
            shared_group = None

        self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                        self._work_load_list, self._data_shapes,
                                                        self._label_shapes, self._param_names,
                                                        for_training, inputs_need_grad,
                                                        shared_group, logger=self.logger,
                                                        fixed_param_names=self._fixed_param_names,
                                                        grad_req=grad_req, group2ctxs=self._group2ctxs,
                                                        state_names=self._state_names)
        self._total_exec_bytes = self._exec_group._total_exec_bytes
        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params
        elif self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)
        else:
            assert self._arg_params is None and self._aux_params is None
            param_arrays = [
                zeros(shape=x[0].shape, dtype=x[0].dtype, stype=x[0].stype)
                for x in self._exec_group.param_arrays
            ]
            self._arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)}

            aux_arrays = [
                zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.aux_arrays
            ]
            self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)}

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)


class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """
    def __init__(self, gen_sym, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        self._gen_sym = gen_sym
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        if max_label_shapes is None:
            inputs = dict(max_data_shapes)
        else:
            inputs = dict(max_data_shapes+max_label_shapes)
        self._symbol = gen_sym(inputs, len(self._context) if isinstance(self._context, list) else 1)
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix

        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init, allow_extra=allow_extra)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes))

        max_data_shapes = list()
        for name, shape in data_shapes:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if label_shapes is not None:
            for name, shape in label_shapes:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None
        
        module = OldModule(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)
        module.bind(max_data_shapes, max_label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = dict(self._curr_module.data_shapes + self._curr_module.label_shapes)
        else:
            current_shapes = dict(self._curr_module.data_shapes)

        # get input_shapes
        if data_batch.provide_label is not None:
            input_shapes = dict(data_batch.provide_data + data_batch.provide_label)
        else:
            input_shapes = dict(data_batch.provide_data)

        # decide if shape changed
        shape_changed = False
        for k, v in current_shapes.items():
            if v != input_shapes[k]:
                shape_changed = True

        if shape_changed:
            sym = self._gen_sym(input_shapes, len(self._context) if isinstance(self._context, list) else 1)
            self._curr_module._symbol = sym
            self._curr_module.binded=False
            self._curr_module.bind(data_batch.provide_data, data_batch.provide_label, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)

        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)
