# -*- coding: utf-8 -*-
# junky lib
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of tools and utilities to use with PyTorch.
"""
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import re
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def get_max_dims(array, str_isarray=False, max_dims=None, dim_no=0):
    """Returns max sizes of nested *array* on the all levels of nestedness.

    :param array: nested lists or tuples.
    :param str_isarray: if True, strings are treated as arrays of chars and
        form additional dimension.
    :param max_dims: for internal use only. Leave it as it is.
    :param dim_no: for internal use only. Leave it as it is.
    """
    if max_dims is None:
        max_dims = []  # it's importaint! don't try to move
                       # max_dims=[] to the args list!

    max_dims_len = len(max_dims)
    res = max_dims
    isnew = dim_no == len(max_dims)
    if not isnew:
        max_dim = max_dims[dim_no]
        if max_dim is None:
            res = None

    if res is not None:
        array_len = None
        isstr = isinstance(array, str)
        try:
            if not str_isarray and isstr:
                raise TypeError()
            array_len = len(array)
        except TypeError:
            res = None

        if isnew:
            max_dims.append(array_len)
        elif array_len is None:
            max_dims[dim_no:] = [array_len]
        else:
            if array_len > max_dim:
                max_dims[dim_no] = array_len
            if isstr:
                max_dims[dim_no + 1:] = [None]

        if res is not None and not isstr:
            dim_no_ = dim_no + 1
            for el in array:
                if not get_max_dims(el, str_isarray=str_isarray,
                                    max_dims=max_dims, dim_no=dim_no_):
                    break

    if dim_no == 0 and res is not None:
        res = res[:-1]

    return res

def insert_to_ndarray(array, ndarray, shift='left'):
    """Inserts a nested *array* with data of any allowed for numpy type to the
    numpy *ndarray*. NB: all dimensions of *ndarray* must be no less than sizes
    of corresponding subarrays of the *array*.

    :param array: nested lists or tuples.
    :param shift: how to place data of *array* to *ndarray* in the case if
        size of some subarray less than corresponding dimension of *ndarray*.
        Allowed values:
            'left': shift to start;
            'right': shift to end;
            'center': place by center (or 1 position left from center if
                      evennesses of subarray's size and ndarray dimension are
                      not congruent);
            'rcenter': the same as 'center', but if evennesses are not
                       congruent, the shift will be 1 position right.
    """
    asize = len(array)
    nsize = len(ndarray)

    if shift == 'left':
        start_idx = 0
    elif shift == 'right':
        start_idx = nsize - asize
    elif shift in ['center', 'lcenter']:
        start_idx = (nsize - asize) // 2
    elif shift == 'rcenter':
        start_idx = int((nsize - asize + 1) // 2)
    else:
        raise ValueError('ERROR: Unknown shift value "{}"'.format(shift))

    if len(ndarray.shape) > 1:
        for idx in range(len(array)):
            insert_to_ndarray(array[idx], ndarray[start_idx + idx],
                              shift=shift)
    else:
        ndarray[start_idx:start_idx + len(array)] = array

def pad_array(array, padding_value=0):
    """Converts nested *array* with data of any allowed for numpy type to
    numpy.ndarray with *padding_value* instead of missing data.

    :param array: nested lists or tuples.
    :rtype: numpy.ndarray
    """

    dims = get_max_dims(array)
    out_array = np.full(shape=dims, fill_value=padding_value)
    insert_to_ndarray(array, out_array)

    return out_array

def pad_array_torch(array, padding_value=0, **kwargs):
    """Just a wropper for ``pad_array()`` that returns *torch.Tensor*.

    :param kwargs: keyword args for the ``torch.tensor()`` method.
    :rtype: torch.Tensor
    """
    return torch.tensor(pad_array(array, padding_value=padding_value),
                        **kwargs)

def torch_autotrain(
    make_model_method, train_method, create_loaders_method=None,
    make_model_args=(), make_model_kwargs=None, make_model_fit_params=None,
    train_args=(), train_kwargs=None, devices=torch.device('cpu'),
    best_model_file_name='model.pt', best_model_device=None, seed=None
):
    """This is a tool for model's hyperparameters selection. May work in
    parallel using multiple devices. If some of parallel threads die during
    training (because of `MemoryError` of anything), their tasks will be
    redone after all other threads have finished with their work.

    :param make_model_method: method to create the model. Returns the model
        and, maybe, some other params that should be passed to *train_method*.
    :type make_model_method: callable(
            *make_model_args, **make_model_kwargs,
            **fit_kwargs
        ) -> model|tuple(model, <other train args>)
        fit_kwargs - params that are constructed from *make_model_fit_params*.
    :param train_method: method to train and validate the model.
    :type train_method: callable(
            device, loaders, model, *other_train_args,
            best_model_backup_method, log_prefix,
            *train_args, **train_kwargs
        ) -> <train statistics>
        device - one of *devices* that is assigned to train the model;
        loaders - the return of *create_loaders_method* or () if
            *create_loaders_method* is None (default);
        other_train_args - params returned by *make_model_method* besides the
            model (if any). E.g.: optimizer, criterion, etc.;
        best_model_backup_method - the method that saves the best model over
            all runs. Signature: callable(best_model, best_model_score).
            This method must be invoked in *train_method* to save its best
            model;
        log_prefix - prefix that should use *train_method* in the beginning of
            any output. Elsewise, you can't distinct messages from parallel
            threads.
    :param create_loaders_method: method to create DataLoaders objects to use
        in *train_method*. Every thread creates it only once and then pass to
        *train_method* of every model that this thread is assigned for. If
        None (default), *train_method* must create loaders by itself.
        Important: you can't use one DataLoader in several threads. You must
        have separate DataLoader for every thread; otherwise, your training is
        gonna be broken.
    :type create_loaders_method: callable() -> <loader>|tuple(<loaders>)
    :param make_model_args: positional args for *make_model_method*. Will be
        passed as is.
    :type make_model_args: tuple
    :param make_model_kwargs: keyword args for *make_model_method*. Will be
        passed as is.
    :type make_model_kwargs: dict
    :param make_model_fit_params: a list of combinations of varying
        *make_model_method*'s fit_kwargs among which we want to find the best.
    :type make_model_fit_params: iterable of iterables; nestedness is
        unlimited. Examples:
            [('a', [50, 100]), ('b': [.1, .5])]
            produces fit_kwargs {'a': 50, 'b': .1},
                                {'a': 50, 'b': .5},
                                {'a': 100, 'b': .1},
                                {'a': 100, 'b': .5};
            [('a', [50, 100]),
             [('b': [.1, .5])], [('b': None), ('c': ['X', 'Y'])]]
            produces {'a': 50, 'b': .1},
                     {'a': 50, 'b': .5},
                     {'a': 100, 'b': .1},
                     {'a': 100, 'b': .5},
                     {'a': 50, 'b': None, 'c': 'X'},
                     {'a': 50, 'b': None, 'c': 'Y'},
                     {'a': 100, 'b': None, 'c': 'X'},
                     {'a': 100, 'b': None, 'c': 'Y'}.
    :param train_args: positional args for *train_method*. Will be passed
        as is.
    :type train_args: tuple
    :param train_kwargs: keyword args for *train_method*. Will be passed
        as is.
    :type make_model_args: dict
    :param devices: what devices use for training. This can be a separate
        device, a list of available devices, or a dict of available devices
        with max number of simultaneous threads.
    :type devices: <device>|tuple(<device>)|dict({<device>: int})
        Examples: torch.device('cpu') - one thread on CPU (default);
                  ('cuda:0', 'cuda:1', 'cuda:2') - 3 GPU, 1 thread on each;
                  {'cuda:0': 3, 'cuda:1': 3} - 2 GPU, 3 threads on each.
        NB: <device> == (<device>,) == {<device>: 1}
    :param best_model_file_name: a name of the file to save the best model
        where. Default 'model.pt'.
    :type best_model_file_name: str
    :param best_model_device: device to load the best model where. If None, we
        won't load the best model in memory.
    :return: tuple(best_model, best_model_name, best_model_score,
                   best_model_params, stats)
        best_model - the best model if best_model_device is not None,
            else None;
        best_model_name - the key of the best model stats;
        best_model_score - the score of the best model;
        best_model_params - fit_kwargs of the best model;
        stats - all returns of all *train_method*s. Format:
            [(<model name>, <model best score>, <model params>,
              <*train_method* return>),
             ...]
            stats is sorted by <model best score>, in such a way that stats[0]
            corresponds to the best model.
    """

    def run_model(lock, device, seed, best_model_file_name,
                  best_model_score, best_model_name, stats,
                  make_model_method, make_model_args,
                  make_model_kwargs, fit_kwargs,
                  train_method, train_args, train_kwargs,
                  create_loaders_method, get_exception_method):

        t = threading.current_thread()
        with lock:
            print('{} started: {}'.format(t.name, device))
        iter_name = t.name

        loaders = create_loaders_method() if create_loaders_method else ()
        if not isinstance(loaders, tuple):
            loaders = [loaders]

        class local_model_score: value = -1.
        def backup_method(model, model_score):
            e = get_exception_method()
            if e:
                raise e
            with lock:
                if model_score > local_model_score.value:
                    local_model_score.value = model_score
                    print('{}: new maximum score {:.8f}'
                              .format(iter_name, model_score),
                          end='')
                    if model_score > best_model_score.value:
                        print(': OVERALL MAXIMUM')
                        best_model_score.value = model_score
                        best_model_name.value = iter_name
                        torch.save(model.state_dict(), best_model_file_name)
                    else:
                        print(' (less than maximum {:.8f} of {})' \
                                  .format(best_model_score.value,
                                          best_model_name.value))

        iter_no = 0
        while True:
            local_model_score.value = -1.

            iter_name = '{}_{}'.format(t.name, iter_no)

            if seed:
                enforce_reproducibility(seed=seed)

            kwargs = None
            with lock:
                try:
                    kwargs = fit_kwargs.pop(0)
                except IndexError:
                    break

            with lock:
                e = get_exception_method()
                if e:
                    raise e
                print('\n{}: {}\n'.format(iter_name, kwargs))
                other_train_args = make_model_method(
                    *deepcopy(make_model_args),
                    **deepcopy(make_model_kwargs),
                    **dict(deepcopy(kwargs))
                )
                model, other_train_args = \
                    (other_train_args[0], other_train_args[1:]) \
                        if isinstance(other_train_args, tuple) else \
                    (other_train_args[1:], [])
                #model = deepcopy(model)
                f_ = model.forward
                def forward(*args, **kwargs):
                    e = get_exception_method()
                    if e:
                        raise e
                    return f_(*args, **kwargs)
                model.forward = forward
                model.to(device)
                print('\n' + iter_name, 'model:', model,
                      next(model.parameters()).device,'\n')
            stat = train_method(
                device, loaders, model, *other_train_args, backup_method,
                '{}: '.format(iter_name), *train_args, **train_kwargs
            )
            with lock:
                stats.append((iter_name, local_model_score.value, kwargs,
                              stat))

            iter_no += 1

        with lock:
            print(
                '{} finished: {} ({})'.format(
                    t.name, device, '{} cycles'
                                        .format(iter_no) if iter_no else
                                    'no data'
            ))

    print('AUTOTRAIN STARTED')
    print('=================')

    if train_kwargs is None:
        train_kwargs = {}
    if make_model_kwargs is None:
        make_model_kwargs = {}
    if make_model_fit_params is None:
        make_model_fit_params = []

    def parse_kwargs(kwargs):
        res = []
        if isinstance(kwargs, dict):
            kwargs = kwargs.items()
        if len(kwargs) == 0:
            res = [[]]
        else:
            for param, vals in kwargs:
                assert isinstance(param, str), \
                       'ERROR: make_model_fit_params has invalid format'
                vals = list(vals if isinstance(vals, Iterable) else [vals])
                if len(vals) > 0:
                    res = [
                        [(param, val)] + kwarg for val in vals
                                               for kwarg in res
                    ] if res else [
                        [(param, val)] for val in vals
                    ]
        return res

    def parse_params(params):
        assert isinstance(params, Iterable), \
               'ERROR: make_model_fit_params has invalid format'
        res = []
        if len(params) == 0:
            res = [[]]
        if isinstance(params, dict):
            res = parse_kwargs(params)
        else:
            params = list(params)
            kwargs = []
            params_ = []
            for pars in params:
                if not isinstance(pars, dict) and len(pars) == 2 \
                                              and isinstance(pars[0], str):
                    kwargs.append(pars)
                else:
                    params_.append(pars)
            kwargs = parse_kwargs(kwargs)
            params = []
            for pars_ in params_:
                for pars in parse_params(pars_):
                    params.append(pars)
            if params:
                if kwargs:
                    for params_ in params:
                        for kwargs_ in kwargs:
                            res.append(kwargs_ + params_)
                else:
                    res = params
            else:
                res = kwargs
        return res

    fit_kwargs = []
    for kwargs in [tuple(sorted(x, key=lambda x: str(x)))
                       for x in parse_params(make_model_fit_params)]:
        if kwargs not in fit_kwargs:
            fit_kwargs.append(kwargs)
    fit_kwargs = sorted(fit_kwargs, key=lambda x: str(x))

    if isinstance(devices, dict):
        devices = {torch.device(x): y for x, y in devices.items()}
        devices_ = []
        is_empty = False
        while True:
            is_empty = True
            for device in sorted(devices,
                                 key=lambda x: x.index if x.index else
                                               0 if x.type == 'cuda' else
                                               float('inf')):
                cnt = devices[device]
                if cnt > 0:
                    devices_.append(device)
                    cnt -= 1
                    devices[device] = cnt
                    if cnt:
                        is_empty = False
            if is_empty:
                break
        devices = devices_
    elif not isinstance(devices, Iterable):
        devices = [devices]

    def print_items(items, qnt=3):
        last_idx = len(items) - 1
        for i, item in enumerate(items):
            print('{', end='')
            last_jdx = len(item) - 1
            if last_jdx < 0:
                print('}', end='')
            else:
                print()
                j_ = 0
                for j, (key, val) in enumerate(item):
                    if j_ == 0:
                        print('    ', end='')
                    print('{}: {}'.format(key, val), end='')
                    if j < last_jdx:
                        print(', ', end='')
                        j_ += 1
                        if j_ > 2:
                            j_ = 0
                            print()
                print()
                print('}', end='')
                if i < last_idx:
                    print(', ', end='')

    print('make_model args: {}'.format(make_model_args))
    print('make_model kwargs: ', end='')
    print_items([make_model_kwargs.items()])
    print()
    print('=========')
    print('make_model fit kwargs (total {} combinations): ['
              .format(len(fit_kwargs)),
          end='')
    print_items(fit_kwargs)
    print(']')
    print('=========')
    print('train args: {}'.format(train_args))
    print('train kwargs: ', end='')
    print_items([train_kwargs.items()])
    print()
    print('=========')
    head = 'devices: ['
    print(head, end='')
    for i, device in enumerate(devices):
        print('{}{}'.format(('\n' + (' ' * len(head))) if i else '',
                            device),
              end='')
    print(']')
    print('=========')
    print()

    class best_model_score: value = -1.
    class best_model_name: value = None
    stats = []
    params = {}

    threads_pool, lock = [], threading.Lock()
    exception = None
    def get_exception():
        return exception
    try:
        has_errors = False
        while fit_kwargs:
            if has_errors:
                print('\n=== {} thread{} terminated abnormally. Repeating ==='
                          .format(len(fit_kwargs), 's were'
                                      if len(fit_kwargs) > 1 else
                                  ' was'))
            has_errors = True
            pool_size = min(len(devices), len(fit_kwargs))
            print('\n=== Creating {} threads ===\n'.format(pool_size))
            fit_kwargs_ = fit_kwargs[:]
            for device in devices[:pool_size]:
                t = threading.Thread(target=run_model,
                                     args=(lock, device, seed, best_model_file_name,
                                           best_model_score, best_model_name, stats,
                                           make_model_method, make_model_args,
                                           make_model_kwargs, fit_kwargs_,
                                           train_method, train_args, train_kwargs,
                                           create_loaders_method, get_exception),
                                     kwargs={})
                threads_pool.append(t)
                t.start()
            for t in threads_pool:
                t.join()
            if not stats:
                raise RuntimeError(
                    '\n=== ERROR: All threads terminated abnormally. '
                    "Something's wrong. Process has been stopped ==="
                )
            for stat in stats:
                if stat[2] in fit_kwargs:
                    fit_kwargs.remove(stat[2])

    except BaseException as e:
        exception = SystemExit()
        while True:
            try:
                for t in threads_pool:
                    t.join()
            except BaseException:
                continue
            break
        raise e

    best_model, best_model_name, best_model_score, \
    best_model_params, args_ = \
        None, best_model_name.value, best_model_score.value, \
        None, None
    for model_name, _, kwargs, _ in stats:
        if model_name == best_model_name:
            best_model_params = kwargs
            if best_model_device:
                best_model, _, _ = make_model_method(
                    *deepcopy(make_model_args),
                    **deepcopy(make_model_kwargs),
                    **dict(deepcopy(kwargs))
                )
                best_model = best_model.to(best_model_device)
                best_model.load_state_dict(
                    torch.load(best_model_file_name,
                               map_location=best_model_device)
                )
                best_model.eval()
                args_ = ', '.join(str(x) for x in make_model_args)
                if args_ and (make_model_kwargs or kwargs):
                    args_ += ', '
                kwargs_ = ', '.join('{}={}'.format(x, y)
                                        for x, y in make_model_kwargs.items())
                if kwargs_ and kwargs:
                    kwargs_ += ', '
                args_ += kwargs_ + ', '.join('{}={}'.format(x, y)
                                                 for x, y in kwargs)
                break
    print('==================')
    print('AUTOTRAIN FINISHED')
    print('==================')
    print('best model name = {}'.format(best_model_name))
    print('best model score = {}'.format(best_model_score))
    head = 'best_model_params=('
    print(head, end='')
    for i, param in enumerate(best_model_params):
        if i:
            print(',\n' + ' ' * len(head), end='')
        print(param, end='')
    print(')')
    if args_:
        print()
        print('best_model = make_model({})'.format(args_))
        print("best_model = best_model.to('{}')".format(best_model_device))
        print("best_model.load_state_dict(torch.load('{}'))"
                  .format(best_model_file_name))

    return best_model, best_model_name, best_model_score, best_model_params, \
           sorted(stats, key=lambda x: (-x[1], x[2]))

def parse_autotrain_log(log_fn, silent=False):
    """The tool to parse output of the ``torch_autotrain()`` method.

    :param log_fn: a file name of the ``torch_autotrain()`` log file.
    :type log_fn: str
    :param silent: if True, suppress output.
    :return: list[tuple(<model name>, <model best score>, <model params>,
                        <is training finished>)]
    """
    scores = {}
    with open(log_fn, 'rt', encoding='utf-8') as f:
        for line in f:
            match = re.match('([^:\s]+): (\((?:\(.+\))?,?\))$', line.strip())
            if match:
                name, args = match.groups()
                if name in scores:
                    scores[name] = (args, scores[name][1], False)
                else:
                    scores[name] = (args, -1., False)
            else:
                match = re.match('([^:]+): new maximum score ([.\d]+)', line)
                if match:
                    name, score = match.groups()
                    score = float(score)
                    if name in scores:
                        if score > scores[name][1]:
                            scores[name] = (scores[name][0], score, False)
                    else:
                        scores[name] = (None, score, False)
                else:
                    match = re.match('([^:]+): Maximum bad epochs exceeded. '
                                     'Process has stopped', line)
                    if match:
                        name, = match.groups()
                        if name in scores:
                            scores[name] = (scores[name][0], scores[name][1],
                                            score)
                        else:
                            scores[name] = (None, None, True)

    stat = []
    for name in sorted(scores, key=lambda x: (-scores[x][1], scores[x][0])):
        name_ = ('' if scores[name][2] else '*') + name
        stat.append((name, scores[name][1], scores[name][0], scores[name][2]))
        if not silent:
            print('{}\t{}\t{}'.format(name_, scores[name][1], scores[name][0]))

    return stat

def enforce_reproducibility(seed=None):
    """Re-init random number generators.
    [We stole this method from Stanford C224U assignments]"""
    if seed:
        # Sets seed manually for both CPU and CUDA
        torch.manual_seed(seed)
        # For atomic operations there is currently 
        # no simple way to enforce determinism, as
        # the order of parallel operations is not known.
        #
        # CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # System based
        np.random.seed(seed)


class Masking(nn.Module):
    """
    Replaces certain elemens of the incoming data to the `mask` given.

    Args:
        input_size: The number of expected features in the input `x`.
        mask: Replace to what.
        indices_to_highlight: What positions in the `feature` dimension of the
            masked positions of the incoming data must not be replaced to the
            `mask`.
        highlighting_mask: Replace data in that positions to what. If
            ``None``, the data will keep as is.
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)` (<==> `(N, *, H)`). Default:
            ``False``.

    Shape:
        - Input: :math:`(*, N, H)` where :math:`*` means any number of
          additional dimensions and :math:`H = \text{input_size}`.
        - Output: :math:`(*, N, H)` where all are the same shape as the input
          and :math:`H = \text{input_size}`.

    .. note:: Masking layer was made for using right before Softmax. In that
        case and with `mask`=``-inf`` (default), the Softmax output will have
        zeroes in all positions corresponding to `indices_to_mask`.

    .. note:: Usually, you'll mask positions of all non-pad tags in padded
        endings of the input data. Thus, after Softmax, you'll always have the
        padding tag predicted for that endings. As the result, you'll have
        loss = 0, that prevents your model for learning on padding.

    Examples::

        >>> m = Masking(4, batch_first=True)
        >>> input = torch.randn(2, 3, 4)
        >>> output = m(input, torch.tensor([1, 3]))
        >>> print(output)
        tensor([[[ 1.1912, -0.6164,  0.5299, -0.6446],
                 [   -inf,    -inf,    -inf,  1.0000],
                 [   -inf,    -inf,    -inf,  1.0000]],

                [[-0.3011, -0.7185,  0.6882, -0.1656],
                 [-0.3316, -0.3521, -0.9717,  0.5551],
                 [ 0.7721,  0.2061,  0.8932, -1.5827]]])
    """
    __constants__ = ['batch_first', 'highlighting_mask',
                     'indices_to_highlight', 'input_size', 'mask']

    def __init__(self, input_size, mask=float('-inf'),
                 indices_to_highlight=-1, highlighting_mask=1,
                 batch_first=False):
        super().__init__()

        if not isinstance(indices_to_highlight, Iterable):
            indices_to_highlight = [indices_to_highlight]

        self.input_size = input_size
        self.mask = mask
        self.indices_to_highlight = indices_to_highlight
        self.highlighting_mask = highlighting_mask
        self.batch_first = batch_first

        output_mask = torch.tensor([mask] * input_size)
        if indices_to_highlight is not None:
            if highlighting_mask is None:
                output_mask0 = torch.tensor([0] * input_size,
                                            dtype=output_mask.dtype)
                for idx in indices_to_highlight:
                    output_mask0[idx] = 1
                    output_mask[idx] = 0
                output_mask = torch.stack((output_mask0, output_mask))
            else:
                for idx in indices_to_highlight:
                    output_mask[idx] = highlighting_mask
        self.register_buffer('output_mask', output_mask)

    def forward(self, x, lens):
        """
        :param lens: array of lengths of **x** by the `seq` dimension.
        """
        output_mask = self.output_mask
        output_mask0, output_mask = \
            output_mask if len(output_mask.shape) == 2 else \
            (None, output_mask)
        device = output_mask.get_device() if output_mask.is_cuda else \
                 torch.device('cpu')
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens, device=device)

        seq_len = x.shape[self.batch_first]
        padding_mask = \
            torch.arange(seq_len, device=device) \
                 .expand(lens.shape[0], seq_len) >= lens.unsqueeze(1)
        if not self.batch_first:
            padding_mask = padding_mask.transpose(0, 1)
        x[padding_mask] = output_mask if output_mask0 is None else \
                          x[padding_mask] * output_mask0 + output_mask

        return x

    def extra_repr(self):
        return ('{}, mask={}, indices_to_highlight={}, highlighting_mask={}, '
                'batch_first={}').format(
                    self.input_size, self.mask, self.indices_to_highlight,
                    self.highlighting_mask, self.batch_first
                )


class CharEmbeddingRNN(nn.Module):
    """
    Produces character embeddings using bidirectional LSTM.

    Args:
        alphabet_size: length of character vocabulary.
        emb_layer: optional pre-trained embeddings, 
            initialized as torch.nn.Embedding.from_pretrained() or elsewise.
        emb_dim: character embedding dimensionality.
        pad_idx: indices of padding element in character vocabulary.
        out_type: 'final_concat'|'final_mean'|'all_mean'.
            `out_type` defines what to get as a result from the LSTM.
            'final_concat' concatenate final hidden states of forward and
                           backward lstm;
            'final_mean' take mean of final hidden states of forward and
                         backward lstm;
            'all_mean' take mean of all timeframes.

    Shape:
        - Input:
            x: [batch[seq[word[ch_idx + pad] + word[pad]]]]; torch tensor of
                shape :math:`(N, S(padded), C(padded))`, where `N` is
                batch_size, `S` is seq_len and `C` is max char_len in a word
                in current batch.
            lens: [seq[word_char_count]]; torch tensor of shape
                :math:`(N, S(padded), C(padded))`, word lengths for each
                sequence in batch. Used in masking & packing/unpacking
                sequences for LSTM.
        - Output: :math:`(N, S, H)` where `N`, `S` are the same shape as the
            input and :math:` H = \text{lstm hidden size}`.
    
    .. note:: In LSTM layer, we ignore padding by applying mask to the tensor
        and eliminating all words of len=0. After LSTM layer, initial
        dimensions are restored using the same mask.
    """
    __constants__ = ['alphabet_size', 'emb_dim', 'out_type', 'pad_idx']

    def __init__(self, alphabet_size, emb_layer=None, emb_dim=300, pad_idx=0,
                 out_type='final_concat'):
        """
        :param out_type: 'final_concat'|'final_mean'|'all_mean'
        """
        super().__init__()

        self.alphabet_size = alphabet_size
        self.emb_dim = None if emb_layer else emb_dim
        self.pad_idx = pad_idx
        self.out_type = out_type

        self._emb_l = emb_layer if emb_layer else \
                      nn.Embedding(alphabet_size, emb_dim,
                                   padding_idx=pad_idx)
        self._rnn_l = nn.LSTM(input_size=self._emb_l.embedding_dim,
                              hidden_size=self._emb_l.embedding_dim // (
                                  2 if out_type in ['final_concat',
                                                    'all_mean'] else
                                  1 if out_type in ['final_mean'] else
                                  0 # error
                              ),
                              num_layers=1, batch_first=True,
                              dropout=0, bidirectional=True)

    def forward(self, x, lens):
        """
        x: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device

        # сохраняем форму батча символов:
        # [#предложение в батче:#слово в предложении:#символ в слове]
        #                                                       <==> [N, S, C]
        x_shape = x.shape
        # все слова во всех батчах сцепляем вместе. из-за паддинга все
        # предложения одной длины, так что расцепить обратно не будет
        # проблемой. теперь у нас один большой батч из всех слов:
        #                                             [N, S, C] --> [N * S, C]
        # важно: слова из-за паддинга тоже все одной длины. при этом многие
        #        пустые, т.е., состоят только из паддинга
        x = x.flatten(end_dim=1)

        # прогоняем через слой символьного эмбеддинга (обучаемый):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        # сохраняем эту форму тоже
        x_e_shape = x.shape

        # создаём массив длин всех слов. для этого в список массивов длин слов
        # в предложениях добавляем нули в качестве длин слов, состоящих только
        # из паддинга, после чего сцепляем их вместе так же, как мы до этого
        # сцепили символы: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # дальше предполагалось передать x и lens0 в pack_padded_sequence,
        # в надежде что она нейтрализует влияние слов нулевой длины. однако
        # выяснилось, что эта функция нулевые длины не принимает. поэтому:
        # 1. делаем маску длин слов: True, если длина не равна нулю (слово не
        # из паддинга)
        mask = lens0 != 0
        # 2. убираем нулевые слова из массива символьных эмбеддингов
        x_m = x[mask]
        # 3. убираем нулевые длины, чтобы массив длин соответствовал новому
        # массиву символьных эмбеддингов
        lens0_m = lens0[mask]

        # теперь у нас остались только ненулевые слова и есть массив их длин.
        # запаковываем
        x_m = pack_padded_sequence(x_m, lens0_m,
                                   batch_first=True, enforce_sorted=False)
        # lstm
        x_m, (x_hid, x_cstate) = self._rnn_l(x_m)

        ### в качестве результата можно брать либо усреднение/суммирование/
        ### что-то ещё hidden state на всех таймфреймах (тогда надо будет
        ### вначале распаковать x_m, который содержит конкатенацию hidden
        ### state прямого и обратного lstm на каждом таймфрейме); либо можно
        ### взять финальные значения hidden state для прямого и обратного
        ### lstm и, например, использовать их конкатенацию.
        ### важно: если мы используем конкатенацию (даже неявно, когда
        ### не разделяем x_cm_m), то размер hidden-слоя д.б. в 2 раза меньше,
        ### чем реальная размерность, которую мы хотим получить на входе
        ### в результате.
        if self.out_type == 'all_mean':
            ## если результат - среднее значение hidden state на всех
            ## таймфреймах:
            # 1. распаковываем hidden state. 
            x_m, _ = pad_packed_sequence(x_m, batch_first=True)
            # 2. теперь x_m имеет ту же форму, что и перед запаковыванием.
            # помещаем его в то же место x, откуда забрали (используем
            # ту же маску)
            x[mask] = x_m
            # 3. теперь нужно результат усреднить. нам не нужны значения для
            # каждого символа, мы хотим получить эмбеддинги слов. мы решили,
            # что эмбеддинг слова - это среднее значение эмбеддингов всех
            # входящих в него символов. т.е., нужно сложить эмбеддинги
            # символов и разделить на длину слова. в слове есть паддинг, но
            # после pad_packed_sequence его вектора должны быть нулевыми.
            # однако у нас есть слова полностью из паддинга, которые мы
            # удаляли, а теперь вернули. их вектора после слоя эмбеддинга
            # будут нулевыми, поскольку при создании слоя мы указали индекс
            # паддинга. иначе можно было бы их занулить явно: x[~mask] = .0
            # 4. чтобы посчитать среднее значение эмбеддинга слова, нужно
            # сложить эмбеддинги его символов и разделить на длину слова:
            # 4a. установим длину нулевых слов в 1, чтобы не делить на 0.
            # мы обнулили эти эмбеддинги, так что проблемы не будет
            lens1 = pad_sequence(lens, padding_value=1).flatten()
            # 4b. теперь делим все символьные эмбеддинги на длину слов,
            # которые из них составлены (нормализация):
            x /= lens1[:, None, None]
            # 4c. теперь возвращаем обратно уровень предложений (в
            # результате будет форма [N, S, C, E]) и складываем уже
            # нормализованные эмбеддинги символов в рамках каждого
            # слова (получим форму [N, S, E]). т.е., теперь у нас в x_
            # будут эмбеддинги слов
            x = x_ch.view(*x_shape, -1).sum(-2)

        elif self.out_type in ['final_concat', 'final_mean']:
            ## если результат - конкатенация либо средние значения последних
            ## hidden state прямого и обратного lstm:
            if self.out_type == 'final_concat':
                # 1. конкатенация
                x_m = x_hid.transpose(-1, -2) \
                           .reshape(1, x_hid.shape[-1] * 2, -1) \
                           .transpose(-1, -2)
            elif self.out_type == 'final_mean':
                # 1. среднее
                x_m = x_hid.transpose(-1, -2) \
                           .mean(dim=0) \
                           .transpose(-1, -2)
            # 2. в этой точке нам нужна x_e_shape: forma x после
            # прохождения слоя эмбеддинга. причём измерение символов нам
            # надо убрать, т.е., перейти от [N * S, C, E] к [N * S, E].
            # при этом на месте пустых слов нам нужны нули. создадим
            # новый тензор:
            x = torch.zeros(x_e_shape[0], x_e_shape[2],
                            dtype=torch.float, device=device)
            # 3. сейчас x_m имеет ту же форму, что и перед запаковыванием,
            # но измерения символов уже нет. помещаем его в то же место x,
            # откуда брали (используем ту же маску)
            x[mask] = x_m
            # сейчас у нас x в форме [N * S, E]. переводим его в форму
            # эмбеддингов слов [N, S, E]:
            x = x.view(*x_shape[:-1], -1)

        return x

    def extra_repr(self):
        return '{}, {}, pad_idx={}, out_type={}'.format(
            self.alphabet_size, self.emb_dim, self.pad_idx, self.out_type
        ) if self.emb_dim else \
        '{}, external embedding layer, out_type={}'.format(
            self.alphabet_size, self.out_type
        )


class CharEmbeddingCNN(nn.Module):
    """
    Produces character embeddings using multiple-filter CNN. Max-over-time
    pooling and ReLU are applied to concatenated convolution layers.

    Args:
        alphabet_size: length of character vocabulary.
        emb_layer: optional pre-trained embeddings, 
            initialized as torch.nn.Embedding.from_pretrained() or elsewise.
        emb_dim: character embedding dimensionality.
        emb_dropout: dropout for embedding layer. Default: 0.0 (no dropout).
        pad_idx: indices of padding element in character vocabulary.
        kernels: convoluiton filter sizes for CNN layers. 
        cnn_kernel_multiplier: defines how many filters are created for each 
            kernel size. Default: 1.
        
    Shape:
        - Input:
            x: [batch[seq[word[ch_idx + pad] + word[pad]]]]; torch tensor of
                shape :math:`(N, S(padded), C(padded))`, where `N` is
                batch_size, `S` is seq_len with padding and `C` is char_len
                with padding in current batch. 
            lens: [seq[word_char_count]]; torch tensor of shape
                :math:`(N, S, C)`, word lengths for each sequence in batch.
                Used for eliminating padding in CNN layers.
        - Output: :math:`(N, S, E)` where `N`, `S` are the same shape as the
            input and :math:` E = \text{emb_dim}`.
    """
    __constants__ = ['alphabet_size', 'emb_dim', 'kernels', 'cnn_kernel_multiplier', 'pad_idx']

    def __init__(self, alphabet_size, emb_layer=None, emb_dim=300, emb_dropout=0.0,
                 pad_idx=0, kernels=[3, 4, 5], cnn_kernel_multiplier=1):
        super().__init__()

        self.kernels = list(kernels)
        self.alphabet_size = alphabet_size
        self.emb_dim = None if emb_layer else emb_dim
        self.pad_idx = pad_idx

        self._emb_l = emb_layer if emb_layer else \
                      nn.Embedding(alphabet_size, emb_dim,
                                   padding_idx=pad_idx)
                                   
        self._emb_dropout = nn.Dropout(p=emb_dropout)

        self._conv_ls = nn.ModuleList(
            [nn.Conv1d(in_channels=self._emb_l.embedding_dim,
                       out_channels=self._emb_l.embedding_dim,
                       padding=0, kernel_size=kernel)
                 for kernel in kernels] * cnn_kernel_multiplier
        )

    def forward(self, x, lens):
        """
        x: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device
        max_len = x.shape[-1]

        # сохраняем форму батча символов:
        # [#предложение в батче:#слово в предложении:#символ в слове]
        #                                                       <==> [N, S, C]
        x_shape = x.shape
        # все слова во всех батчах сцепляем вместе. из-за паддинга все
        # предложения одной длины, так что расцепить обратно не будет
        # проблемой. теперь у нас один большой батч из всех слов:
        #                                             [N, S, C] --> [N * S, C]
        # важно: слова из-за паддинга тоже все одной длины. при этом многие
        #        пустые, т.е., состоят только из паддинга
        x = x.flatten(end_dim=1)

        # прогоняем через слой символьного эмбеддинга (обучаемый):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        x = self._emb_dropout(x)
        # сохраняем эту форму тоже
        x_e_shape = x.shape

        # создаём массив длин всех слов. для этого в список массивов длин слов
        # в предложениях добавляем нули в качестве длин слов, состоящих только
        # из паддинга, после чего сцепляем их вместе так же, как мы до этого
        # сцепили символы: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # теперь маскируем слова нулевой длины:
        # 1. делаем маску длин слов: True, если длина не равна нулю (слово не
        # из паддинга)
        mask = lens0 != 0
        # 2. убираем нулевые слова из массива символьных эмбеддингов
        x_m = x[mask]
        # 3. убираем нулевые длины, чтобы массив длин соответствовал новому
        # массиву символьных эмбеддингов
        lens0_m = lens0[mask]

        # Добавим здесь три сверточных слоя с пулингом
        # NB! CNN принимает на вход тензор размерностью
        #     [batch_size, hidden_size, seq_len] 
        # CNN tensor input shape:
        #     [nonzero(N * S), E, C]
        # tensor after convolution shape:
        #     [nonzero(N * S), E, C - cnn_kernel_size + 1]
        # tensor after pooling:
        #     [nonzero(N * S), E, (C - cnn_kernel_size + 1)
        #                       - (pool_kernel_size - 1)]
        # example:
        ## N=32, E=300, C=45
        ## CNN (kernel_size=5) [32, 300, 41]
        ## pooling (kernel_size=8) [32, 300, 34]
        x_m = x_m.transpose(1, 2)

        x_ms = []
        for conv_l in self._conv_ls:
            if conv_l.kernel_size[0] <= max_len:
                #x_ms.append(F.relu(F.adaptive_max_pool1d(conv_l(x_m),
                #                                         output_size=2)))
                x_ms.append(conv_l(x_m))

        x_m = torch.cat(x_ms, dim=-1)
        # сейчас x_m имеет форму [N * S, E, pool_concat]. нам нужно привести
        # её к виду [N * S, E]. нам не нужно транспонировать его обратно, мы
        # просто берём среднее с учётом текущего расположения измерений
        #x_m = torch.mean(x_m, -1)
        x_m = F.relu(torch.max(x_m, -1)[0])

        # в этой точке нам нужна x_e_shape: forma x после
        # прохождения слоя эмбеддинга. причём измерение символов нам
        # надо убрать, т.е., перейти от [N * S, C, E] к [N * S, E].
        # при этом на месте пустых слов нам нужны нули. создадим
        # новый тензор:
        x = torch.zeros(x_e_shape[0], x_e_shape[2],
                        dtype=torch.float, device=device)
        # 3. сейчас x_m имеет ту же форму, что и перед запаковыванием,
        # но измерения символов уже нет. помещаем его в то же место x,
        # откуда брали (используем ту же маску)
        x[mask] = x_m
        # сейчас у нас x в форме [N * S, E]. переводим его в форму
        # эмбеддингов слов [N, S, E]:
        x = x.view(*x_shape[:-1], -1)

        return x

    def extra_repr(self):
        return '{}, {}, pad_idx={}, kernels={}'.format(
            self.alphabet_size, self.emb_dim, self.pad_idx, self.kernels
        ) if self.emb_dim else \
        '{}, external embedding layer, kernels={}'.format(
            self.alphabet_size, self.kernels
        )


class Highway(nn.Module):
    """ 
    Highway layer for Highway Networks as described in
    https://arxiv.org/abs/1505.00387 and https://arxiv.org/abs/1507.06228
    articles.

    Applies H(x)*T(x) + x*(1 - T(x)) transformation, where:
    .. H(x) - affine trainsform followed by a non-linear activation. The layer
           that we make Highway around;
    .. T(x) - transform gate: affine transform followed by a sigmoid
           activation;
    .. * - element-wise multiplication.

    Args:
        dim: size of each input and output sample.
        H_layer: H(x) layer. If ``None`` (default), affine transform is used.
        H_activation: non-linear activation after H(x). If ``None`` (default),
            then, if H_layer is ``None``, too, we apply F.relu; otherwise,
            activation function is not used.
    """
    __constants__ = ['H_layer', 'H_activation', 'dim']

    def __init__(self, dim, H_layer=None, H_activation=None):
        super().__init__()

        self._H = H_layer if H_layer else nn.Linear(dim, dim)
        self._H_activation = H_activation if H_activation else \
                             F.relu

        self._T = nn.Linear(dim, dim)
        self._T_activation = torch.sigmoid
        nn.init.constant_(self._T.bias, -1)

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, seq_len, emb_size]
        :return: tensor with shape [batch_size, seq_len, emb_size]
        """
        gate = self._T_activation(self._T(x))
        hx = self._H(x)
        if self._H_activation:
            hx = self._H_activation(hx)

        return hx * gate + x * (1 - gate)

    def extra_repr(self):
        return '{}, H_layer={}, H_activation={}'.format(
            self._T.dim, self._H, self._H_activation
        )


class HighwayNetwork(nn.Module):
    """ 
    Highway Network is described in
    https://arxiv.org/abs/1505.00387 and https://arxiv.org/abs/1507.06228 and
    it's formalation is: H(x)*T(x) + x*(1 - T(x)), where:
    .. H(x) - affine trainsformation followed by a non-linear activation;
    .. T(x) - transformation gate: affine transformation followed by a sigmoid
           activation;
    .. * - element-wise multiplication.

    There are some variations of it, so we implement more universal 
    architectute: U(x)*H(x)*T(x) + x*C(x), where:
    .. U(x) - user defined layer that we make Highway around; By default,
           U(x) = I (identity matrix);
    .. C(x) - carry gate: generally, affine transformation followed by a sigmoid
           activation. By default, C(x) = 1 - T(x).

    Args:
        in_features: number of features in input.
        out_features: number of features in output. If ``None`` (default),
            **out_features** = **in_features**.
        U_layer: layer that implements U(x). Default is ``None``. If U_layer
            is callable, it will be used to create the layer; elsewise, we'll
            use it as is (if **num_layers** > 1, we'll copy it). Note that
            number of input features of U_layer must be equal to
            **out_features** if **num_layers** > 1.
        U_init_: callable to inplace init weights of **U_layer**.
        U_dropout: if non-zero, introduces a Dropout layer on the outputs of
            U(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        H_features: number of input features of H(x). If ``None`` (default),
            H_features = in_features. If ``0``, don't use H(x).
        H_activation: non-linear activation after H(x). If ``None``, then no
            activation function is used. Default is ``F.relu``.
        H_dropout: if non-zero, introduces a Dropout layer on the outputs of
            H(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        gate_type: a type of the transform and carry gates:
            'generic' (default): C(x) = 1 - T(x);
            'independent': use both independent C(x) and T(x);
            'T_only': don't use carry gate: C(x) = I;
            'C_only': don't use carry gate: T(x) = I;
            'none': C(x) = T(x) = I.
        global_highway_input: if ``True``, we treat the input of all the
            network as the highway input of every layer. Thus, we use T(x)
            and C(x) only once. If **global_highway_input** is ``False``
            (default), every layer receives the output of the previous layer
            as the highway input. So, T(x) and C(x) use different weights
            matrices in each layer.
        num_layers: number of highway layers.
    """
    __constants__ = ['H_activation', 'H_dropout', 'H_features', 'U_dropout',
                     'U_init', 'U_layer', 'gate_type', 'global_highway_input',
                     'last_dropout', 'out_dim', 'num_layers']

    def __init__(self, in_features, out_features=None,
                 U_layer=None, U_init_=None, U_dropout=0,
                 H_features=None, H_activation=F.relu, H_dropout=0,
                 gate_type='generic', global_highway_input=False,
                 num_layers=1):
        super().__init__()

        if out_features is None:
            out_features = in_features
        if H_features is None:
            H_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.U_layer = U_layer
        self.U_init_ = U_init_
        self.U_dropout = U_dropout
        self.H_features = H_features
        self.H_activation = H_activation
        self.H_dropout = H_dropout
        self.gate_type = gate_type
        self.global_highway_input = global_highway_input
        self.num_layers = num_layers

        if U_layer:
            self._U = U_layer() if callable(U_layer) else U_layer
            if U_init_:
                U_init_(U_layer)
        else:
            self._U = None
        self._H = nn.Linear(H_features, out_features) if H_features else None
        if self.gate_type not in ['C_only', 'none']:
            self._T = nn.Linear(in_features, out_features)
            nn.init.constant_(self._T.bias, -1)
        if self.gate_type not in ['generic', 'T_only', 'none']:
            self._C = nn.Linear(in_features, out_features)
            nn.init.constant_(self._C.bias, 1)

        self._U_do = \
            nn.Dropout(p=U_dropout) if self._U and U_dropout else None
        self._H_do = \
            nn.Dropout(p=H_dropout) if self._H and H_dropout else None

        self._H_activation = H_activation
        self._T_activation = torch.sigmoid
        self._C_activation = torch.sigmoid

        if self.num_layers > 1:
            self._Us = nn.ModuleList() if U_layer else None
            self._Hs = nn.ModuleList() if H_features else None
            if not self.global_highway_input:
                if self.gate_type not in ['C_only', 'none']:
                    self._Ts = nn.ModuleList()
                if self.gate_type not in ['generic', 'T_only', 'none']:
                    self._Cs = nn.ModuleList()

            for i in range(self.num_layers - 1):
                if self._Us is not None:
                    U = U_layer() if callable(U_layer) else deepcopy(U_layer)
                    if U_init_:
                       U_init_(U)
                    self._Us.append(U)
                if self._Hs is not None:
                    self._Hs.append(nn.Linear(out_features, out_features))
                if not self.global_highway_input:
                    if self.gate_type not in ['C_only', 'none']:
                        T = nn.Linear(in_features, out_features)
                        nn.init.constant_(T.bias, -1)
                        self._Ts.append(T)
                    if self.gate_type not in ['generic', 'T_only', 'none']:
                        C = nn.Linear(in_features, out_features)
                        nn.init.constant_(C.bias, -1)
                        self._Cs.append(C)

    def forward(self, x, x_hw, *U_args, **U_kwargs):
        """
        :param x: tensor with shape [batch_size, seq_len, emb_size]
        :param x_hw: tensor with shape [batch_size, in_features, emb_size]
            if ``None``, x is used
        :return: tensor with shape [batch_size, seq_len, emb_size]
        """
        if x_hw is None:
            x_hw = x

        if self._U:
            x = self._U(x, *U_args, **U_kwargs)
            if self._U_do:
                x = self._U_do(x)
        if self._H:
            x = self._H(x)
        if self._H_activation:
            x = self._H_activation(x)
            if self._H_do:
                x = self._H_do(x)

        if self.gate_type == 'generic':
            x_t = self._T_activation(self._T(x_hw))
            x = x_t * x
            x_hw = (1 - x_t) * x_hw
        elif self.gate_type not in ['C_only', 'none']:
            x = self._T_activation(self._T(x_hw)) * x
        elif self.gate_type not in ['T_only', 'none']:
            x_hw = self._C_activation(self._C(x_hw)) * x_hw

        x += x_hw

        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                if self._Us:
                    x = self._Us[i](x, *U_args, **U_kwargs)
                    if self._U_do:
                        x = self._U_do(x)
                if self._Hs:
                    x = self._Hs[i](x)
                if self._H_activation:
                    x = self._H_activation(x)
                    if self._H_do:
                        x = self._H_do(x)

                if not self.global_highway_input:
                    if self.gate_type == 'generic':
                        x_t = self._T_activation(self._Ts[i](x_hw))
                        x = x_t * x
                        x_hw = (1 - x_t) * x_hw
                    elif self.gate_type not in ['C_only', 'none']:
                        x = self._T_activation(self._Ts[i](x_hw)) * x
                    elif self.gate_type not in ['T_only', 'none']:
                        x_hw = self._C_activation(self._Cs[i](x_hw)) * x_hw

                x += x_hw
                if not self.global_highway_input:
                    x_hw = x

        return x

    def extra_repr(self):
        return (
            '{}, {}, U_layer={}, U_init_={}, U_dropout={}, '
            'H_features={}, H_activation={}, H_dropout={}, '
            "gate_type='{}', global_highway_input={}, num_layers={}"
        ).format(self.in_features, self.out_features,
                 None if not self.U_layer else
                 '<callable>' if callable(self.U_layer) else
                 '<layer>' if isinstance(self.U_layer, nn.Model) else
                 '<ERROR>',
                 None if not self.U_init_ else
                 '<callable>' if callable(self.U_init_) else
                 '<ERROR>', self.U_dropout,
                 self.H_features,
                 None if not self.H_activation else
                 '<callable>' if callable(self.H_activation) else
                 '<ERROR>', self.H_dropout,
                 self.gate_type, self.global_highway_input, self.num_layers)
