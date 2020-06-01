# -*- coding: utf-8 -*-
# junky lib: autotrain
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a tool for hyperparameters selection.
"""
from collections.abc import Iterable
from copy import deepcopy
import junky
import re
import threading
import torch


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
                junky.enforce_reproducibility(seed=seed)

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
