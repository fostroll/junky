# -*- coding: utf-8 -*-
# junky lib
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of tools and utilites to use with PyTorch.
"""
import numpy as np
import threading
import torch


def get_max_dims(array, max_dims=None, str_isarray=False, dim_no=0):
    """Returns max sizes of nested **array** on the all levels
    of nestedness.

    :param array: nested lists or tuples.
    :param str_isarray: if True, strings are treated as arrays of chars and
        form additional dimension.
    :param dim_no: for internal used only. Stay it as it is.
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
                if not get_max_dims(el, max_dims=max_dims, dim_no=dim_no_,
                                    str_isarray=str_isarray):
                    break

    if dim_no == 0 and res is not None:
        res = res[:-1]

    return res

def insert_to_ndarray(array, ndarray, shift='left'):
    """Inserts a nested **array** with data of any allowed for *numpy* type to
    the *numpy* **ndarray**. NB: all dimensions of **ndarray** must be no less
    than sizes of corresponding subarrays of the
    **array**.

    :param array: nested lists or tuples.
    :param shift: how to place data of **array** to **ndarray** if size of
        some subarray less than corresponding dimension of **ndarray**.
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
    """Converts nested **array** with data of any allowed for *numpy* type to
    *numpy.ndarray* with **padding_value** instead of missing data.

    :param array: nested lists or tuples.
    :rtype: numpy.ndarray
    """

    dims = get_max_dims(array)
    out_array = np.full(shape=dims, fill_value=padding_value)
    insert_to_ndarray(array, out_array)

    return out_array

def pad_array_torch(array, padding_value=0, **kwargs):
    """Just a wropper for ``pad_array()`` that returns *torch.Tensor*.

    :param kwargs: keyword args for the ``tensor.tensor()`` method.
    :rtype: torch.Tensor
    """
    return torch.tensor(pad_array(sequences, padding_value=padding_value),
                        **kwargs)

def torch_autotrain(
    make_model_method, train_method, create_loaders_method=None,
    make_model_args=(), make_model_kwargs=None, make_model_fit_params=None,
    train_args=(), train_kwargs=None, devices=torch.device('cpu'),
    best_model_file_name='model.pt', best_model_device=None, seed=None):
    """Model hyperparameters selection. May work in parallel using multiple
    devices.

    :param make_model_method: method to create the model. Returns the model
        and, maybe, some other params that should be passed to *train_method*.
    :type make_model_method: callable(
            *make_model_args, **make_model_kwargs,
            **fit_kwargs
        ) -> model|tuple(model, <other_train_args>)
        fit_kwargs - params that are constructed from *make_model_fit_params*.
    :param train_method: method to train and validate the model.
    :type train_method: callable(
            device, loaders, model, *other_train_args,
            best_model_backup_method, log_prefix,
            *train_args, **train_kwargs
        )
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
        Important: you can't use one DataLoader in several threads; you must
        have separate DataLoader for every thread, otherwise, your training is
        gonna be broken.
    :type create_loaders_method: callable() -> <loader>|tuple(<loaders>)
    :param make_model_args: positional args for *make_model_method*. Will be
        passed as is.
    :type make_model_args: tuple.
    :param make_model_kwargs: keyword args for *make_model_method*. Will be
        passed as is.
    :type make_model_args: dict.
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
    :type train_args: tuple.
    :param train_kwargs: keyword args for *train_method*. Will be passed
        as is.
    :type make_model_args: dict.
    :param devices: what devices use for training. This can be a separate
        device, a list of available devices, or a dict of available devices
        with max number of simultaneous threads.
    :type devices: <device>|tuple(<device>)|dict({<device>: int})
        Examples: torch.device('cpu') - one thread on CPU (default);
                  ('cuda:0', 'cuda:1', 'cuda:2') - 3 GPU, 1 thread on each;
                  {'cuda:0': 3, 'cuda:1': 3} - 2 GPU, 3 threads on each.
        NB: <device> == (<device>,) == {<device>: 1}
    :param best_model_file_name: a name of the file to save the best model
        where. Default 'model.pt'
    :type best_model_file_name: str.
    :param best_model_device: device to load best model where. If None, we
        won't load the best model in memory.
    :return: (best_model, best_model_score, best_model_name, stats)
        best_model - the best model if best_model_device is not None,
            else None;
        best_model_score - the score of the best model;
        best_model_name - the key of the best model stats;
        stats - all returns of all *train_method*s. Format:
            [(<model name>, <model fit_kwargs>, <*train_method* return>), ...]
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

        def backup_method(model, model_score):
            e = get_exception_method()
            if e:
                raise e
            with lock:
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
            iter_name = '{}_{}'.format(t.name, iter_no)

            #if seed:
            #    enforce_reproducibility(seed=seed)

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
                stats.append((iter_name, kwargs, stat))

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
        for param, vals in kwargs:
            assert isinstance(param, str), \
                   'ERROR: make_model_fit_params has invalid format'
            vals = list(vals if isinstance(vals, Iterable) else [vals])
            if len(vals) > 0:
                res = [
                    kwarg + [(param, val)] for val in vals
                                           for kwarg in res
                ] if res else [
                    [(param, val)] for val in vals
                ]
        return res

    def parse_params(params):
        assert isinstance(params, Iterable), \
               'ERROR: make_model_fit_params has invalid format'
        res = []
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

    fit_kwargs = parse_params(make_model_fit_params)

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
    head = 'make_model fit kwargs: ['
    print(head, end='')
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

    class best_model_score: value = -1
    class best_model_name: value = None
    stats = []
    params = {}

    threads_pool, lock = [], threading.Lock()
    exception = None
    def get_exception():
        return exception
    try:
        for device in devices:
            t = threading.Thread(target=run_model,
                                 args=(lock, device, seed, best_model_file_name,
                                       best_model_score, best_model_name, stats,
                                       make_model_method, make_model_args,
                                       make_model_kwargs, fit_kwargs,
                                       train_method, train_args, train_kwargs,
                                       create_loaders_method, get_exception),
                                 kwargs={})
            threads_pool.append(t)
            t.start()
        for t in threads_pool:
            t.join()

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

    best_model, best_model_score, best_model_name, args_ = \
        None, best_model_score.value, best_model_name.value, None
    for model_name, kwargs, _ in stats:
        if model_name == best_model_name:
            if best_model_device:
                best_model, _, _ = make_model(
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
                break
            args_ = ', '.join(make_model_args)
            if args_:
                args_ += ', '
            kwargs_ = ', '.join('{}={}'.format(x, y)
                                    for x, y in make_model_kwargs.items())
            if kwargs_:
                kwargs_ += ', '
            args_ += kwargs_ + ', '.join('{}={}'.format(x, y)
                                             for x, y in kwargs)
    print('==================')
    print('AUTOTRAIN FINISHED')
    print('==================')
    print('best model score = {}, best model name = {}'
              .format(best_model_score, best_model_name))
    if args_:
        print('best_model = make_model{}'.format(args_))
        print('best_model = best_model.to({})'.format(best_model_device))
        print('best_model.load_state_dict(torch.load({}))'
                  .format(best_model_file_name))

    return best_model, best_model_score, best_model_name, stats


class Masking(nn.Module):
    """
    Replaces certain elemens of the incoming data to the `mask` given.

    Args:
        input_size: The number of expected features in the input `x`.
        indices_to_mask: What positions in the `feature` dimension of the
            incoming data must be replaced to the `mask`.
        mask: Replace to what.
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Default: ``False``.

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
    __constants__ = ['batch_first', 'indices_to_mask', 'input_size', 'mask']

    def __init__(self, input_size, indices_to_mask=-1, mask=float('-inf'),
                 batch_first=False):
        super().__init__()

        if not isinstance(indices_to_mask, Iterable):
            indices_to_mask = [indices_to_mask]

        self.input_size = input_size
        self.indices_to_mask = indices_to_mask
        self.mask = mask
        self.batch_first = batch_first

        if indices_to_mask is not None:
            output_mask = torch.tensor([mask] * input_size)
            for idx in indices_to_mask:
                output_mask[idx] = 1
            self.register_buffer('output_mask', output_mask)

    def forward(self, x, lens):
        output_mask = self._buffers.get('output_mask')
        if output_mask.is_cuda:
            device = output_mask.get_device()
        else:
            device = CPU

        if output_mask is not None:
            seq_len = x.shape[self.batch_first]
            padding_mask = \
                torch.arange(seq_len) \
                     .to(device) \
                     .expand(lens.shape[0], seq_len) >= lens.unsqueeze(1)
            if not self.batch_first:
                padding_mask = padding_mask.transpose(0, 1)
            x[padding_mask] = output_mask

        return x

    def extra_repr(self):
        return '{}, indices_to_mask={}, mask={}, batch_first={}'.format(
            self.input_size, self.indices_to_mask, self.mask, self.batch_first
        )


class CharEmbeddingRNN(nn.Module):

    def __init__(self, alphabet_size,
                 emb_layer=None, emb_dim=300, pad_idx=0,
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
                                  2 if out_type in ['final_concat', 'all_mean'] else
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

        # ��������� ����� ����� ��������:
        # [#����������� � �����:#����� � �����������:#������ � �����] <==> [N, S, C]
        x_shape = x.shape
        # ��� ����� �� ���� ������ �������� ������. ��-�� �������� ��� �����������
        # ����� �����, ��� ��� ��������� ������� �� ����� ���������. ������ � ���
        # ���� ������� ���� �� ���� ����: [N, S, C] --> [N * S, C]
        # �����: ����� ��-�� �������� ���� ��� ����� �����. ��� ���� ������ ������,
        #        �.�., ������� ������ �� ��������
        x = x.flatten(end_dim=1)

        # ��������� ����� ���� ����������� ���������� (���������):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        # ��������� ��� ����� ����
        x_e_shape = x.shape

        # ������ ������ ���� ���� ����. ��� ����� � ������ �������� ���� ����
        # � ������������ ��������� ���� � �������� ���� ����, ��������� ������
        # �� ��������, ����� ���� �������� �� ������ ��� ��, ��� �� �� �����
        # ������� �������: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # ������ �������������� �������� x � lens0 � pack_padded_sequence
        # � ������� ��� ��� ������������ ������� ���� ������� �����. ������
        # ����������, ��� ��� ������� ������� ����� �� ���������. �������:
        # 1. ������ ����� ���� ����: True, ���� ����� �� ����� ���� (����� ��
        # �� ��������)
        mask = lens0 != 0
        # 2. ������� ������� ����� �� ������� ���������� �����������
        x_m = x[mask]
        # 3. ������� ������� �����, ����� ������ ���� �������������� ������
        # ������� ���������� �����������
        lens0_m = lens0[mask]

        # ������ � ��� �������� ������ ��������� ����� � ���� ������ �� ����.
        # ������������
        x_m = pack_padded_sequence(x_m, lens0_m,
                                   batch_first=True, enforce_sorted=False)
        # lstm
        x_m, (x_hid, x_cstate) = self._rnn_l(x_m)

        ### � �������� ���������� ����� ����� ���� ����������/������������/
        ### ���-�� ��� hidden state �� ���� ����������� (����� ���� �����
        ### ������� ����������� x_m, ������� �������� ������������ hidden
        ### state ������� � ��������� lstm �� ������ ����������); ���� �����
        ### ����� ��������� �������� hidden state ��� ������� � ���������
        ### lstm �, ��������, ������������ �� ������������.
        ### �����: ���� �� ���������� ������������ (���� ������, �����
        ### �� ��������� x_cm_m), �� ������ hidden-���� �.�. � 2 ���� ������,
        ### ��� �������� �����������, ������� �� ����� �������� �� �����
        ### � ����������.
        if self.out_type == 'all_mean':
            ## ���� ��������� - ������� �������� hidden state �� ���� �����������:
            # 1. ������������� hidden state. 
            x_m, _ = pad_packed_sequence(x_m, batch_first=True)
            # 2. ������ x_m ����� �� �� �����, ��� � ����� ��������������.
            # �������� ��� � �� �� ����� x, ������ ������� (����������
            # �� �� �����)
            x[mask] = x_m
            # 3. ������ ����� ��������� ���������. ��� �� ����� �������� ���
            # ������� �������, �� ����� �������� ���������� ����. �� ������,
            # ��� ��������� ����� - ��� ������� �������� ����������� ����
            # �������� � ���� ��������. �.�., ����� ������� ���������� ��������
            # � ��������� �� ����� �����. � ����� ���� �������, �� �����
            # pad_packed_sequence ��� ������� ������ ���� ��������. ������
            # � ��� ���� ����� ��������� �� ��������, ������� �� �������,
            # � ������ �������. �� ������� ����� ���� ���������� �����
            # ��������, ��������� ��� �������� ���� �� ������� ������ ��������.
            # ����� ����� ���� �� �� �������� ����: x[~mask] = .0
            # 4. ����� ��������� ������� �������� ���������� �����, �����
            # ������� ���������� ��� �������� � ��������� �� ����� �����:
            # 4a. ��������� ����� ������� ���� � 1, ����� �� ������ �� 0.
            # �� �������� ��� ����������, ��� ��� �������� �� �����
            lens1 = pad_sequence(lens, padding_value=1).flatten()
            # 4b. ������ ����� ��� ���������� ���������� �� ����� ����,
            # ������� �� ��� ���������� (������������):
            x /= lens1[:, None, None]
            # 4c. ������ ���������� ������� ������� ����������� (�
            # ���������� ����� ����� [N, S, C, E]) � ���������� ���
            # ��������������� ���������� �������� � ������ �������
            # ����� (������� ����� [N, S, E]). �.�., ������ � ��� � x_
            # ����� ���������� ����
            x = x_ch.view(*x_shape, -1).sum(-2)

        elif self.out_type in ['final_concat', 'final_mean']:
            ## ���� ��������� - ������������ ���� ������� �������� ���������
            ## hidden state ������� � ��������� lstm:
            if self.out_type == 'final_concat':
                # 1. ������������
                x_m = x_hid.transpose(-1, -2) \
                           .reshape(1, x_hid.shape[-1] * 2, -1) \
                           .transpose(-1, -2)
            elif self.out_type == 'final_mean':
                # 1. �������
                x_m = x_hid.transpose(-1, -2) \
                           .mean(dim=0) \
                           .transpose(-1, -2)
            # 2. � ���� ����� ��� ����� x_e_shape: forma x �����
            # ����������� ���� ����������. ������ ��������� �������� ���
            # ���� ������, �.�., ������� �� [N * S, C, E] � [N * S, E].
            # ��� ���� �� ����� ������ ���� ��� ����� ����. ��������
            # ����� ������:
            x = torch.zeros(x_e_shape[0], x_e_shape[2],
                            dtype=torch.float, device=device)
            # 3. ������ x_m ����� �� �� �����, ��� � ����� ��������������,
            # �� ��������� �������� ��� ���. �������� ��� � �� �� ����� x,
            # ������ ����� (���������� �� �� �����)
            x[mask] = x_m
            # ������ � ��� x � ����� [N * S, E]. ��������� ��� � �����
            # ����������� ���� [N, S, E]:
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

    def __init__(self, alphabet_size,
                 emb_layer=None, emb_dim=300, pad_idx=0,
                 kernels=[3, 4, 5]):
        super().__init__()

        self.kernels = list(kernels)
        self.alphabet_size = alphabet_size
        self.emb_dim = None if emb_layer else emb_dim
        self.pad_idx = pad_idx

        self._emb_l = emb_layer if emb_layer else \
                      nn.Embedding(alphabet_size, emb_dim,
                                   padding_idx=pad_idx)

        self._conv_ls = nn.ModuleList(
            [nn.Conv1d(in_channels=self._emb_l.embedding_dim,
                      out_channels=self._emb_l.embedding_dim,
                      padding=0, kernel_size=kernel)
                 for kernel in kernels]
        )

    def forward(self, x, lens):
        """
        x: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device
        max_len = x.shape[-1]

        # ��������� ����� ����� ��������:
        # [#����������� � �����:#����� � �����������:#������ � �����] <==> [N, S, C]
        x_shape = x.shape
        # ��� ����� �� ���� ������ �������� ������. ��-�� �������� ��� �����������
        # ����� �����, ��� ��� ��������� ������� �� ����� ���������. ������ � ���
        # ���� ������� ���� �� ���� ����: [N, S, C] --> [N * S, C]
        # �����: ����� ��-�� �������� ���� ��� ����� �����. ��� ���� ������ ������,
        #        �.�., ������� ������ �� ��������
        x = x.flatten(end_dim=1)

        # ��������� ����� ���� ����������� ���������� (���������):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        # ��������� ��� ����� ����
        x_e_shape = x.shape

        # ������ ������ ���� ���� ����. ��� ����� � ������ �������� ���� ����
        # � ������������ ��������� ���� � �������� ���� ����, ��������� ������
        # �� ��������, ����� ���� �������� �� ������ ��� ��, ��� �� �� �����
        # ������� �������: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # ������ �������������� �������� x_ch_cnn � lens_ch0 � pack_padded_sequence
        # � ������� ��� ��� ������������ ������� ���� ������� �����. ������
        # ����������, ��� ��� ������� ������� ����� �� ���������. �������:
        # 1. ������ ����� ���� ����: True, ���� ����� �� ����� ���� (����� ��
        # �� ��������)
        mask = lens0 != 0
        # 2. ������� ������� ����� �� ������� ���������� �����������
        x_m = x[mask]
        # 3. ������� ������� �����, ����� ������ ���� �������������� ������
        # ������� ���������� �����������
        lens0_m = lens0[mask]

        # ������� ����� ��� ���������� ���� � ��������
        # NB! CNN ��������� �� ���� ������ ������������
        #     [batch_size, hidden_size, seq_len] 
        # CNN tensor input shape:
        #     [nonzero(N * S), E, C]
        # tensor after convolution shape:
        #     [nonzero(N * S), E, C - cnn_kernel_size + 1]
        # tensor after pooling:
        #     [nonzero(N * S), E, (C - cnn_kernel_size + 1) - (pool_kernel_size - 1)]
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
        # ������ x_m ����� ����� [N * S, E, pool_concat]. ��� ����� ��������
        # � � ���� [N * S, E]. ��� �� ����� ��������������� ��� �������, ��
        # ������ ���� ������� � ������ �������� ������������ ���������
        #x_m = torch.mean(x_m, -1)
        x_m = F.relu(torch.max(x_m, -1)[0])

        # � ���� ����� ��� ����� x_e_shape: forma x �����
        # ����������� ���� ����������. ������ ��������� �������� ���
        # ���� ������, �.�., ������� �� [N * S, C, E] � [N * S, E].
        # ��� ���� �� ����� ������ ���� ��� ����� ����. ��������
        # ����� ������:
        x = torch.zeros(x_e_shape[0], x_e_shape[2],
                        dtype=torch.float, device=device)
        # 3. ������ x_m ����� �� �� �����, ��� � ����� ��������������,
        # �� ��������� �������� ��� ���. �������� ��� � �� �� ����� x,
        # ������ ����� (���������� �� �� �����)
        x[mask] = x_m
        # ������ � ��� x � ����� [N * S, E]. ��������� ��� � �����
        # ����������� ���� [N, S, E]:
        x = x.view(*x_shape[:-1], -1)

        return x

    def extra_repr(self):
        return '{}, {}, pad_idx={}, kernels={}'.format(
            self.alphabet_size, self.emb_dim, self.pad_idx, self.kernels
        ) if self.emb_dim else \
        '{}, external embedding layer, kernels={}'.format(
            self.alphabet_size, self.kernels
        )
