# -*- coding: utf-8 -*-
# junky lib: utils
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of utilities to use with PyTorch.
"""
import numpy as np
import torch

CPU = torch.device('cpu')


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

def pad_sequences_with_tensor(sequences, padding_tensor=0.):
    """Pad the `seq` dimension of *sequences* of shape
    `(batch, seq, features)` with *padding_tensor*.

    :param sequences: the list of sequences of shape `(seq, features)`.
    :type sequences: list([torch.Tensor(seq, features)]]
    :param padding_tensor: scalar or torch.Tensor of the shape of `features`
        to pad *sequences*.
    :type padding_tensor: torch.Tensor(features)|float|int
    :return: padded list converted to torch.Tensor.
    :rtype: torch.Tensor(batch, seq, features)
    """
    t = sequences[0]
    device = t.get_device() if t.is_cuda else CPU
    N, S = len(sequences), max(s.shape[0] for s in sequences)
    if not isinstance(padding_tensor, torch.Tensor):
        if isinstance(padding_tensor, Iterable):
            padding_tensor = torch.tensor(padding_tensor, device=device)
        else:
            padding_tensor = t.new_full(t.shape[2:], padding_tensor)
    #res = padding_tensor.to(device).expand(N, S, *t.shape[2:]).clone()
    res = padding_tensor.to(device).repeat(N, S, 1)

    for s_i, s in enumerate(sequences):
        res[s_i, :s.shape[0]] = s

    return res

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

def get_rand_vector(shape, norm, shift=0., dtype=np.float64):
    """Create random vector with the norm given.

    :param shape: the shape of the new vector.
    :type shape: tuple|list
    :param norm: the norm of the new vector.
    :type norm: float
    :param shift: relative shift of the new vector's mean from 0.
    :type shift: float
    :rtype: numpy.ndarray
    """
    if norm:
        vector = np.random.rand(*shape).astype(dtype) - .5 + shift
        vector *= norm / np.linalg.norm(vector)
    else:
        vector = np.zeros(shape)
    return vector

def add_mean_vector(vectors, axis=0, shift=0., scale=1.):
    """Append *vectors* with a vector that has norm equals to the mean norm
    (possibly, scaled) of *vectors*.

    :param vectors: the array of floats.
    :type vectors: numpy.ndarray
    :param axis: the axis of *vectors* along which the new vector is appended.
    :type axis: int|tuple(int)
    :param shift: relative shift of the new vector's mean from 0.
    :type shift: float
    :param scale: the coef to increase (or decrease if *scale* < 1) the norm
        or the new vector.
    :type scale: float > 0
    :return: vectors appended
    :rtype: numpy.ndarray
    """
    norm = np.linalg.norm(vectors.shape[:axis] + vectors.shape[axis + 1:],
                          vectors, axis=axis).mean() * scale
    norm = np.linalg.norm(vectors, axis=tuple(
        list(range(0)) + list(range(1, len(emb_model.vectors.shape)))
    )).mean()
    vector = np.expand_dims(get_rand_vector(norm, shift, dtype=vectors.dtype))
    return np.append(vectors, vector, axis=axis)

def absmax(array, axis=None):
    """Returns an array with absolute maximum values of *array* according to
    *axis*.

    :type array: numpy.ndarray|list([numpy.ndarray])
    :type axis: int
    :rtype: numpy.ndarray
    """
    if not isinstance(array, np.ndarray):
        array = list(array)
        assert len(array) > 0, 'ERROR: array has no data'
        assert isinstance(array[0], np.ndarray), \
               'ERROR: array has unsupported type'
        array = np.stack(array, axis=-1 if axis is None else axis)

    if axis is None:
        res = array.max()
        res_min = array.min()
        if -res_min > res:
            res = res_min

    else:
        amax = np.broadcast_to(
            np.expand_dims(
                abs(array).argmax(axis=axis), axis=axis
            ), array.shape
        )

        shape = [(*[None] * len(amax.shape))]
        shape[axis] = ...
        mask = np.broadcast_to(
            np.arange(amax.shape[axis])[tuple(shape)],
            array.shape
        )

        res = array.copy()
        res[mask != amax] = 0

        res = res.sum(axis=axis)

    return res

def absmax_torch(tensors, dim=None):
    """Returns a tensor with absolute maximum values of *tensors* according to
    *dim*.

    :type tensors: torch.Tensor|list([torch.Tensor])
    :type axis: int
    :rtype: torch.Tensor
    """
    if not isinstance(tensors, torch.Tensor):
        tensors = list(tensors)
        assert len(tensors) > 0, 'ERROR: tensors has no data'
        assert isinstance(tensors[0], torch.Tensor), \
               'ERROR: tensors have unsupported type'
        tensors = torch.stack(tensors, dim=-1 if dim is None else dim)

    if dim is None:
        res = tensors.max()
        res_min = tensors.min()
        if -res_min > res:
            res = res_min

    else:
        amax = tensors.abs() \
                      .argmax(dim=dim) \
                      .unsqueeze(dim=dim) \
                      .expand(tensors.shape)

        shape = [(*[None] * len(amax.shape))]
        shape[dim] = ...
        mask = torch.arange(amax.shape[dim])[tuple(shape)] \
                    .expand(amax.shape)

        if tensors.is_cuda:
            mask = mask.to(tensors.device)

        res = tensors.clone()
        res[mask != amax] = 0

        res = res.sum(dim=dim)

    return res
