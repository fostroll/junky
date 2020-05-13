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
