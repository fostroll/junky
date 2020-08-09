# -*- coding: utf-8 -*-
# junky lib: utils
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of utilities to use with PyTorch.
"""
from collections.abc import Iterable
from collections import OrderedDict, Counter
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch
import math

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
            padding_tensor = t.new_full(t.shape[1:], padding_tensor)
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

def kwargs (**kwargs):
    """Return any keyword arguments as a dict."""
    return kwargs

def kwargs_nonempty (**kwargs):
    """Return any keyword arguments with non-empty values as a dict."""
    return {x: y for x, y in kwargs.items() if y}

def get_func_params(func, func_locals, keep_self=False):
    """Return params of *func* as `args` and `kwargs` arrays. Method is called
    inside *func*; *func_locals* is an output of the locals() call inside
    *func*.

    If *keep_self* is ``True``, don't remove `self` variable from `args`."""
    all_args = func.__code__.co_varnames[:func.__code__.co_argcount]
    n_kwargs = len(func.__defaults__)
    args = [func_locals[x] for x in all_args[:-n_kwargs]
                if x != 'self' or keep_self]
    kwargs = {x: func_locals[x] for x in all_args[-n_kwargs:]}
    return args, kwargs

def filter_embeddings(pretrained_embs, corpus, min_abs_freq=1, save_name=None,
                   include_emb_info=False, pad_token=None, unk_token=None,
                   extra_tokens=None):
    """Filters pretrained word embeddings' vocabulary, leaving only tokens
    that are present in the specified `corpus` which are more frequent than
    minimum absolute frequency `min_abs_freq`. This method allows to
    significantly reduce memory usage and speed up word embedding process.
    The drawbacks include lower performance on unseen data.

    Args:

    **vectors**: file with pretrained word vectors in text format (not
    binary), where the first line is
    `<vocab_size> <embedding_dimensionality>`.

    **corpus**: a list of lists or tuples with already tokenized sentences.
    Filtered result will not contain any tokens outside of this corpus.

    **min_abs_freq** (`int`): minimum absolute frequency; only tokens the
    frequency of which is equal or greater than this specified value will be
    included in the filtered word embeddings. Default `min_abs_freq=1`,
    meaning all words from the corpus that have corresponding word vectors in
    `pretrained_embs` are preserved.

    **save_name**(`str`): if specified, filtered word embeddings are saved in
    a file with the specified name.

    **include_emb_info**(`bool`): whether to include `<vocab_size> <emb_dim>`
    as the first line to the filtered embeddings file. Default is `False`,
    embedding info line is skipped. Relevant only if `save_name` is not None.

    For the arguments below note, that typically pretrained embeddings already
    include PAD or UNK tokens. But these argumets are helpful if you need to
    specify your custom pad/unk/extra tokens or make sure they are at the top
    of the vocab (thus, pad_token will have index=0 for convenience).

    **pad_token** (`str`): custom padding token, which is initialized with
    zeros and included at the top of the vocabulary.

    **unk_token** (`str`): custom token for unknown words, which is
    initialized with small random numbers and included at the top of the
    vocabulary.

    **extra_tokens** (`list`): list of any extra custom tokens. For now, they
    are initialized with small random numbers and included at the top of the
    vocabulary. Typically, used for special tokens, e.g. start/end tokens etc.

    If `save_name` is specified, saves the filtered vocabulary. Otherwise,
    returns word2index OrderedDict and a numpy array of corresponding word
    vectors.
    """

    filter_vocab = OrderedDict(
        sorted(
            {k: v
             for k, v in Counter(chain.from_iterable(corpus)).items()
             if v>=min_abs_freq}.items(),
            key=lambda t: t[1]))

    word2index = OrderedDict()
    vectors = []

    # model in vec or txt format
    # (not binary, first line is <vocab_size> <emb_dim>)
    word2vec_file = open(pretrained_embs)

    n_words, embedding_dim = word2vec_file.readline().split()
    n_words, embedding_dim = int(n_words), int(embedding_dim)

    if pad_token:
        # Zero vector for PAD
        vectors.append(np.zeros((1, embedding_dim)))
        word2index[pad_token] = len(word2index)

    if unk_token:
        # Initializing UNK vector with small random numbers
        vectors.append(
            np.random.rand(1, embedding_dim) / math.sqrt(embedding_dim))
        word2index[unk_token] = len(word2index)

    if extra_tokens:
        # random-small-number vectors for extra_tokens
        for x_t in extra_tokens:
            vectors.append(
                np.random.rand(1, embedding_dim) / math.sqrt(embedding_dim))
            word2index[x_t] = len(word2index)

    progress_bar = tqdm(desc='Filtering vectors',
                        total=n_words, mininterval=2)

    while True:
        line = word2vec_file.readline().strip()

        if not line:
            break

        current_parts = line.split()
        current_word = ' '.join(current_parts[:-embedding_dim])

        if current_word in filter_vocab:

            word2index[current_word] = len(word2index)

            current_vectors = current_parts[-embedding_dim:]
            current_vectors = np.array(list(map(float, current_vectors)))
            current_vectors = np.expand_dims(current_vectors, 0)

            vectors.append(current_vectors)

        progress_bar.update(1)

    progress_bar.close()
    word2vec_file.close()

    vectors = list(np.concatenate(vectors))

    if save_name:
        with open(save_name, 'w') as f:
            if include_emb_info:
                print(len(word2index), embedding_dim, file=f)

            for word, vector in tqdm(zip(word2index.keys(), vectors),
                         desc='Saving filtered vectors', total=len(vectors)):
                print(word, ' '.join(str(v) for v in vector),
                      end=' \n', file=f)
    else:
        return word2index, vectors
