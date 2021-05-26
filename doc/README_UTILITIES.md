<h2 align="center">junky lib: PyTorch utilities</h2>

## Utilities

The lib contains a bunch of utilities to use with *PyTorch* models or without.

```python
import junky
max_dims = junky.get_max_dims(array, str_isarray=False, max_dims=None,
                              dim_no=0)
```
Returns max sizes of nested **array** on all levels of nestedness.

Here, **array**: nested lists or tuples.

**str_isarray**: if `True`, strings are treated as arrays of chars and form
additional dimension.

**max_dims**, **dim_no**: for internal use only. Leave it as they are.

```python
junky.insert_to_ndarray(array, ndarray, shift='left')
```
Inserts a nested **array** with data of any allowed *numpy* type to the
*numpy* **ndarray**.

**NB:** all dimensions of **ndarray** must not be less than sizes of
corresponding subarrays of the **array**.

Params:

**array**: nested lists or tuples.

**shift**: how to place data of **array** to **ndarray** in case if the size
of some subarray is less than corresponding dimension of **ndarray**. Allowed
values:<br />
`'left'`: shift to start;<br />
`'right'`: shift to end;<br />
`'center'`: place in center (or 1 position left from center if evennesses of
subarray's size and ndarray dimension are not congruent);<br />
`'rcenter'`: the same as `'center'`, but if evennesses are not congruent, the
shift will be 1 position right.

```python
array = junky.pad_array(array, padding_value=0)
```
Converts nested **array** with data of any allowed *numpy* type to
`numpy.ndarray` with **padding_value** instead of missing data.

**array**: nested lists or tuples.

Returns `numpy.ndarray` with padded **array**.

```python
array = junky.pad_array_torch(array, padding_value=0, **kwargs)
```
Just a wrapper for the `pad_array()` method that returns `torch.Tensor`.

**kwargs**: keyword args for the `torch.tensor()` method.

```python
padded_seq = junky.pad_sequences_with_tensor(sequences, padding_tensor=0.)
```
Pads the `seq` dimension of **sequences** of shape `(batch, seq, features)`
with **padding_tensor**.

Params:

**sequences** (`list([torch.Tensor(seq, features)]`): the `list` of sequences
of shape `(seq, features)`.

**padding_tensor** (`torch.Tensor(features)`|`float`|`int`): scalar or
`torch.Tensor` of the shape of `features` to pad **sequences**.

Returns padded `list` converted to `torch.Tensor(batch, seq, features)`.

```python
junky.enforce_reproducibility(seed=None)
```
Re-inits random number generators.

```python
vec = junky.get_rand_vector(shape, norm, shift=0., dtype=float)
```
Creates random `numpy.ndarray` with the **shape**, **norm** and **dtype**
given.

Params:

**shape** (`tuple`|`list`): the shape of the new vector.

**norm** (`float`): the norm of the new vector.

**shift** (`float`): relative shift of the new vector's mean from `0`.

**dtype** (`numpy.dtype`): type of the new vector's.

```python
junky.add_mean_vector(vectors, axis=0, shift=0., scale=1.)
```
Appends **vectors** with a vector that has norm equals to the mean norm
(possibly, scaled) of **vectors**.

Params:

**vectors** (`numpy.ndarray`): the array of floats.

**axis** (`int`|`tuple(int)`): the axis of **vectors** along which the new
vector is appended.

**shift** (`float`): relative shift of the new vector's mean from `0`.

**scale** (`float` > `0`): the coef to increase (or decrease if **scale** <
`1`) the norm of the new vector.

Returns **vectors** appended.

```python
junky.absmax(array, axis=None)
```
Returns `numpy.ndarray` with absolute maximum values of **array** according to
**axis**.

Here, **array** is a `numpy.ndarray` or a `list` of `numpy.ndarray`, **axis**
is of `int` type.

```python
junky.absmax_torch(tensors, dim=None)
```
Returns `torch.Tensor` with absolute maximum values of **tensors** according
to **dim**

Here, **tensors** is a `torch.Tensor` or a `list` of `torch.Tensors`, **dim**
is of `int` type.

```python
kwargs = junky.kwargs(**kwargs)
```
Returns any keyword arguments as a `dict`.

```python
kwargs_nonempty (**kwargs)
```
Returns any keyword arguments with non-empty values as a `dict`.

```python
get_func_params(func, func_locals)
```
Returns params of **func** as `args` and `kwargs` arrays.

Method is called inside **func**; **func_locals** is an output of the
`locals()` call inside **func**.

If **keep_self** is `True`, don't remove `self` variable from `args`.

```python
cls = add_class_lock(cls, lock_name='lock', isasync=False, lock_object=None)
```
Adds additional lock property **lock_name** to class **cls**.

if **isasync** is `True` the lock property is of `asyncio.Lock` type.
Otherwise (default) it's of `threading.Lock` type.

It can be used as follows:

```python
from junky add_class_lock
from pkg import Cls

Cls = add_class_lock(Cls)
o = Cls()
#async with o.lock:  # if isasync is True
with o.lock:         # if isasync is False (default)
    # some thread safe operations here
    pass
```
Also, one can add lock to the particular object directly:
`o = add_class_lock(Cls())`.

If you need, you can use your own lock object. Use param **lock_object** for
that. In that case, param **isasync** is ignored.

```python
word2index, vectors = filter_embeddings(
    pretrained_embs, corpus, min_abs_freq=1, save_name=None,
    include_emb_info=False, pad_token=None, unk_token=None, extra_tokens=None
)
```
Filters pretrained word embeddings' vocabulary, leaving only tokens that are
present in the specified `corpus` which are more frequent than minimum
absolute frequency `min_abs_freq`. This method allows to significantly reduce
memory usage and speed up word embedding process. The drawbacks include lower
performance on unseen data.

Args:

**vectors**: file with pretrained word vectors in text format (not binary),
where the first line is `<vocab_size> <embedding_dimensionality>`.

**corpus**: a list of lists or tuples with already tokenized sentences.
Filtered result will not contain any tokens outside of this corpus.

**min_abs_freq** (`int`): minimum absolute frequency; only tokens the
frequency of which is equal or greater than this specified value will be
included in the filtered word embeddings. Default `min_abs_freq=1`, meaning
all words from the corpus that have corresponding word vectors in
`pretrained_embs` are preserved.

**save_name**(`str`): if specified, filtered word embeddings are saved in a
file with the specified name.

**include_emb_info**(`bool`): whether to include `<vocab_size> <emb_dim>` as
the first line to the filtered embeddings file. Default is `False`, embedding
info line is skipped. Relevant only if `save_name` is not None.

For the arguments below note, that typically pretrained embeddings already
include PAD or UNK tokens. But these argumets are helpful if you need to
specify your custom pad/unk/extra tokens or make sure they are at the top of
the vocab (thus, pad_token will have index=0 for convenience).

**pad_token** (`str`): custom padding token, which is initialized with zeros
and included at the top of the vocabulary.

**unk_token** (`str`): custom token for unknown words, which is initialized
with small random numbers and included at the top of the vocabulary.

**extra_tokens** (`list`): list of any extra custom tokens. For now, they are
initialized with small random numbers and included at the top of the
vocabulary. Typically, used for special tokens, e.g. start/end tokens etc.

If `save_name` is specified, saves the filtered vocabulary. Otherwise, returns
word2index OrderedDict and a numpy array of corresponding word vectors.
