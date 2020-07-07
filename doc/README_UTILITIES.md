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
values:<br/>
`'left'`: shift to start;<br/>
`'right'`: shift to end;<br/>
`'center'`: place in center (or 1 position left from center if evennesses of
subarray's size and ndarray dimension are not congruent);<br/>
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
Pad the `seq` dimension of **sequences** of shape `(batch, seq, features)`
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
Re-init random number generators.

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
Append **vectors** with a vector that has norm equals to the mean norm
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
def kwargs_nonempty (**kwargs):
```
Returns any keyword arguments with non-empty values as a `dict`.

```python
def get_func_params(func, func_locals):
```
Returns params of **func** as `args` and `kwargs` arrays.

Method is called inside **func**; **func_locals** is an output of the
`locals()` call inside **func**.
