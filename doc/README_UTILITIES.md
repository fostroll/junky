<h2 align="center">junky lib: PyTorch utilities</h2>

## Utilities

The lib contains some utilities to use in *PyTorch* models.

```python
import junky
junky.get_max_dims(array, max_dims=None, str_isarray=False, dim_no=0)
```
Returns max sizes of nested **array** on the all levels of nestedness.

Here, **array**: nested `lists` or `tuples`.

**str_isarray**: if `True`, strings are treated as arrays of chars and form
additional dimension.

**dim_no**: for internal used only. Stay it as it is.

```python
junky.insert_to_ndarray(array, ndarray, shift='left')
```
Inserts a nested **array** with data of any allowed for *numpy* type to the
*numpy* **ndarray**.

**NB:** all dimensions of **ndarray** must be no less than sizes of
corresponding subarrays of the **array**.

Params:

**array**: nested `list`s or `tuple`s.

**shift**: how to place data of **array** to **ndarray** in the case if size
of some subarray less than corresponding dimension of **ndarray**. Allowed
values:<br/>
`'left'`: shift to start;<br/>
`'right'`: shift to end;<br/>
`'center'`: place by center (or 1 position left from center if evennesses of
subarray's size and ndarray dimension are not congruent);<br/>
`'rcenter'`: the same as `'center'`, but if evennesses are not congruent, the
shift will be 1 position right.

```python
junky.pad_array(array, padding_value=0)
```
Converts nested **array** with data of any allowed for *numpy* type to
`numpy.ndarray` with **padding_value** instead of missing data.

**array**: nested `list`s or `tuple`s.

Returns `numpy.ndarray` with padded **array**.

```python
junky.pad_array_torch(array, padding_value=0, **kwargs)
```
Just a wropper for the `pad_array()` method that returns `torch.Tensor`.

**kwargs**: keyword args for the `torch.tensor()` method.
