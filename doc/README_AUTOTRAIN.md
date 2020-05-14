<h2 align="center">junky lib: PyTorch utilities</h2>

## Autotrain

This is a tool for *PyTorch* *model*'s hyperparameters selection. May work in
parallel using multiple devices. If some of parallel threads die during
training (because of `MemoryError` of anything), their tasks will be redone
after all other threads have finished their work.

```python
import junky
junky.torch_autotrain(
    make_model_method, train_method, create_loaders_method=None,
    make_model_args=(), make_model_kwargs=None, make_model_fit_params=None,
    train_args=(), train_kwargs=None, devices=torch.device('cpu'),
    best_model_file_name='model.pt', best_model_device=None, seed=None
)
```
Args:

**make_model_method**: method to create the *model*. Returns the *model* and,
if specified, some other params that should be passed to **train_method**. The method
has a signature as follows:<br/>
`callable(*make_model_args, **make_model_kwargs,**fit_kwargs) ->
model|tuple(model, <other train args>)`.<br/>
Here, ***fit_kwargs*** - params that are constructed from
**make_model_fit_params**.

**train_method**: method to train and validate the *model*. Signature:<br/>
`train_method: callable(device, loaders, model, *other_train_args,
best_model_backup_method, log_prefix, *train_args, **train_kwargs) ->
<train statistics>`.<br/>
Here:<br/>
***device*** - one of the **devices** that is assigned to train the *model*;<br/>
***loaders*** - the return of **create_loaders_method** or `()` if
**create_loaders_method** is `None` (default);<br/>
***other_train_args*** - params returned by **make_model_method** besides the
*model* (if any). E.g.: *optimizer*, *criterion*, etc.;<br/>
***best_model_backup_method*** - method that saves the best *model* over
all runs. Signature:<br/>
`callable(best_model, best_model_score)`.<br/>
This method must be invoked in **train_method** to save the best *model*;<br/>
***log_prefix*** - prefix that should use **train_method** in the beginning of
any output. Elsewise, you can't distinct messages from parallel threads.

**create_loaders_method**: method to create `torch.utils.data.DataLoaders`
objects to use in **train_method**. Every thread creates it only once and then
passes to **train_method** of every *model* that this thread is assigned for.
The signature of **create_loaders_method**:<br/>
`callable() -> <loader>|tuple(<loaders>)`.<br/>
If `None` (default), **train_method** must create loaders by itself.

**Important:** you can't use one `DataLoader` in several threads. You must
have separate `DataLoader` for every thread; otherwise, your training is gonna
be broken.

**make_model_args**: positional args (of `tuple` type) for
**make_model_method**. Will be passed as is.

**make_model_kwargs**: keyword args (of `dict` type) for
**make_model_method**. Will be passed as is.

**make_model_fit_params**: a list of combinations of varying
**make_model_method**'s ***fit_kwargs*** among which we want to find the best.
The type of **make_model_fit_params**: iterable of iterables; nestedness is
unlimited. Examples:<br/>
`[('a', [50, 100]), ('b': [.1, .5])]` produces ***fit_kwargs***:
```python
{'a': 50, 'b': .1},
{'a': 50, 'b': .5},
{'a': 100, 'b': .1},
{'a': 100, 'b': .5};
```
`[('a', [50, 100]), [('b': [.1, .5])], [('b': None), ('c': ['X', 'Y'])]]`
produces
```python
{'a': 50, 'b': .1},
{'a': 50, 'b': .5},
{'a': 100, 'b': .1},
{'a': 100, 'b': .5},
{'a': 50, 'b': None, 'c': 'X'},
{'a': 50, 'b': None, 'c': 'Y'},
{'a': 100, 'b': None, 'c': 'X'},
{'a': 100, 'b': None, 'c': 'Y'}.
```

**train_args**: positional args (of `tuple` type) for **train_method**. Will
be passed as is.

**train_kwargs**: keyword args (or `dict` type) for **train_method**. Will be
passed as is.

**devices**: what devices to use for training. This can be a separate device, a
`list` of available devices, or a `dict` of available devices with max number
of simultaneous threads. The possible types are: `<device>`,
`tuple(<device>)`, `dict({<device>: int})`. Examples:<br/>
`torch.device('cpu')` - one thread on CPU (default);<br/>
`('cuda:0', 'cuda:1', 'cuda:2')` - 3 GPU, 1 thread on each;<br/>
`{'cuda:0': 3, 'cuda:1': 3}` - 2 GPU, 3 threads on each.<br/>
**NB:** `<device>` == `(<device>,)` == `{<device>: 1}`

**best_model_file_name**: file name for the best *model* when saving.
Default `'model.pt'`.

**best_model_device**: the device where the best *model* will be loaded. 
If `None`, the best *model* will not be loaded in memory.

The tool returns `tuple(best_model, best_model_name, best_model_score,
best_model_params, stats)`. Here:<br/>
***best_model*** - the best *model* if best_model_device is not `None`, else
`None`;<br/>
***best_model_name*** - the key of the best *model* stats;<br/>
***best_model_score*** - the score of the best *model*;<br/>
***best_model_params*** - ***fit_kwargs*** of the best *model*;<br/>
***stats*** - all returns of all **train_method**s. Format:<br/>
`[(<model name>, <model best score>, <model params>, <*train_method* return>),
...]`<br/>
***stats*** is sorted by `<model best score>`, in such a way that stats[0]
corresponds to the best *model*.

Sometimes, it's necessary to extract results from the ouput of
`torch_autotrain()`. The method to do so is:
```python
junky.parse_autotrain_log(log_fn, silent=False)
```
Here, **log_fn** is a file name of the `torch_autotrain()` log file.

**silent**: if `True`, suppress output.

Returns `list([tuple(<model name>, <model best score>, <model params>,
<is training finished>)]` sorted by `<model best score>`.

**NB:** if you use `torch_autotrain()` from *jupyter notebook*, you don't have
to copy only its output. Usually, you can just select and copy full text from
the *notebook* page and save it to "log" file. Then, pass this file to
`parse_autotrain_log()`.

If training of some model has not finished yet, it's name in output will be
started from `*` sign.
