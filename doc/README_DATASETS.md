<h2 align="center">junky lib: PyTorch utilities</h2>

## Datasets

The lib contains a few descendants of `torch.utils.data.Dataset` that can be
used together with *PyTorch* models.

### TokenDataset

Maps tokenized sentences to sequences of their tokens' indices.

```python
from junky.dataset import TokenDataset
ds = TokenDataset(sentences, unk_token=None, pad_token=None,
                  extra_tokens=None, transform=False, skip_unk=False,
                  keep_empty=False, int_tensor_dtype=int64)
```
Params:

**sentences** (`list([list([str])])`): already tokenized sentences.

**unk_token** (`str`): add a token for tokens that are not present in the
internal dict (further, **&lt;UNK&gt;**).

**pad_token** (`str`): add a token for padding (further, **&lt;PAD&gt;**).

**extra_tokens** (`list([str])`): add tokens for any other purposes.

**transform**: if `True`, invoke `.transform(sentences, save=True)` right
after object creation.

**skip_unk**, **keep_empty**: params for the `.transform()` method.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

#### Attributes

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

`ds.transform_dict` (`dict({str: int})`): tokens to indices mapping.

`ds.reconstruct_dict` (`dict({int: str})`): indices to tokens mapping.

`ds.unk` (`int`): index of **&lt;UNK&gt;**.

`ds.pad` (`int`): index of **&lt;PAD&gt;**.

`ds.data` (`list([torch.Tensor([int])])`): the source of `Dataset`.

Generally, you don't need to change any attribute directly.

#### Methods

To re-initialize the `Dataset`, call
```python
ds.fit(sentences, unk_token=None, pad_token=None, extra_tokens=None)
```
The method fits the `Dataset` model to **sentences**. All params here has the
same meaning as in the constructor, and this method is invoked from the
constructor. So, you need it only if you want to reuse already existing
`Dataset` object for some new task with a different set of tokens. Really, you
can just create a new object for that.

```python
idx = ds.token_to_idx(token, skip_unk=False)
```
Returns the index of the **token**. If the **token** is not present in the
internal dict, returns index of **&lt;UNK&gt;** token or `None` if it's not
defined or **skip_unk** is `True`.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
present in the dictionary. So, if exactly that token will be met, its index
will be returned by the method even with `skip_unk=True` param.

```python
token = ds.idx_to_token(idx, skip_unk=False, skip_pad=True)
```
Returns the **token** by its index. If the **idx** is not present in the
internal dict, returns **&lt;UNK&gt;** or empty string if it's not defined or
**skip_unk** is `True`. If **skip_pad** is `True` (default), index of
**&lt;PAD&gt;** will be replaced to empty string, too.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this tokens is
present in the dictionary. So, if exactly its index will be met, the token
will be returned by the method even with `skip_unk=True` params.
Alternatively, with `skip_pad=True`, the method removes padding if the
**&lt;PAD&gt;** token is present in the dictionary.

```python
ids = ds.transform_tokens(tokens, skip_unk=False)
```
Converts a token or a `list` of tokens to the corresponding index|`list` of
indices. If **skip_unk** is `True`, unknown tokens will be skipped.

```python
tokens = ds.reconstruct_tokens(ids, skip_unk=False, skip_pad=True)
```
Converts an index or a `list` of indices to the corresponding
token|`list` of tokens. If **skip_unk** is `True`, unknown indices will be
replaced to empty strings. If **skip_pad** is `True` (default), indices of
**&lt;PAD&gt;** will be replaced to empty strings, too.

```python
ds.transform(sentences, skip_unk=False, keep_empty=False, save=True)
```
Converts tokenized **sentences** to the sequences of the corresponding indices
and adjust their format for `torch.utils.data.Dataset`. If **skip_unk** is
`True`, unknown tokens will be skipped. If **keep_empty** is `False`
(default), we'll remove sentences that have no data after converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **save** is `False`, the method returns the result of the transformation.
Elsewise, `None` is returned.

```python
sentences = ds.reconstruct(sequences, skip_unk=False, skip_pad=True,
                           keep_empty=False)
```
Converts **sequences** of indices in `Dataset` format to the **sentences**
of the corresponding tokens. If **skip_unk** is `True`, unknown indices
will be skipped. If **skip_pad** is `True` (default), **&lt;PAD&gt;**
tokens will be removed from the result. If **keep_empty** is `False`
(default), we'll remove sentences that have no data after converting.

```python
ds.fit_transform(sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, skip_unk=False, keep_empty=False,
                 save=True)
```
Fits the `Dataset` model to **sentences** and then transforms them. In
sequence, calls the `.fit()` and the `.transform()` methods. All params are
the params of those methods. Returns the return of the `.transform()`.

Consider new object creation instead. One boy is invoked this method and died.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `TokenDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes attributes will be copied.

```python
ds.save(file_path, with_data=True)
```
Saves the `TokenDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. The model and all
other attributes will be saved.

```python
ds = TokenDataset.load(file_path):
```
Load the `TokenDataset` object from **file_path**.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
ds.create_loader(self, batch_size=32, shuffle=False, num_workers=0, **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0` don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't bear it. Better, create
several instances of `DataLoader` for **ds** (each with `workers=0`) and use
them in parallel.

### CharDataset

Maps tokenized sentences to sequences of lists of indices of their tokens'
chars.

```python
from junky.dataset import CharDataset
ds = CharDataset(sentences, unk_token=None, pad_token=None,
                  extra_tokens=None, allowed_chars=None, exclude_chars=None,
                  transform=False, skip_unk=False, keep_empty=False,
                  int_tensor_dtype=int64, min_len=None)
```
Params:

**sentences** (`list([list([str])])`): already tokenized sentences.

**unk_token** (`str`): add a token for characters that are not present in the
dict (further, **&lt;UNK&gt;**).

**pad_token** (`str`): add a token for padding (further, **&lt;PAD&gt;**).

**extra_tokens** (`list([str])`): add tokens for any other purposes.

**allowed_chars** (`str`|`list([str])`): if not `None`, all charactes not from
**allowed_chars** will be removed.

**exclude_chars** (`str`|`list([str])`): if not `None`, all charactes from
**exclude_chars** will be removed.

**transform**: if `True`, invoke `.transform(sentences, save=True)` right
after object creation.

**skip_unk**, **keep_empty**: params for the `.transform()` method.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**min_len** (`int`): if specified, `collate_fn` of internal `DataLoader` will
pad sentences in batch that are shorter than this value.

#### Attributes

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

`ds.transform_dict` (`dict({str: int})`): tokens to indices mapping.

`ds.reconstruct_dict` (`dict({int: str})`): indices to tokens mapping.

`ds.unk` (`int`): index of **&lt;UNK&gt;**.

`ds.pad` (`int`): index of **&lt;PAD&gt;**.

`ds.data` (`list([list([torch.Tensor([int])])])`): the source of `Dataset`.

`ds.min_len` (`int`): if specified, `collate_fn` of internal `DataLoader` will
pad sentences in batch that are shorter than this value.

Generally, you don't need to change any attribute directly.

#### Methods

To re-initialize the `Dataset`, call
```python
ds.fit(sentences, unk_token=None, pad_token=None, extra_tokens=None,
       allowed_chars=None, exclude_chars=None)
```
The method fits the `Dataset` model to **sentences**. All params here has the
same meaning as in the constructor, and this method is invoked from the
constructor. So, you need it only if you want to reuse already existing
`Dataset` object for some new task with a different set of tokens. Really, you
can just create a new object for that.

```python
idx = char_to_idx(char, skip_unk=False)
```
Returns the index of the **char**. If the **char** is not present in the
internal dict, returns index of **&lt;UNK&gt;** token or `None` if it's not
defined or **skip_unk** is `True`.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
present in the dictionary. So, if exactly that token will be met, its index
will be returned by the method even with `skip_unk=True` param.

```python
char = idx_to_char(idx, skip_unk=False, skip_pad=True):
```
Returns the **token** by its index. If the **idx** is not present in the
internal dict, returns **&lt;UNK&gt;** or empty string if it's not defined or
**skip_unk** is `True`. If **skip_pad** is `True` (default), index of
**&lt;PAD&gt;** will be replaced to empty string, too.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this tokens is
present in the dictionary. So, if exactly its index will be met, the token
will be returned by the method even with `skip_unk=True` params.
Alternatively, with `skip_pad=True`, the method removes padding if the
**&lt;PAD&gt;** token is present in the dictionary.

```python
ids = ds.token_to_ids(token, skip_unk=False)
```
Converts a token or a `list` of characters to the `list` of indices of its
chars. If some characters are not present in the internal dict, we'll use the
index of **&lt;UNK&gt;** for them, or empty strings if it's not defined or
**skip_unk** is `True`. If **skip_pad** is `True`, padding indices will be
replaced to empty string, too.

```python
token = ds.ids_to_token(idx, skip_unk=False, skip_pad=True, aslist=False)
```
Convert a `list` of indices to the corresponding token or a `list` of
characters. If some indices are not present in the internal dict, we'll use
**&lt;UNK&gt;** token for them, or `None` if it's not defined.

If **aslist** is `True`, we want `list` of characters instead of token as the
result.

```python
ids = ds.transform_tokens(tokens, skip_unk=False)
```
Converts a token or a sequence of tokens to the corresponding list or a
sequence of lists of indices. If skip_unk is `True`, unknown tokens will be
skipped.

```python
tokens = ds.reconstruct_tokens(ids, skip_unk=False, skip_pad=True)
```
Converts a `list` of indices or a sequence of lists of indices to the
corresponding token or a sequence of tokens. If skip_unk is `True`, unknown
indices will be skipped (or replaced to empty strings, if *aslist* is `True`).
If *skip_pad* is `True`, padding indices also will be removed or replaced to
empty strings.

If **aslist** is `True`, we want list of characters instead of tokens in the
result.

```python
ds.transform(sentences, skip_unk=False, keep_empty=False, save=True)
```
Converts tokenized **sentences** to the sequences of the lists of the indices
corresponding to token's chars and adjust their format for
`torch.utils.data.Dataset`. If **skip_unk** is `True`, unknown chars will be
skipped. If **keep_empty** is `False` (default), we'll remove tokens and
sequences that have no data after converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **save** is `False`, the method returns the result of the transformation.
Elsewise, `None` is returned.

```python
sentences = ds.reconstruct(sequences, skip_unk=False, skip_pad=True,
                           keep_empty=False)
```
Converts **sequences** of the lists of the indices in `Dataset` format to the
**sentences** of the corresponding tokens. If **skip_unk** is `True`, unknown
indices will be skipped. If **skip_pad** is `True` (default), **&lt;PAD&gt;**
tokens will be removed from the result. If **keep_empty** is `False`
(default), we'll remove sentences that have no data after converting.

If **aslist** is `True`, we want list of characters instead of tokens in the
result.

```python
ds.fit_transform(sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, skip_unk=False, keep_empty=False,
                 save=True)
```
Fits the `Dataset` model to **sentences** and then transforms them. In
sequence, calls the `.fit()` and the `.transform()` methods. All params are
the params of those methods. Returns the return of the `.transform()`.

Consider new object creation instead. Anyhow, before call that method, call
the doctor.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `CharDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes attributes will be copied.

```python
ds.save(file_path, with_data=True)
```
Saves the `CharDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. The model and all
other attributes will be saved.

```python
ds = CharDataset.load(file_path):
```
Load the `CharDataset` object from **file_path**.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
ds.create_loader(self, batch_size=32, shuffle=False, num_workers=0, **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0` don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't bear it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`) and
use them in parallel.

### WordDataset

Maps tokenized sentences to sequences of their tokens' vectors.

```python
from junky.dataset import WordDataset
ds = WordDataset(emb_model, vec_size,
                 unk_token=None, unk_vec_norm=1e-2,
                 pad_token=None, pad_vec_norm=0.,
                 extra_tokens=None, extra_vec_norm=1e-2,
                 sentences=None, skip_unk=False, keep_empty=False,
                 float_tensor_dtype=float32, int_tensor_dtype=int64)
```
Params:

**emb_model**: dict or any other object that allow the syntax
`vector = emb_model[word]` and `if word in emb_model:`.

**vec_size** (`int`): the length of the word's vector.

**unk_token** (`str`): add a token for tokens that are not present in the
internal dict (further, **&lt;UNK&gt;**).

**unk_vec_norm** (`float`): the norm of the vector for **unk_token**.

**pad_token** (`str`): add a token for padding (further, **&lt;PAD&gt;**).

**pad_vec_norm** (`float`): the norm of the vector for **pad_token**.

**extra_tokens** (`list([str])`): add tokens for any other purposes.

**extra_vec_norm** (`float`): the norm of the vectors for **extra_tokens**.

**sentences** (`list([list([str])])`): already tokenized sentences of words.
If not `None`, they will be transformed and saved.

**skip_unk**, **keep_empty**: params for the `.transform()` method.

**float_tensor_dtype** (`torch.dtype`, default `torch.float32`): type for
float tensors. Don't change it.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

#### Attributes

`ds.emb_model`: model that transforms words to their vectors.

`ds.extra_model` (`dict({str: numpy.ndarray([float])})`): model that
transforms special tokens to their vectors.

`ds.vec_size`: the length of word's vector.

`ds.float_tensor_dtype` (`torch.dtype`): type for float tensors.

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

`ds.unk` (`numpy.ndarray([float])`): vector of **&lt;UNK&gt;**.

`ds.pad` (`numpy.ndarray([float])`): vector of **&lt;PAD&gt;**.

`ds.data` (`list([torch.Tensor([numpy.ndarray([float])])])`): the source of
`Dataset`.

Generally, you don't need to change any attribute directly.

#### Methods

```python
vec = ds.word_to_vec(token, skip_unk=False)
```
Converts a **token** to its **vector**. If the **token** is not present in the
model, return vector of **&lt;UNK&gt;** token or `None` if it's not defined.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
present in the model. So, if exactly that token will be met, its vector
will be returned by the method even with `skip_unk=True` param.

```python
vecs = ds.transform_words(words, skip_unk=False)
```
Converts a word or a `list` of words to the corresponding vector|`list` of
vectors. If **skip_unk** is `True`, unknown words will be skipped.

```python
ds.transform(sentences, skip_unk=False, keep_empty=False, save=True)
```
Converts tokenized **sentences** to the sequences of the corresponding vectors
and adjust their format for `torch.utils.data.Dataset`. If **skip_unk** is
`True`, unknown tokens will be skipped. If **keep_empty** is `False`
(default), we'll remove sentences that have no data after converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **save** is `False`, the method returns the result of the transformation.
Elsewise, `None` is returned.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `WordDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes attributes will be copied. The `ds.emb_model` attribute is copied
by link.

```python
emb_model = ds.save(file_path, with_data=True)
```
Saves the `WordDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. All other attributes
will be saved but `ds.emb_model` that is returned by the method for you saved
it if need by its own method.

```python
ds = WordDataset.load(file_path, emb_model):
```
Load the `WordDataset` object from **file_path**. You should specify
**emb_model** that you used during object's creation.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
ds.create_loader(self, batch_size=32, shuffle=False, num_workers=0, **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0` don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't bear it. Better, create
several instances of `DataLoader` for **ds** (each with `workers=0`) and use
them in parallel.

### FrameDataset

A frame for use several objects of `junky.dataset.*Dataset` conjointly. All
the `Datasets` must have the data of equal length.

```python
from junky.dataset import FrameDataset
ds = FrameDataset()
```

#### Attributes

`ds.datasets` (`dict({str: [junky.dataset.BaseDataset, int, kwargs]})`): the
list of `Datasets`. Format: {name: [`Dataset`, <number of data columns>,
<collate kwargs>]}

Generally, you don't need to change any attribute directly.

#### Methods

```python
ds.add(name, dataset, **collate_kwargs)
```
Adds **dataset** with a specified **name**.

Param **collate_kwargs** is a keyword arguments for the **dataset**'s
`._frame_collate()` method.

```python
ds.remove(name)
```
Removes `Dataset` with a specified **name** from model `ds.datasets`.

```python
ds_ = ds.get(name)
```
Returns `tuple(dataset, collate_kwargs)` for the `Dataset` with a specified
**name**.

```python
ds_list = ds.list()
```
Returns names of the added `Datasets` in order of addition.

```python
transform(sentences, skip_unk=False, keep_empty=False, save=True)
```
Invoke `.transform(sentences, skip_unk, keep_empty, save)` methods for all
added `Dataset` objects.

If **save** is `False`, we'll return the stacked result of objects' returns.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `FrameDataset` object. The **with_data** param is
defined if the nested datasets will be cloned with their data sources or
without.

```python
ds.save(file_path, with_data=True)
```
Saves the `FrameDataset` object to **file_path**.  The **with_data** param is
defined if the nested datasets will be saved with their data sources or
without.

```python
ds = FrameDataset.load(file_path):
```
Load the `FrameDataset` object from **file_path**.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods of all nested datasets.

```python
ds.create_loader(self, batch_size=32, shuffle=False, num_workers=0, **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0` don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't bear it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`) and
use them in parallel.


