<h2 align="center">junky lib: PyTorch utilities</h2>

## Datasets

The lib contains a few descendants of `torch.utils.data.Dataset` that can be
used together with *PyTorch* models.

### Table of Contents

0. [BaseDataset](#base)
1. [TokenDataset](#token)
2. [CharDataset](#char)
3. [WordDataset](#word)
4. [BertDataset](#bert)
5. [BertTokenizedDataset](#bertok)
6. [FrameDataset](#frame)
7. [DummyDataset](#dummy)
8. [LenDataset](#len)
9. [WordCatDataset](#wordcat)
10. [LabelDataset](#label)
11. [Examples](#examples)

### BaseDataset <a name="base"></a>

Base class for `junky.dataset.*Dataset` classes described below.

```python
ds = BaseDataset(data=None)
```

Params:

**data** (`list`): any list of data to save as Dataset source. `None` by
default.

#### Attributes

`self.data` (`list`): Dataset source

#### Methods

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `BaseDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes will be copied.

```python
ds.save(file_path, with_data=True, method='torch')
```
Saves the object to **file_path**. If **with_data** is `False`,
the `data` attribute of the saved object will be empty. The param **method**
can be either `'torch'` (default) or `'pickle'`.

```python
ds = TokenDataset.load(file_path, method=None):
```
Load previously saved `BaseDataset` object from **file_path**. The param
**method** can be either `'torch'` (default) or `'pickle'`. If the **method**
is `None` (default), we detect it by trial and error.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` for all the elements of the `Dataset` source
that have `torch.Tensor` or `torch.nn.Model` type. All the params are
transferred as is.

```python
ds.transform(data, append=False)
```

Saves any list of **data** as the Dataset source. If **append** is `True`, the
converted sentences will be appended to the existing Dataset source. Elsewise
(default), the existing Dataset source will be replaced.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 

To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                           collate_kwargs=None, loglevel=0)
```

Sequentially makes batches from **sentences** and calls
`.transform(batch, save=False, **transform_kwargs)` and
`._collate(batch, **collate_kwargs)` methods.

### TokenDataset  <a name="token"></a>

Maps tokenized sentences to sequences of their tokens' indices.

```python
from junky.dataset import TokenDataset
ds = TokenDataset(sentences, unk_token=None, pad_token=None,
                  extra_tokens=None, int_tensor_dtype=int64,
                  transform=False, skip_unk=False, keep_empty=False)
```
Params:

**sentences** (`list([list([str])])`): already tokenized sentences.

**unk_token** (`str`): add a token for tokens that are not present in the
internal dict (further, **&lt;UNK&gt;**).

**pad_token** (`str`): add a token for padding (further, **&lt;PAD&gt;**).

**extra_tokens** (`list([str])`): add tokens for any other purposes.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**transform**: if `True`, invoke `.transform(sentences, save=True)` right
after object creation.

**skip_unk**, **keep_empty**: params for the `.transform()` method.

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
The method fits the `Dataset` model to **sentences**. All params here have the
same meaning as in the constructor, and this method is invoked from the
constructor. So, you need it only if you want to reuse already existing
`Dataset` object for a new task with a different set of tokens. In fact, you
should just create a new object for that.

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
**&lt;PAD&gt;** will be replaced by an empty string, too.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
present in the dictionary. So, if exactly its index will be met, the token
will be returned by the method even with `skip_unk=True` param.
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
replaced by empty strings. If **skip_pad** is `True` (default), indices of
**&lt;PAD&gt;** will be replaced by empty strings, too.

```python
ds.transform(sentences, skip_unk=False, keep_empty=False, save=True,
             append=False)
```
Converts tokenized **sentences** to the sequences of the corresponding indices
and adjust their format for `torch.utils.data.Dataset`. If **skip_unk** is
`True`, unknown tokens will be skipped. If **keep_empty** is `False`
(default), we'll remove sentences that have no data after converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **append** is `True`, we'll append the converted sentences to the existing
`Dataset` source. Elsewise (default), the existing `Dataset` source will be
replaced. The param is used only if `save` is `True`.

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

Consider new object creation instead. One boy has invoked this method and died.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `TokenDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes will be copied.

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
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

The created `DataLoader` will return batches in the following format: (*\<`list` of
indices of tokens>*, *\<length of the sentence>*). If you use `TokenDataset`
as a part of `FrameDataset`, you can set the param **with_lens** to `False` to
omit the lengths from the batches:
```python
# fds - object of junky.dataset.FrameDataset
fds.add('y', ds, with_lens=False)
```

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 

To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                           collate_kwargs=None, loglevel=0)
```

### CharDataset <a name="char"></a>

Maps tokenized sentences to sequences of lists of indices of their tokens'
characters.

```python
from junky.dataset import CharDataset
ds = CharDataset(sentences, unk_token=None, pad_token=None,
                  extra_tokens=None, allowed_chars=None, exclude_chars=None,
                  int_tensor_dtype=int64, transform=False, skip_unk=False,
                  keep_empty=False)
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

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**transform**: if `True`, invoke `.transform(sentences, save=True)` right
after object creation.

**skip_unk**, **keep_empty**: params for the `.transform()` method.

#### Attributes

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

`ds.transform_dict` (`dict({str: int})`): tokens to indices mapping.

`ds.reconstruct_dict` (`dict({int: str})`): indices to tokens mapping.

`ds.unk` (`int`): index of **&lt;UNK&gt;**.

`ds.pad` (`int`): index of **&lt;PAD&gt;**.

`ds.data` (`list([list([torch.Tensor([int])])])`): the source of `Dataset`.

Generally, you don't need to change any attribute directly.

#### Methods

To re-initialize the `Dataset`, call
```python
ds.fit(sentences, unk_token=None, pad_token=None, extra_tokens=None,
       allowed_chars=None, exclude_chars=None)
```
The method fits the `Dataset` model to **sentences**. All params here have the
same meaning as in the constructor, and this method is invoked from the
constructor. So, you need it only if you want to reuse already existing
`Dataset` object for some new task with a different set of tokens. Really, you
should just create a new object for that.

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
**&lt;PAD&gt;** will be replaced by an empty string, too.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
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
replaced by an empty string, too.

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
sequence of lists of indices. If **skip_unk** is `True`, unknown tokens will be
skipped.

```python
tokens = ds.reconstruct_tokens(ids, skip_unk=False, skip_pad=True)
```
Converts a `list` of indices or a sequence of lists of indices to the
corresponding token or a sequence of tokens. If **skip_unk** is `True`, unknown
indices will be skipped (or replaced to empty strings, if *aslist* is `True`).
If **skip_pad** is `True`, padding indices also will be removed or replaced to
empty strings.

If **aslist** is `True`, we want a list of characters instead of tokens in the
result.

```python
ds.transform(sentences, skip_unk=False, keep_empty=False, save=True,
             append=False)
```
Converts tokenized **sentences** to the sequences of the lists of the indices
corresponding to token's chars and adjusts their format for
`torch.utils.data.Dataset`. If **skip_unk** is `True`, unknown chars will be
skipped. If **keep_empty** is `False` (default), we'll remove tokens and
sequences that have no data after converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **append** is `True`, we'll append the converted sentences to the existing
`Dataset` source. Elsewise (default), the existing `Dataset` source will be
replaced. The param is used only if `save` is `True`.

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

If **aslist** is `True`, we want a list of characters instead of tokens in the
result.

```python
ds.fit_transform(sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, skip_unk=False, keep_empty=False,
                 save=True)
```
Fits the `Dataset` model to **sentences** and then transforms them. In
sequence, calls the `.fit()` and the `.transform()` methods. All params are
the params of those methods. Returns the return of the `.transform()`.

Consider new object creation instead. Anyhow, before calling this method, call
the doctor.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `CharDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes will be copied.

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
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

The created `DataLoader` will return batches of the format (*\<`list` of
`list` of indices of tokens' characters>*, *\<length of the sentence>*,
*\<`list` of lengths of tokens>*). If you use `CharDataset` as a part of
`FrameDataset`, you can set the param **with_lens** to `False` to omit sequence
lengths from the batches:
```python
# fds - object of junky.dataset.FrameDataset
fds.add('x_ch', ds, with_lens=False)
```

If you don't need the lengths of tokens, you can set the param
**with_token_lens** to `False`.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### WordDataset <a name="word"></a>

Maps tokenized sentences to sequences of their words' vectors.

```python
from junky.dataset import WordDataset
ds = WordDataset(emb_model, vec_size,
                 unk_token=None, unk_vec_norm=1e-2,
                 pad_token=None, pad_vec_norm=0.,
                 extra_tokens=None, extra_vec_norm=1e-2,
                 float_tensor_dtype=float32, int_tensor_dtype=int64,
                 sentences=None, skip_unk=False, keep_empty=False)
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

**float_tensor_dtype** (`torch.dtype`, default `torch.float32`): type for
float tensors. Don't change it.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**sentences** (`list([list([str])])`): already tokenized sentences of words.
If not `None`, they will be transformed and saved.

**check_lower**, **skip_unk**, **keep_empty**: params for the `.transform()`
method.

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
vec = ds.word_to_vec(word, check_lower=True, force_lower=False,
                     skip_unk=False)
```
Converts a **word** to its **vector**. If the **word** is not present in the
model, returns a vector of **&lt;UNK&gt;** word or `None` if it's not
defined.

If **check_lower** is `True` (default), try to find vector of `word.lower()`
if vector of **word** is absent.

If **force_lower** is `True`, convert **word** to lower case before looking
its vectors up. Default is `False`.

If **skip_unk** is `True`, unknown words will be skipped.

**NB:** If you created the `Dataset` with **&lt;UNK&gt;**, this token is
present in the model. So, if exactly that token will be met, its vector
will be returned by the method even with `skip_unk=True` param.

```python
vecs = ds.transform_words(words, check_lower=True, force_lower=False,
                          skip_unk=False)
```
Converts a word or a `list` of words to the corresponding vector|`list` of
vectors.

If **check_lower** is `True` (default), try to find a vector of a lower
cased word if vector of a word is absent.

If **force_lower** is `True`, convert words to lower case before looking their
vectors up. Default is `False`.

If **skip_unk** is `True`, unknown words will be skipped.

```python
ds.transform(sentences, check_lower=True, force_lower=False, skip_unk=False,
             keep_empty=False, save=True, append=False)
```
Converts tokenized **sentences** to the sequences of the corresponding vectors
and adjusts their format for `torch.utils.data.Dataset`.

If **check_lower** is `True` (default), try to find a vector of a lower
cased word if vector of a word is absent.

If **force_lower** is `True`, convert sentences to lower case before looking
their vectors up. Default is `False`.

If **skip_unk** is `True`, unknown tokens will be skipped. If **keep_empty**
is `False` (default), we'll remove sentences that have no data after
converting.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

If **append** is `True`, we'll append the converted sentences to the existing
`Dataset` source. Elsewise (default), the existing `Dataset` source will be
replaced. The param is used only if `save` is `True`.

If **save** is `False`, the method returns the result of the transformation.
Elsewise, `None` is returned.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `WordDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes will be copied. The `ds.emb_model` attribute is copied
by link.

```python
emb_model = ds.save(file_path, with_data=True)
```
Saves the `WordDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. All other attributes
will be saved, except `ds.emb_model` which is returned by the method.
You can save it, if needed, using its own method.

```python
ds = WordDataset.load(file_path, emb_model):
```
Load the `WordDataset` object from **file_path**. You should specify
**emb_model** that you used during object creation (or received from the
`.save()` method).

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

The created `DataLoader` will return batches of the format (*<`list` of words'
vectors>*, *\<length of the sentence>*). If you use `WordDataset` as a part of
`FrameDataset`, you can set the param **with_lens** to `False` to omit the
lengths from the batches:
```python
# fds - object of junky.dataset.FrameDataset
fds.add('x', ds, with_lens=False)
```

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### BertDataset <a name="bert"></a>

Maps tokenized sentences to sequences of their contextual word vectors.

```python
from junky.dataset import BertDataset
ds = BertDataset(model, tokenizer, int_tensor_dtype=torch.int64, 
                 sentences=None, **kwargs)
```
`Dataset` processes sentences of any length without cutting. If the length of
subtokens in any sentence is greater than **max_len**, we split the sentence with
overlap, process all parts separately and then combine vectors of all parts in a
single sequence vector. We make splits by word's borders, so, if any word
contains a number of subtokens that is greater than *effective max_len*
(`max_len - 2`, taking into account `[CLS]` and `[SEP]` tokens), it will cause
`RuntimeError`.

In general mode, `BertDataset` replaces **max_len** with the length of the
longest sentence in batch (if that length is less than **max_len**). Also, we
sort all the sequences of subtokens by size before splitting them to batches
and feeding to the **model**. By doing so, we speed up the processing
drasticaly without quality loss, but that behavior can be changed (see
*Attributes* section). **NB:** We don't sort sequences of subtokens if their
number is not greater than **batch_size**.

Params:

**model**: one of the token classification models from the
[*transformers*](https://huggingface.co/transformers/index.html) package. It
should be created with *config* containing `output_hidden_states=True`. NB:
Don't forget to set **model** in the `eval` mode before using it with this
class.

**tokenizer**: a tokenizer from the *transformers* package corresponding to
**model** chosen.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**sentences** (`list([list([str])])`): already tokenized sentences of words.
If not `None`, they will be transformed and saved. NB: All the sentences must
not be empty.

All other args are params for the `.transform()` method. They are used only if
**sentences** is not `None`. You can use any args but `save` that is set to
`True`.

Example:
```python
from corpuscula.corpus_utils import syntagrus
import junky
from junky.dataset import BertDataset
from transformers import BertConfig, BertForTokenClassification, BertTokenizer

bert_model_name = 'bert-base-multilingual-cased'

bert_tokenizer = BertTokenizer.from_pretrained(
    bert_model_name, do_lower_case=False
)
bert_config = BertConfig.from_pretrained(bert_model_name,
                                         output_hidden_states=True)
bert_model = BertForTokenClassification.from_pretrained(
    bert_model_name, config=bert_config
).to('cuda:0')
bert_model.eval()

train = junky.get_conllu_fields(syntagrus.train, fields=[])

ds = BertDataset(bert_model, bert_tokenizer)
ds.transform(train, max_len=0, batch_size=256, hidden_ids=range(9, 13),
             aggregate_hiddens_op='mean', aggregate_subtokens_op='max',
             to=junky.CPU, loglevel=2)

loader = ds.create_loader(shuffle=True)
x, lens = next(iter(loader))
print(x.shape)
print(lens[0])
```

#### Attributes

`ds.model`: a model from the *transformers* package.

`ds.tokenizer`: a tokenizer from the *transformers* package corresponding to
`ds.model`.

`ds.vec_size`: the length of word's vector.

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

Generally, you don't need to change those attributes directly.

Next attributes define the overlap processing. They allow only manual
changing.

`ds.overlap_shift = .5`: Defines the overlap's `shift` from the sentence's
start. We count it in words, so, if sentence has `9` words, the shift will be
`int(.5 * 9)`, i.e., `4`. The minimum value for `shift` is `1`. If you set
`ds.overlap_shift` > `1`, we will treat it as absolute value (but reduce it to
`max_len` if your `ds.overlap_shift` would be greater.

`ds.overlap_border = 2`: The overlap is processed as follows. The left zone of
width equal to `ds.overlap_border` is taken from the earlier part of the
sentence; the right zone - from the later. The zone between borders is
calculated as weighted sum of both parts. The weights are proportional to
the distance to the middle of the zone: the beginning has dominance to the
left from the middle, the ending has dominance to the right. In the very
middle (if it exists), both weights are equal to `.5`. If you set
`ds.overlap_border` high enough (greater than `(max_len - shift) / 2`) or
`None`, it will be set to the middle of the overlap zone. Thus, weighted
algorithm will be dwindled. Also note that weights are applied to tokens (not
subtokens). I.e. all subtokens of any particular token have equal weights when
summing.

`ds.use_batch_max_len = True`: Either we want to use the length of the longest
sentence in the batch instead of the `max_len` param of `.transform()`. We use
it only if that length is less than `max_len`, and as result, with high
**max_len**, we have a substantial speed increase without any quality change
or resulting data.

`ds.sort_dataset = True`: Do we want to sort the dataset before feeding it to
`ds.model`. With high **max_len** it highly increases processing speed, and
affects resulting data only because of different sentences' grouping
(deviation is about `1e-7`).

#### Methods

```python
ds.transform(sentences, max_len=None, batch_size=None, hidden_ids=0,
             aggregate_hiddens_op='mean', aggregate_subtokens_op='absmax',
             with_grad=False, to=CPU, save=True, append=False, loglevel=1)
```
Converts tokenized **sentences** to the sequences of the corresponding
contextual vectors and adjusts their format for `torch.utils.data.Dataset`.

**max_len** is a param for `ds.tokenizer`. We'll transform lines of any
length, but the quality is higher if **max_len** is greater. `None` (default)
or `0` means the maximum for the `ds.model` (usually, `512`).

**batch_size** affects only the execution time. Greater is faster, but big
**batch_size** may be cause of CUDA Memory Error. If `None` or `0`, we'll try
to convert all **sentences** in one batch.

**hidden_ids**: hidden score layers that we need to aggregate. Allowed `int`
or `tuple(int)`. If `None`, we'll aggregate all the layers.

**aggregate_hidden_op**: how to aggregate hidden scores. The ops allowed:
`'absmax'`, `'cat'`, `'max'`, `'mean'`, `'sum'`. For the `'absmax'` method we
take into account absolute values of the compared items.

**aggregate_subtokens_op**: how to aggregate subtokens vectors to form only
one vector for each input word. The ops allowed: `None`, `'absmax'`,
`'expand'`, `'first'`, `'last'`, `'max'`, `'mean'`, `'sum'`. For the
`'absmax'` method we take into account absolute values of the compared items.

If you want to get the result placed on some exact device, specify the device
with **to** param. If **to** is `None`, the data will be placed to the very
device that `ds.model` is used.

If **save** is `True` (default), we'll keep the converted **sentences** as the
`Dataset` source.

*with_grad*: calculate gradients during forward propagation through
`ds.model`. Default is ``False``.

If **append** is `True`, we'll append the converted sentences to the existing
`Dataset` source. Elsewise (default), the existing `Dataset` source will be
replaced. The param is used only if `save` is `True`.

**loglevel** can be set to `0`, `1` or `2`. `0` means no output.

If **save** is `False`, the method returns the result of the transformation.
Elsewise, `None` is returned.

The result depends on **aggregate_subtokens_op** param. If it is `None`,
then for each word we keep a tensor with stacked vectors for all
its subtokens in the result. Otherwise, if any **aggregate_subtokens_op** is
used, each sentence will be converted to exactly one tensor of shape
\[*\<sentence length>*, *\<vector size>*].

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `BertDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. All attributes will be
copied but `ds.model` and `ds.tokenizer` that are copied by link.

```python
model, tokenizer = ds.save(file_path, with_data=True)
```
Saves the `BertDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. All attributes will be
saved but `ds.model` and `ds.tokenizer` that are returned by the method for
you saved them if need by their own methods.

```python
ds = BertDataset.load(file_path, (model, tokenizer)):
```
Load the `BertDataset` object from **file_path**. You should specify **model**
and **tokenizer** that you used during object creation (or received from the
`.save()` method).

**NB:** Really, without `ds.model` and `ds.tokenizer`, `BertDataset` is almost
empty. It's easier to create the object from scratch instead of bother with
all those `save` / `load` / `clone` actions.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

The created `DataLoader` will return batches of the format (*\<`list` of words'
vectors>*, *\<length of the sentence>*\[, *\<`list` of numbers subtokens in
corresponding words>*]). If you use `BertDataset` as part of `FrameDataset`,
you can set the param **with_lens** to `False` to omit sequence lengths from the
batches:
```python
# fds - object of junky.dataset.FrameDataset
fds.add('x', ds, with_lens=False)
```

If you don't need the lengths of tokens, you can set the param
**with_token_lens**  to `False`. Note, that you have it only if you invoked
`.transform(save=True)` with `aggregate_subtokens_op=None` option.

Also, you can specify here a param `append_subtokens=True`. It also allowed
only if `.transform(save=True)` was called with `aggregate_subtokens_op=None`
option, and in this case the loader will return vectors exactly as if you'd
specify `aggregate_subtokens_op='expand'` in `.transform()`. But you can
receive the lengths of tokens.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### BertTokenizedDataset <a name="bertok"></a>

Maps text sentences to sequences prepared for input of
`transformers.BertModel`.

```python
from junky.dataset import BertTokenizedDataset
ds = BertTokenizedDataset(tokenizer, int_tensor_dtype=torch.int64,
                          sentences=None, **kwargs)
```
`Dataset` processes text sentences or already tokenized sentences.

Params:

**tokenizer**: a tokenizer from the *transformers* package.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**sentences** (`list([str])|list([list([str])])`): If not `None`, they will be
transformed and saved. NB: All the sentences must not be empty.

All other args are params for the `.transform()` method. They are used only if
**sentences** is not `None`. You can use any args but `save` that is set to
`True`.

Example:
```python
from corpuscula.corpus_utils import syntagrus
import junky
from junky.dataset import BertTokenizedDataset
from transformers import AutoTokenizer

bert_model_name = 'bert-base-multilingual-cased'

bert_tokenizer = AutoTokenizer.from_pretrained(
    bert_model_name, do_lower_case=False
)

train = junky.get_conllu_fields(syntagrus.train, fields=[])

ds = BertTokenizedDataset(bert_tokenizer, sentences=train)

loader = ds.create_loader(shuffle=True)
x, lens = next(iter(loader))
print(x.shape)
print(lens[0])
```

#### Attributes

`ds.tokenizer`: a tokenizer from the *transformers* package.

`ds.int_tensor_dtype` (`torch.dtype`): type for int tensors.

Generally, you don't need to change those attributes directly.

#### Methods

```python
ds.transform(sentences, add_special_tokens=True, is_pretokenized=False,
             max_len=None, save=True, append=False):
```
Convert text **sentences** to the `transformers.BertModel` input. Already
tokenized sentences are also allowed but will be joined before tokenizing with
space character.

**max_len**, **add_special_tokens** and **is_pretokenized** are params for the
tokenizer. **max_len** `0` means the highest number of subtokens for the model
(usually, `512`). `None` (default) means no limit.

**Important!** `is_pretokenized=True` means, that each sentence is splitten on
tokens ("pre-tokenized"), not tokenized with some transformers' tokenizer. If
sentences are fully tokenized on **sub**tokens, use `is_pretokenized=False`.

If **save** is `True`, we'll keep the converted sentences as the `Dataset`
source.

If **append** is `True`, we'll append the converted sentences to the existing
`Dataset` source. Elsewise (default), the existing `Dataset` source will be
replaced. The param is used only if **save** is `True`.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `BertTokenizedDataset` object. If **with_data** is
`False`, the `Dataset` source in the new object will be empty. All attributes
will be copied but `ds.model` and `ds.tokenizer` that are copied by link.

```python
tokenizer, = ds.save(file_path, with_data=True)
```
Saves the `BertTokenizedDataset` object to **file_path**. If **with_data** is
`False`, the `Dataset` source of the saved object will be empty. All
attributes will be saved but `ds.tokenizer` that is returned by the method for
you saved them if need by their own methods.

```python
ds = BertTokenizedDataset.load(file_path, (tokenizer, )):
```
Load the `BertTokenizedDataset` object from **file_path**. You should specify
**tokenizer** that you used during object creation (or received from the
`.save()` method).

**NB:** Really, without `ds.tokenizer`, `BertTokenizedDataset` is almost
empty. It's easier to create the object from scratch instead of bother with
all those `save` / `load` / `clone` actions.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

The created `DataLoader` will return batches of the format (*<`list` of words'
vectors>*, *\<length of the sentence>*\[, *\<`list` of numbers subtokens in
corresponding words>*]). If you use `BertDataset` as part of `FrameDataset`,
you can set the param **with_lens** to `False` to omit sequence lengths from the
batches:
```python
# fds - object of junky.dataset.FrameDataset
fds.add('x', ds, with_lens=False)
```

If you don't need the lengths of tokens, you can set the param
**with_token_lens**  to `False`. Note, that you have it only if you invoked
`.transform(save=True)` with `aggregate_subtokens_op=None` option.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### FrameDataset <a name="frame"></a>

A frame for use several objects of `junky.dataset.Dataset` conjointly. All
the `Dataset` objects must have the data of equal length.

```python
from junky.dataset import FrameDataset
ds = FrameDataset()
```

#### Attributes

`ds.datasets` (`dict({str: [junky.dataset.BaseDataset, int, kwargs]})`): the
list of `Dataset` objects. Format: {name: \[`Dataset`, *\<number of data
columns>*, *\<collate kwargs>*]}

Generally, you don't need to change any attribute directly.

#### Methods

```python
ds.add(name, dataset, **collate_kwargs)
```
Adds **dataset** with a specified **name**.

Param **collate_kwargs** is a keyword arguments for the **dataset**'s
`._frame_collate()` method. Refer `help(dataset._frame_collate)` to learn
what params the **dataset** has.

**NB:** the **dataset** should not deliver data of the `tuple` type because if
it does, all elements but first in each data row will be hidden.

```python
ds.remove(name)
```
Removes `Dataset` with a specified **name** from `ds.datasets`.

```python
ds_elem = ds.get(name)
```
Returns `tuple(dataset, collate_kwargs)` for the `Dataset` with a specified
**name**.

```python
ds_ = ds.get_dataset(name)
```
Returns the `Dataset` with a specified **name**.

```python
ds_list = ds.list()
```
Returns names of the nested `Dataset` objects in order of their addition.

```python
ds.transform(sentences, names=None, save=True, append=False,
             part_kwargs=None, **kwargs)
```
Invokes `.transform()` methods for all nested `Dataset` objects.

**names** is a `list` of datasets `.transform()` methods of which will be
called.

**save**, **append** and **\*\*kwargs** will be transfered to any nested
`.transform()` methods.

**part_kwargs** is a `dict` of format: *{\<name>: kwargs, ...}*, where one can
specify separate keyword args for `.transform()` method of certain nested
`Dataset` objects.

If **save** is `False`, we'll return the stacked result of objects' returns.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `FrameDataset` object. The **with_data** param is
defined if the nested `Dataset` objects will be cloned with their data sources
or without.

**NB:** Some nested `Dataset` objects may contain objects that are copied by
link (e.g., `emb_model` of `WordDataset`).

```python
xtrn = ds.save(file_path, with_data=True)
```
Saves the `FrameDataset` object to **file_path**.  The **with_data** param is
defined if the nested `Dataset` objects will be saved with their data sources
or without.

Returns a `tuple` of all objects that nested `Dataset` objects don't allow to
save (e.g., `emb_model` in `WordDataset`). If you want to save them, too, you
have to do it by their own methods.

```python
ds = FrameDataset.load(file_path, xtrn):
```
Load the `FrameDataset` object from **file_path**. Also, you need to pass to
the method the **xtrn** object that you received from the `.save()` method.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods of all nested `Dataset` objects .

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### DummyDataset <a name="dummy"></a>

Maps all data elements to the one particular object.

```python
from junky.dataset import DummyDataset
ds = DummyDataset(output_obj=None, data=None)
```
Params:

**output_obj**: the object that will be returned with every invoke. Default is
`None`.

**data**: an array-like object that support the `len(data)` method or just
`int` value that is treated as the length of that object. **NB:** In the
first case we use only size of it.

#### Attributes

`ds.size` (`int`): a length of the internal pseudo-array.

`ds.value`: an object to return as internal pseudo-array element

Generally, you don't need to change any attribute directly.

#### Methods

```python
ds.transform(data, save=True, append=False)
```
Treats the length of **data** as the size of the internal data array. If
**data** is of `int` type, just keeps that value as the size.

If **save** is `True`, we'll keep the size as the size of the `Dataset`
source.

If **append** is `True`, we'll increase the size of the Dataset source by the
size of **data**.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `DummyDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty.

```python
ds.save(file_path, with_data=True)
```
Saves the `DummyDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty.

```python
ds = DummyDataset.load(file_path):
```
Load the `DummyDataset` object from **file_path**.

**NB:** Really, `DummyDataset` without data empty. You can just recreate it
anew when you need it instead of bother with all those `save` / `load` /
`clone` actions.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### LenDataset <a name="len"></a>

Maps array-like data to the lengths of its elements.

```python
from junky.dataset import LenDataset
ds = LenDataset(data=None)
```
Params:

**data**: an array-like object elements of which support the `len()` method.
If not `None`, it will be transformed and saved.

#### Attributes

`ds.data` (`list([torch.Tensor([int])])`): the source of `Dataset`.

#### Methods

```python
ds.transform(data, save=True, append=False)
```
Store lengths of **data** rows as the internal data array.

If **save** is ``True``, we'll keep the converted sentences as the `Dataset`
source.

If **append** is ``True``, we'll append the converted sentences to the
existing `Dataset` source. Elsewise (default), the existing `Dataset` source
will be replaced. The param is used only if **save** is ``True``.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `LenDataset` object. If **with_data** is `False`, the
`Dataset` source in the new object will be empty.

```python
ds.save(file_path, with_data=True)
```
Saves the `LenDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty.

```python
ds = LenDataset.load(file_path):
```
Load the `LenDataset` object from **file_path**.

**NB:** Really, `LenDataset` without data empty. You can just recreate it anew
when you need it instead of bother with all those `save` / `load` / `clone`
actions.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### WordCatDataset <a name="wordcat"></a>

Concatenate outputs of datasets the sources of which are of `torch.Tensor`
type (e.g., `WordDataset`, `BertDataset`, ...). **Importaint**: The sources
of the datasets must be of equal shape by all dimensions except the last one
which will be concatenated.

```python
from junky.dataset import WordCatDataset
ds = WordCatDataset()
```

#### Attributes

`ds.datasets` (`dict({str: junky.dataset.BaseDataset})`): the list of
`Dataset` objects. Format: {name: `Dataset`}

Generally, you don't need to change any attribute directly.

#### Methods

```python
ds.add(name, dataset)
```
Adds **dataset** with a specified **name**.

Param **collate_kwargs** is a keyword arguments for the **dataset**'s
`._frame_collate()` method. Refer `help(dataset._frame_collate)` to learn
what params the **dataset** has.

```python
ds.remove(name)
```
Removes `Dataset` with a specified **name** from `ds.datasets`.

```python
ds_elem = ds.get(name)
```
Returns `Dataset` with a specified **name**.

```python
ds_ = ds.get_dataset(name)
```
An alias for `ds.get(name)`.

```python
ds_list = ds.list()
```
Returns names of the nested `Dataset` objects in order of their addition.

```python
ds.transform(sentences, names=None, save=True, append=False,
             part_kwargs=None, **kwargs)
```
Invokes `.transform()` methods for all nested `Dataset` objects.

**names** is a `list` of datasets `.transform()` methods of which will be
called.

**save**, **append** and **\*\*kwargs** will be transfered to any nested
`.transform()` methods.

**part_kwargs** is a `dict` of format: *{\<name>: kwargs, ...}*, where one can
specify separate keyword args for `.transform()` method of certain nested
`Dataset` objects.

If **save** is `False`, we'll return the stacked result of objects' returns.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `WordCatDataset` object. The **with_data** param is
defined if the nested `Dataset` objects will be cloned with their data sources
or without.

**NB:** Some nested `Dataset` objects may contain objects that are copied by
link (e.g., `emb_model` of `WordDataset`).

```python
xtrn = ds.save(file_path, with_data=True)
```
Saves the `WordCatDataset` object to **file_path**.  The **with_data** param is
defined if the nested `Dataset` objects will be saved with their data sources
or without.

Returns a `tuple` of all objects that nested `Dataset` objects don't allow to
save (e.g., `emb_model` in `WordDataset`). If you want to save them, too, you
have to do it by their own methods.

```python
ds = WordCatDataset.load(file_path, xtrn):
```
Load the `FrameDataset` object from **file_path**. Also, you need to pass to
the method the **xtrn** object that you received from the `.save()` method.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods of all nested `Dataset` objects .

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

On the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you shouldn't use it because it's not thread-safe. 
To read batches sequentially use:
```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### LabelDataset <a name="label"></a>

Dataset for label-level input.

```python
from junky.dataset import LabelDataset
ds = LabelDataset(labels, unk_label=None, extra_labels=None, 
                 tensor_dtype=int64, transform=False, skip_unk=False,
                 keep_empty=False)
```
Params:

**labels** (`list([str])|list[list([str])])`): list of available labels. Each
row may contain several label values.

**unk_label** (`str`): value for unknown labels that are not present in the
internal dict. `None` by default.

**extra_labels** (`list([str])`): list of any additional labels.

**tensor_dtype** (`torch.dtype`): dtype of tensors, `int64` by default.

**transform** (`bool`): if `True`, transform and save `labels`.

**skip_unk** (`bool`): param for `.transform()` method. If `True`, unknown
labels are skipped. `False` by default.

**keep_empty** (`bool`): param for `.transform()` method. If `True`, empty
rows are removed. `False` by default.

#### Attributes

`ds.tensor_dtype` (`torch.dtype`): tensor type.

`ds.transform_dict` (`dict({str: int})`): tokens to indices mapping.

`ds.reconstruct_dict` (`dict({int: str})`): indices to tokens mapping.

`ds.unk` (`numpy.ndarray([float])`): `unk_label` index.

`ds.data` (`list([torch.Tensor([numpy.ndarray([float])])])`): the source of
`Dataset`.

Generally, you don't need to change any attribute directly.

#### Methods

To re-initialize the Dataset, call
```python
ds.fit(labels, unk_label=None, extra_labels=None)
```
The method fits the `Dataset` model to **labels**. All params here have the
same meaning as in the constructor, and this method is invoked from the
constructor. You need it only to reuse already existing `Dataset` object for
a new task with a different set of tokens. In fact, it's recommended to rather
create a new object for that.

```python
idx = ds.label_to_idx(label, skip_unk=False)
```

Converts a **label** value to its index. If the value is not present in the
internal dict, return index of the `unk_label` or `None` if it's not defined
**skip_unk** is `True`.

**NB:** If `unk_label` was initially in the internal dictionary, its index
will be returned by the method even if `skip_unk=True`.

```python
token = ds.idx_to_label(idx, skip_unk=False)
```
Returns an index of the corresponding label value. If the index is not present
in the internal dict, return `unk_label` or an empty string if it's not
defined or **skip_unk** is `True`. 

**NB:** If `unk_label` was initially in the internal dictionary, it will be
returned by the method even if `skip_unk=True`.

```python
ids = ds.transform(labels, skip_unk=False, keep_empty=False, 
                   save=True, append=False)
```
Convert **labels** of `str` type to the sequences of corresponding indices
and adjust their format for `Dataset`. If **skip_unk** is `True`, unknown
labels will be skipped. If **keep_empty** is `False`, empty rows with no data
after converting will be removed.

Corresponding indices are represented as is (`int` numbers) if **labels** are
of `list([str])` type. Elsewise, if **labels** are of `list` of `list([str])`
type (each row may contain several label values), the indices are represented
as multi-hot vectors. The type of indices' representation is `tensor_dtype`
specified in constructor.

If **save** is `True`, converted labels are kept as Dataset source.

If **append** is `True`, converted labels are appended to the existing Dataset
source. Elsewise (default), the existing Dataset source will be replaced. The
param is used only if **save** is `True`.


```python
tokens = ds.reconstruct(ids, skip_unk=False, keep_empty=False)
```
Convert *sequences* of indices in `Dataset` format to the rows of the
corresponding label values.

If **skip_unk** is `True`, unknown indices
will be skipped.

If **keep_empty** is `False`, the labels that are empty after converting are
removed.

```python
ds.fit_transform(labels, unk_label=None, extra_labels=None,
                 skip_unk=False, keep_empty=False, save=True)
```
Fits the `Dataset` model to **labels** and then transforms them. This method
combines `.fit()` and `.transform()` methods. All the attributes are the same
as in these methods. Returns the return of the `ds.transform()`.

```python
o = ds.clone(with_data=True)
```
Makes a deep copy of the `LabelDataset` object. If **with_data** is `False`,
the `Dataset` source in the new object will be empty. The model and all other
attributes will be copied.

```python
ds.save(file_path, with_data=True)
```
Saves the `LabelDataset` object to **file_path**. If **with_data** is `False`,
the `Dataset` source of the saved object will be empty. The model and all
other attributes will be saved.

```python
ds = LabelDataset.load(file_path):
```
Load the `LabelDataset` object from **file_path**.

```python
ds.to(*args, **kwargs):
```
Invokes `.to(*args, **kwargs)` methods for all the elements of the `Dataset`
source that have `torch.Tensor` or `torch.nn.Model` type. All the params will
be transferred as is.

```python
loader = ds.create_loader(batch_size=32, shuffle=False, num_workers=0,
                          **kwargs)
```
Creates `torch.utils.data.DataLoader` for this object. All params are the
params of `DataLoader`. Only **dataset** and **collate_fn** can't be changed.

**NB:** If you set **num_workers** != `0`, don't move the **ds** source to
*CUDA*. The `torch` multiprocessing implementation can't handle it. Better,
create several instances of `DataLoader` for **ds** (each with `workers=0`)
and use them in parallel.

During the inference stage you don't have to use the combination of
`.transform(sentences, save=True)` and `.create_loader(shuffle=False)`
methods. Actually, you even shouldn't use it because it's not thread-safe. 
To read batches sequentially use:

```python
loader = ds.transform_collate(sentences, batch_size=32, transform_kwargs=None,
                              collate_kwargs=None, loglevel=0)
```

### Examples <a name="examples"></a>

Let us suppose that we have sets of data for training (**train** and
**train_labels**), validation (**dev**, **dev_labels**) and testing
(**test**, **test_labels**). Each set is a `list` of already tokenized
sentences and a `list` of sequences of the corresponding labels. We want a
`DataLoader` that will return batches for each set.

Format of the element of the each batch is: (*\<`list` of words' vectors>*, 
*\<length of the sentence>*, *\<`list` of `list` of indices of words'
characters>*, *\<`list` of lengths of words>*, *\<`list` of indices of words'
labels>*.

Firstly, we create 3 datasets: for words' vectors, for chars' indices and for
labels' indices.

```python
x_train = WordDataset(emb_model=emb_model,
                      vec_size=emb_model.vector_size,
                      unk_token='<UNK>', pad_token='<PAD>',
                      sentences=train)
x_dev = x_train.clone(with_data=False)
x_dev.transform(dev)
x_test = x_train.clone(with_data=False)
x_test.transform(test)
```

```python
x_ch_train = CharDataset(train + dev + test,
                         unk_token='<UNK>', pad_token='<PAD>')
x_ch_train.transform(train)
x_ch_dev = x_ch_train.clone(with_data=False)
x_ch_dev.transform(dev)
x_ch_test = x_ch_train.clone(with_data=False)
x_ch_test.transform(test)
```

```python
y_train = TokenDataset(train_labels, pad_token='<PAD>',
                       transform=True, keep_empty=False)
y_dev = y_train.clone(with_data=False)
y_dev.transform(dev_labels)
y_test = y_train.clone(with_data=False)
y_test.transform(test_labels)
```

Then, we create combining `Dataset` objects that conjoin output of those
created `Dataset` objects as we demanded.

```python
ds_train = FrameDataset()
ds_train.add('x', x_train)
# we don't need one more *lens* field from char dataset:
ds_train.add('x_ch', x_ch_train, with_lens=False)
# again, we don't need one more *lens* field from token dataset:
ds_train.add('y', y_train, with_lens=False)
```

```python
ds_dev = FrameDataset()
ds_dev.add('x', x_dev)
ds_dev.add('x_ch', x_ch_dev, with_lens=False)
ds_dev.add('y', y_dev, with_lens=False)
```

```python
ds_test = FrameDataset()
ds_test.add('x', x_test)
ds_test.add('x_ch', x_ch_test, with_lens=False)
ds_test.add('y', y_test, with_lens=False)
```

Create loaders:
```python
loader_train = ds_train.create_loader(shuffle=True)
loader_dev = ds_dev.create_loader()
loader_test = ds_test.create_loader()
```
