<h2 align="center">junky lib: PyTorch utilities</h2>

## Datasets

The lib contains a few descendants of `torch.utils.data.Dataset` that can be
used together with *PyTorch* models.

### TokenDataset

Maps `str` tokens to their indices.

```python
from junky.dataset import TokenDataset
ds = TokenDataset(sentences, unk_token=None, pad_token=None,
                  extra_tokens=None, transform=False, skip_unk=False,
                  keep_empty=False, int_tensor_dtype=int64, batch_first=False)
```
Params:

**sentences** (`list([list([str])])`): already tokenized sentences.

**unk_token** (`str`): add a token for tokens that are not present in the
dict (further, **&lt;UNK&gt;**).

**pad_token** (`str`): add a token for padding (further, **&lt;PAD&gt;**).

**extra_tokens** (`list([str])`): add tokens for any other purposes.

**transform**: if ``True``, invoke `.transform(sentences, save=True)` right
after object creation.

**skip_unk**, **keep_empty**: params for the `transform()` method.

**int_tensor_dtype** (`torch.dtype`, default `torch.int64`): type for int
tensors. Don't change it.

**batch_first**: if ``True``, then the input and output tensors are provided
as `(batch, seq, feature)`. Otherwise (default), `(seq, batch, feature)`.

To initialize the *dataset*, call
```python
ds.fit(sentences, unk_token=None, pad_token=None, extra_tokens=None)
```
All params here has the same meaning as in constructor, and this method is
invoked from constructor. So, you need it only if you want to reuse already
existing *dataset* object for some new task with a different set of tokens.
Really, you can just create a new object for that.

```python
idx = token_to_idx(token, skip_unk=False)
```
Convert a **token** to its index. If the **token** is not present in the
internal dict, return index of **UNK** token or None if it's not
        defined.

```python
    def idx_to_token(self, idx, skip_unk=False, skip_pad=True):
```

```python
    def transform_tokens(self, tokens, skip_unk=False):
```

```python
    def reconstruct_tokens(self, ids, skip_unk=False, skip_pad=True):
```

```python
    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
```

```python
    def reconstruct(self, sequences, skip_unk=False, skip_pad=True,
                    keep_empty=False):
```python
    def fit_transform(self, sentences, unk_token=None, pad_token=None,
                      extra_tokens=None, skip_unk=False, keep_empty=False,
                      save=True):
```

```python
    def clone(self, with_data=True):
```

```python
    def save(self, file_path, with_data=True):
```

```python
    def load(file_path):
```

```python
    def to(self, *args, **kwargs):
```

```python
    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
```
