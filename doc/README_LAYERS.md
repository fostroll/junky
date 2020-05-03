<h2 align="center">junky lib: PyTorch utilities</h2>

## Layers

The lib contains some layers to use in *PyTorch* models.

### Masking

```python
import junky
layer = junky.Masking(input_size, mask=float('-inf'),
                      indices_to_highlight=-1, highlighting_mask=1,
                      batch_first=False)
output = layer(x, lens)
```
Replaces certain elemens of the incoming data **x** to the **mask** given.

Args:

**input_size**: The number of expected features in the input **x**.

**mask**: Replace to what. Default is `-inf`

**indices_to_highlight**: What positions in the `feature` dimension of the
masked positions of the incoming data must not be replaced to the `mask`.
Default is `-1`.

**highlighting_mask**: Replace data in that positions to what. If `None`, the
data will keep as is. Default is `1`

**batch_first**: If `True`, then the input and output tensors are provided
as `(batch, seq, feature)` (<==> `(N, *, H)`). Default: `False`.

**x**: Input data.

**lens**: Array of lengths of **x** by the `seq` dimension.

Shape:<br/>
- Input: :math:`(*, N, H)` where :math:`*` means any number of additional
dimensions and :math:`H = \text{input_size}`.<br/>
- Output: :math:`(*, N, H)` where all are the same shape as the input and
:math:`H = \text{input_size}`.

**NB:** Masking layer was made for using right before Softmax. In that case
and with `mask`=``-inf`` (default), the Softmax output will have zeroes in all
positions corresponding to `indices_to_mask`.

**NB:** Usually, you'll mask positions of all non-pad tags in padded endings
of the input data. Thus, after Softmax, you'll always have the padding tag
predicted for that endings. As the result, you'll have loss = `0`, that
prevents your model for learning on padding.

Examples:

```python
    >>> m = Masking(4, batch_first=True)
    >>> input = torch.randn(2, 3, 4)
    >>> output = m(input, [1, 3])
    >>> print(output)
    tensor([[[ 1.1912, -0.6164,  0.5299, -0.6446],
             [   -inf,    -inf,    -inf,  1.0000],
             [   -inf,    -inf,    -inf,  1.0000]],

            [[-0.3011, -0.7185,  0.6882, -0.1656],
             [-0.3316, -0.3521, -0.9717,  0.5551],
             [ 0.7721,  0.2061,  0.8932, -1.5827]]])
```

```python
    >>> m = Masking(4, batch_first=True, mask=4.,
                    indices_to_highlight=(1, -1), highlighting_mask=None)
    >>> input = torch.randn(2, 3, 4)
    >>> output = m(input, [1, 3])
    >>> print(output)
    tensor([[[-0.4479, -0.8719, -1.0129, -1.5431],
             [ 4.0000,  0.6978,  4.0000,  0.1203],
             [ 4.0000,  0.1990,  4.0000, -0.4277]],

            [[ 0.2840,  1.1241, -0.5342,  0.2857],
             [ 0.3409,  0.7630,  0.4099,  0.1182],
             [ 1.3610, -0.1528, -1.7044, -0.4466]]])
```

### CharEmbeddingRNN

```python
layer = junky.CharEmbeddingRNN(alphabet_size, emb_layer=None, emb_dim=300,
                               pad_idx=0, out_type='final_concat')
```
