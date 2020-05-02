<h2 align="center">junky lib: PyTorch utilities</h2>

## Layers

The lib contains some layers to use in *PyTorch* models.

### Masking

```python
import junky
layer = junky.Masking(input_size, indices_to_mask=-1, mask=float('-inf'),
                      batch_first=False)
```
Replaces certain elemens of the incoming data to the `mask` given.

Args:

**input_size**: The number of expected features in the input `x`.

**indices_to_mask**: What positions in the `feature` dimension of the incoming
data must be replaced to the `mask`.

**mask**: Replace to what.

**batch_first**: If ``True``, then the input and output tensors are provided
as `(batch, seq, feature)`. Default: `False`.

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
>>> output = m(input, torch.tensor([1, 3]))
>>> print(output)
tensor([[[ 1.1912, -0.6164,  0.5299, -0.6446],
         [   -inf,    -inf,    -inf,  1.0000],
         [   -inf,    -inf,    -inf,  1.0000]],

        [[-0.3011, -0.7185,  0.6882, -0.1656],
         [-0.3316, -0.3521, -0.9717,  0.5551],
         [ 0.7721,  0.2061,  0.8932, -1.5827]]])
```
