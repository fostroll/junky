<h2 align="center">junky lib: PyTorch utilities</h2>

## Layers

The lib contains several layers to use in *PyTorch* models.

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
data will be left as is. Default is `1`

**batch_first**: If `True`, then the input and output tensors are provided
as `(batch, seq, feature)` (<==> `(N, *, H)`). Default: `False`.

Shape:

- Input:<br/>
**x**: :math:`(*, N, H)` where :math:`*` means any number of additional
dimensions and :math:`H = \text{input_size}`.<br/>
**lens**: Array of lengths of **x** by the `seq` dimension. We mask data in
all `seq` positions that greater than **lens**.

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
Produces character embeddings using *Bidirectional LSTM*.

Args:

**alphabet_size**: Length of character vocabulary.

**emb_layer**: Optional pre-trained embeddings initialized as
`torch.nn.Embedding.from_pretrained()` or elsewise.

**emb_dim**: Character embedding dimensionality.

**emb_dropout**: Dropout for embedding layer. Default: 0.0 (no dropout).

**pad_idx**: Indices of padding element in character vocabulary.

**out_type** - defines what to get as a result after the *BiLSTM*. Possible
values:<br/>
`'final_concat'` - concatenate final hidden states of forward and backward
*LSTM*;<br/>
`'final_mean'` - take mean of final hidden states of forward and backward
*LSTM*;<br/>
`'all_mean'` - take mean of all timeframes.

Shape:<br/>

- Input:<br/>
**x**: [batch[seq[word[ch_idx + pad] + word[pad]]]]; `torch.Tensor` of shape
:math:`(N, S(padded), C(padded))`, where `N` is batch_size, `S` is seq_len and
`C` is max char_len in a word in current batch.<br/>
**lens**: [seq[word_char_count]]; `torch.Tensor` of shape
:math:`(N, S(padded), C(padded))`, word lengths for each sequence in batch.
Used in masking & packing/unpacking sequences for *LSTM*.

- Output: :math:`(N, S, H)` where `N`, `S` are the same shape as the input and
:math:`H = \text{lstm hidden size}`.

**NB:** In *LSTM* layer, we ignore padding by applying mask to the tensor and
eliminating all words of len = `0`. After *LSTM* layer, initial dimensions are
restored using the same mask.

### CharEmbeddingCNN

```python
layer = junky.CharEmbeddingCNN(alphabet_size, emb_layer=None, emb_dim=300, emb_dropout=0.0,
							   pad_idx=0, kernels=[3, 4, 5], cnn_kernel_multiplier=1)
```
Produces character embeddings using multiple-filter *CNN*. *Max-over-time
pooling* and *ReLU* are applied to concatenated convolution layers.

Args:

**alphabet_size**: Length of character vocabulary.

**emb_layer**: Optional pre-trained embeddings, initialized as
`torch.nn.Embedding.from_pretrained()` or elsewise.

**emb_dim**: Character embedding dimensionality.

**pad_idx**: Indices of padding element in character vocabulary.

**kernels**: Convoluiton filter sizes for *CNN* layers. 

**cnn_kernel_multiplier**: defines how many filters are created for each 
kernel size. Default: 1.
    
Shape:

- Input:<br/>
**x**: [batch[seq[word[ch_idx + pad] + word[pad]]]]; `torch.Tensor` of shape
:math:`(N, S(padded), C(padded))`, where `N` is batch_size, `S` is seq_len
with padding and `C` is char_len with padding in current batch.<br/>
**lens**: [seq[word_char_count]]; `torch.Tensor` of shape :math:`(N, S, C)`,
word lengths for each sequence in batch. Used for eliminating padding in *CNN*
layers.

- Output: :math:`(N, S, E)` where `N`, `S` are the same shape as the input and
:math:`E = \text{emb_dim}`.

### Highway

```python
layer = junky.Highway(dim, H_layer=None, H_activation=None)
```
*Highway* layer for *Highway Networks* as described in
[Srivastava et al.](https://arxiv.org/abs/1505.00387) and
[Srivastava et al.](https://arxiv.org/abs/1507.06228) articles.

Applies **H(x)\*T(x) + x\*(1 - T(x))** transformation, where:

**H(x)** - affine trainsformation followed by a non-linear activation. The layer
that we make Highway around;<br/>
**T(x)** - transform gate: affine transformation followed by a sigmoid
activation;<br/>
**\*** - element-wise multiplication.

Args:

**dim**: size of each input and output sample.

**H_layer**: **H(x)** layer. If `None` (default), affine transformation is used.

**H_activation**: non-linear activation after **H(x)**. If `None` (default),
then, if **H_layer** is `None`, too, we apply `F.relu`; otherwise, activation
function is not used.


### HighwayNetwork

```python
layer = junky.HighwayNetwork(
    in_features, out_features=None, U_layer=None, U_init_=None,
    H_features=None, H_activation=F.relu, gate_type='generic',
    global_highway_input=False, num_layers=1, dropout=0,
    last_dropout=0
)
layer(x, x_hw, *U_args, **U_kwargs)
```
*Highway Network* is described in
[Srivastava et al.](https://arxiv.org/abs/1505.00387) and
[Srivastava et al.](https://arxiv.org/abs/1507.06228) and it's formalation is:
**H(x)\*T(x) + x\*(1 - T(x))**, where:

**H(x)** - affine trainsformation followed by a non-linear activation;<br/>
**T(x)** - transform gate: affine transformation followed by a sigmoid
activation;<br/>
**\*** - element-wise multiplication.

There are some variations of it, so we implement more universal architectute:
**U(x)\*H(x)\*T(x) + x\*C(x)**, where:

**U(x)** - user defined layer that we make *Highway* around; By default,
**U(x) = I** (identity matrix);<br/>
**C(x)** - carry gate: generally, affine transformation followed by a sigmoid
activation. By default, **C(x) = 1 - T(x)**.

Args:

**in_features**: number of features in input.

**out_features**: number of features in output. If `None` (default),
**out_features = in_features**.

**U_layer**: layer that implements **U(x)**. Default is `None`. If
**U_layer** is callable, it will be used to create the layer; elsewise, we'll
use it as is (if **num_layers** > `1`, we'll copy it). Note that number of
input features of **U_layer** must be equal to **out_features** if
**num_layers** > `1`.

**U_init_**: callable to inplace init weights of **U_layer**.

**U_dropout**: if non-zero, introduces a Dropout layer on the outputs of U(x)
on each layer, with dropout probability equal to **U_dropout**. Default: 0.

**H_features**: number of input features of H(x). If `None` (default),
**H_features = in_features**. If `0`, don't use **H(x)**.

**H_activation**: non-linear activation after **H(x)**. If `None`, then no
activation function is used. Default is ``F.relu``.

**H_dropout**: if non-zero, introduces a Dropout layer on the outputs of H(x)
on each layer, with dropout probability equal to **H_dropout**. Default: 0.

**gate_type**: a type of the transform and carry gates:<br/>
`'generic'` (default): **C(x) = 1 - T(x)**;<br/>
`'independent'`: use both independent **C(x)** and **T(x)**;<br/>
`'T_only'`: don't use carry gate: **C(x) = I**;<br/>
`'C_only'`: don't use carry gate: **T(x) = I**;<br/>
`'none'`: **C(x) = T(x) = I**.

**global_highway_input**: if `True`, we treat the input of all the network as
the highway input of every layer. Thus, we use **T(x)** and **C(x)** only
once. If **global_highway_input** is `False` (default), every layer receives
the output of the previous layer as the highway input. So, **T(x)** and
**C(x)** use different weights matrices in each layer.

**num_layers**: number of highway layers.

**dropout**: if non-zero, introduces a *Dropout* layer on the outputs of each
layer except the last layer, with dropout probability equal to **dropout**.
Default: `0`.

**last_dropout**: if non-zero, introduces a Dropout layer on the output of
last layer with dropout probability equal to **last_dropout**. Default: `0`.

The `.forward()` method receives params as follows:

**x** and **x_hw**: inputs of the network. The first layer of the network
executes formula: **x = U(x)\*H(x)\*T(x_hw) + x_hw\*C(x_hw)**. Next, if
**global_highway_input** is `False`, **x_hw = x**. If `True`, then
**x_hw = x_hw\*C(x_hw)** and it's already won't change on the other layers.
If **x_hw** is `None`, we adopt **x_hw = x**.

**\*U_args** and **\*\*U_kwargs** are params for **U_layer** if it needs ones.
