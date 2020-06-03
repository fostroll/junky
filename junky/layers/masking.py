# -*- coding: utf-8 -*-
# junky lib: layers.Masking
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a Masking layer implementation for PyTorch models.
"""
from collections.abc import Iterable
from junky import CPU
import torch
import torch.nn as nn


class Masking(nn.Module):
    """
    Replaces certain elemens of the incoming data to the `mask` given.

    Args:
        input_size: The number of expected features in the input `x`.
        mask: Replace to what.
        indices_to_highlight: What positions in the `feature` dimension of the
            masked positions of the incoming data must not be replaced to the
            `mask`.
        highlighting_mask: Replace data in that positions to what. If
            ``None``, the data will keep as is.
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)` (<==> `(N, *, H)`). Default:
            ``False``.

    Shape:
        - Input: :math:`(*, N, H)` where :math:`*` means any number of
          additional dimensions and :math:`H = \text{input_size}`.
        - Output: :math:`(*, N, H)` where all are the same shape as the input
          and :math:`H = \text{input_size}`.

    .. note:: Masking layer was made for using right before Softmax. In that
        case and with `mask`=``-inf`` (default), the Softmax output will have
        zeroes in all positions corresponding to `indices_to_mask`.

    .. note:: Usually, you'll mask positions of all non-pad tags in padded
        endings of the input data. Thus, after Softmax, you'll always have the
        padding tag predicted for that endings. As the result, you'll have
        loss = 0, that prevents your model for learning on padding.

    Examples::

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
    """
    __constants__ = ['batch_first', 'highlighting_mask',
                     'indices_to_highlight', 'input_size', 'mask']

    def __init__(self, input_size, mask=float('-inf'),
                 indices_to_highlight=-1, highlighting_mask=1,
                 batch_first=False):
        super().__init__()

        if not isinstance(indices_to_highlight, Iterable):
            indices_to_highlight = [indices_to_highlight]

        self.input_size = input_size
        self.mask = mask
        self.indices_to_highlight = indices_to_highlight
        self.highlighting_mask = highlighting_mask
        self.batch_first = batch_first

        output_mask = torch.tensor([mask] * input_size)
        if indices_to_highlight is not None:
            if highlighting_mask is None:
                output_mask0 = torch.tensor([0] * input_size,
                                            dtype=output_mask.dtype)
                for idx in indices_to_highlight:
                    output_mask0[idx] = 1
                    output_mask[idx] = 0
                output_mask = torch.stack((output_mask0, output_mask))
            else:
                for idx in indices_to_highlight:
                    output_mask[idx] = highlighting_mask
        self.register_buffer('output_mask', output_mask)

    def forward(self, x, lens):
        """
        :param lens: array of lengths of **x** by the `seq` dimension.
        """
        output_mask = self.output_mask
        output_mask0, output_mask = \
            output_mask if len(output_mask.shape) == 2 else \
            (None, output_mask)
        device = output_mask.get_device() if output_mask.is_cuda else CPU
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens, device=device)

        seq_len = x.shape[self.batch_first]
        padding_mask = \
            torch.arange(seq_len, device=device) \
                 .expand(lens.shape[0], seq_len) >= lens.unsqueeze(1)
        if not self.batch_first:
            padding_mask = padding_mask.transpose(0, 1)
        x[padding_mask] = output_mask if output_mask0 is None else \
                          x[padding_mask] * output_mask0 + output_mask

        return x

    def extra_repr(self):
        return ('{}, mask={}, indices_to_highlight={}, highlighting_mask={}, '
                'batch_first={}').format(
                    self.input_size, self.mask, self.indices_to_highlight,
                    self.highlighting_mask, self.batch_first
                )
