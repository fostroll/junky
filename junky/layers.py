# -*- coding: utf-8 -*-
# junky lib: layers
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of PyTorch layers.
"""
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
                               pad_sequence


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
        device = output_mask.get_device() if output_mask.is_cuda else \
                 torch.device('cpu')
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


class CharEmbeddingRNN(nn.Module):
    """
    Produces character embeddings using bidirectional LSTM.

    Args:
        alphabet_size: length of character vocabulary.
        emb_layer: optional pre-trained embeddings, 
            initialized as torch.nn.Embedding.from_pretrained() or elsewise.
        emb_dim: character embedding dimensionality.
        pad_idx: indices of padding element in character vocabulary.
        out_type: 'final_concat'|'final_mean'|'all_mean'.
            `out_type` defines what to get as a result from the LSTM.
            'final_concat' concatenate final hidden states of forward and
                           backward lstm;
            'final_mean' take mean of final hidden states of forward and
                         backward lstm;
            'all_mean' take mean of all timeframes.

    Shape:
        - Input:
            x: [batch[seq[word[ch_idx + pad] + word[pad]]]]; torch tensor of
                shape :math:`(N, S(padded), C(padded))`, where `N` is
                batch_size, `S` is seq_len and `C` is max char_len in a word
                in current batch.
            lens: [seq[word_char_count]]; torch tensor of shape
                :math:`(N, S(padded), C(padded))`, word lengths for each
                sequence in batch. Used in masking & packing/unpacking
                sequences for LSTM.
        - Output: :math:`(N, S, H)` where `N`, `S` are the same shape as the
            input and :math:` H = \text{lstm hidden size}`.
    
    .. note:: In LSTM layer, we ignore padding by applying mask to the tensor
        and eliminating all words of len=0. After LSTM layer, initial
        dimensions are restored using the same mask.
    """
    __constants__ = ['alphabet_size', 'emb_dim', 'out_type', 'pad_idx']

    def __init__(self, alphabet_size, emb_layer=None, emb_dim=300, pad_idx=0,
                 out_type='final_concat'):
        """
        :param out_type: 'final_concat'|'final_mean'|'all_mean'
        """
        super().__init__()

        self.alphabet_size = alphabet_size
        self.emb_dim = None if emb_layer else emb_dim
        self.pad_idx = pad_idx
        self.out_type = out_type

        self._emb_l = emb_layer if emb_layer else \
                      nn.Embedding(alphabet_size, emb_dim,
                                   padding_idx=pad_idx)
        self._rnn_l = nn.LSTM(input_size=self._emb_l.embedding_dim,
                              hidden_size=self._emb_l.embedding_dim // (
                                  2 if out_type in ['final_concat',
                                                    'all_mean'] else
                                  1 if out_type in ['final_mean'] else
                                  0 # error
                              ),
                              num_layers=1, batch_first=True,
                              dropout=0, bidirectional=True)

    def forward(self, x, lens):
        """
        x: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device

        # сохраняем форму батча символов:
        # [#предложение в батче:#слово в предложении:#символ в слове]
        #                                                       <==> [N, S, C]
        x_shape = x.shape
        # все слова во всех батчах сцепляем вместе. из-за паддинга все
        # предложения одной длины, так что расцепить обратно не будет
        # проблемой. теперь у нас один большой батч из всех слов:
        #                                             [N, S, C] --> [N * S, C]
        # важно: слова из-за паддинга тоже все одной длины. при этом многие
        #        пустые, т.е., состоят только из паддинга
        x = x.flatten(end_dim=1)

        # прогоняем через слой символьного эмбеддинга (обучаемый):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        # сохраняем эту форму тоже
        x_e_shape = x.shape

        # создаём массив длин всех слов. для этого в список массивов длин слов
        # в предложениях добавляем нули в качестве длин слов, состоящих только
        # из паддинга, после чего сцепляем их вместе так же, как мы до этого
        # сцепили символы: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # дальше предполагалось передать x и lens0 в pack_padded_sequence,
        # в надежде что она нейтрализует влияние слов нулевой длины. однако
        # выяснилось, что эта функция нулевые длины не принимает. поэтому:
        # 1. делаем маску длин слов: True, если длина не равна нулю (слово не
        # из паддинга)
        mask = lens0 != 0
        # 2. убираем нулевые слова из массива символьных эмбеддингов
        x_m = x[mask]
        # 3. убираем нулевые длины, чтобы массив длин соответствовал новому
        # массиву символьных эмбеддингов
        lens0_m = lens0[mask]

        # теперь у нас остались только ненулевые слова и есть массив их длин.
        # запаковываем
        x_m = pack_padded_sequence(x_m, lens0_m,
                                   batch_first=True, enforce_sorted=False)
        # lstm
        x_m, (x_hid, x_cstate) = self._rnn_l(x_m)

        ### в качестве результата можно брать либо усреднение/суммирование/
        ### что-то ещё hidden state на всех таймфреймах (тогда надо будет
        ### вначале распаковать x_m, который содержит конкатенацию hidden
        ### state прямого и обратного lstm на каждом таймфрейме); либо можно
        ### взять финальные значения hidden state для прямого и обратного
        ### lstm и, например, использовать их конкатенацию.
        ### важно: если мы используем конкатенацию (даже неявно, когда
        ### не разделяем x_cm_m), то размер hidden-слоя д.б. в 2 раза меньше,
        ### чем реальная размерность, которую мы хотим получить на входе
        ### в результате.
        if self.out_type == 'all_mean':
            ## если результат - среднее значение hidden state на всех
            ## таймфреймах:
            # 1. распаковываем hidden state. 
            x_m, _ = pad_packed_sequence(x_m, batch_first=True)
            # 2. теперь x_m имеет ту же форму, что и перед запаковыванием.
            # помещаем его в то же место x, откуда забрали (используем
            # ту же маску)
            x[mask] = x_m
            # 3. теперь нужно результат усреднить. нам не нужны значения для
            # каждого символа, мы хотим получить эмбеддинги слов. мы решили,
            # что эмбеддинг слова - это среднее значение эмбеддингов всех
            # входящих в него символов. т.е., нужно сложить эмбеддинги
            # символов и разделить на длину слова. в слове есть паддинг, но
            # после pad_packed_sequence его вектора должны быть нулевыми.
            # однако у нас есть слова полностью из паддинга, которые мы
            # удаляли, а теперь вернули. их вектора после слоя эмбеддинга
            # будут нулевыми, поскольку при создании слоя мы указали индекс
            # паддинга. иначе можно было бы их занулить явно: x[~mask] = .0
            # 4. чтобы посчитать среднее значение эмбеддинга слова, нужно
            # сложить эмбеддинги его символов и разделить на длину слова:
            # 4a. установим длину нулевых слов в 1, чтобы не делить на 0.
            # мы обнулили эти эмбеддинги, так что проблемы не будет
            lens1 = pad_sequence(lens, padding_value=1).flatten()
            # 4b. теперь делим все символьные эмбеддинги на длину слов,
            # которые из них составлены (нормализация):
            x /= lens1[:, None, None]
            # 4c. теперь возвращаем обратно уровень предложений (в
            # результате будет форма [N, S, C, E]) и складываем уже
            # нормализованные эмбеддинги символов в рамках каждого
            # слова (получим форму [N, S, E]). т.е., теперь у нас в x_
            # будут эмбеддинги слов
            x = x_ch.view(*x_shape, -1).sum(-2)

        elif self.out_type in ['final_concat', 'final_mean']:
            ## если результат - конкатенация либо средние значения последних
            ## hidden state прямого и обратного lstm:
            if self.out_type == 'final_concat':
                # 1. конкатенация
                x_m = x_hid.transpose(-1, -2) \
                           .reshape(1, x_hid.shape[-1] * 2, -1) \
                           .transpose(-1, -2)
            elif self.out_type == 'final_mean':
                # 1. среднее
                x_m = x_hid.transpose(-1, -2) \
                           .mean(dim=0) \
                           .transpose(-1, -2)
            # 2. в этой точке нам нужна x_e_shape: forma x после
            # прохождения слоя эмбеддинга. причём измерение символов нам
            # надо убрать, т.е., перейти от [N * S, C, E] к [N * S, E].
            # при этом на месте пустых слов нам нужны нули. создадим
            # новый тензор:
            x = torch.zeros(x_e_shape[0], x_e_shape[2],
                            dtype=torch.float, device=device)
            # 3. сейчас x_m имеет ту же форму, что и перед запаковыванием,
            # но измерения символов уже нет. помещаем его в то же место x,
            # откуда брали (используем ту же маску)
            x[mask] = x_m
            # сейчас у нас x в форме [N * S, E]. переводим его в форму
            # эмбеддингов слов [N, S, E]:
            x = x.view(*x_shape[:-1], -1)

        return x

    def extra_repr(self):
        return '{}, {}, pad_idx={}, out_type={}'.format(
            self.alphabet_size, self.emb_dim, self.pad_idx, self.out_type
        ) if self.emb_dim else \
        '{}, external embedding layer, out_type={}'.format(
            self.alphabet_size, self.out_type
        )


class CharEmbeddingCNN(nn.Module):
    """
    Produces character embeddings using multiple-filter CNN. Max-over-time
    pooling and ReLU are applied to concatenated convolution layers.

    Args:
        alphabet_size: length of character vocabulary.
        emb_layer: optional pre-trained embeddings, 
            initialized as torch.nn.Embedding.from_pretrained() or elsewise.
        emb_dim: character embedding dimensionality.
        emb_dropout: dropout for embedding layer. Default: 0.0 (no dropout).
        pad_idx: indices of padding element in character vocabulary.
        kernels: convoluiton filter sizes for CNN layers. 
        cnn_kernel_multiplier: defines how many filters are created for each 
            kernel size. Default: 1.
        
    Shape:
        - Input:
            x: [batch[seq[word[ch_idx + pad] + word[pad]]]]; torch tensor of
                shape :math:`(N, S(padded), C(padded))`, where `N` is
                batch_size, `S` is seq_len with padding and `C` is char_len
                with padding in current batch. 
            lens: [seq[word_char_count]]; torch tensor of shape
                :math:`(N, S, C)`, word lengths for each sequence in batch.
                Used for eliminating padding in CNN layers.
        - Output: :math:`(N, S, E)` where `N`, `S` are the same shape as the
            input and :math:` E = \text{emb_dim}`.
    """
    __constants__ = ['alphabet_size', 'emb_dim', 'kernels', 'cnn_kernel_multiplier', 'pad_idx']

    def __init__(self, alphabet_size, emb_layer=None, emb_dim=300, emb_dropout=0.0,
                 pad_idx=0, kernels=[3, 4, 5], cnn_kernel_multiplier=1):
        super().__init__()

        self.kernels = list(kernels)
        self.alphabet_size = alphabet_size
        self.emb_dim = None if emb_layer else emb_dim
        self.pad_idx = pad_idx

        self._emb_l = emb_layer if emb_layer else \
                      nn.Embedding(alphabet_size, emb_dim,
                                   padding_idx=pad_idx)
                                   
        self._emb_dropout = nn.Dropout(p=emb_dropout)

        self._conv_ls = nn.ModuleList(
            [nn.Conv1d(in_channels=self._emb_l.embedding_dim,
                       out_channels=self._emb_l.embedding_dim,
                       padding=0, kernel_size=kernel)
                 for kernel in kernels] * cnn_kernel_multiplier
        )

    def forward(self, x, lens):
        """
        x: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device
        max_len = x.shape[-1]

        # сохраняем форму батча символов:
        # [#предложение в батче:#слово в предложении:#символ в слове]
        #                                                       <==> [N, S, C]
        x_shape = x.shape
        # все слова во всех батчах сцепляем вместе. из-за паддинга все
        # предложения одной длины, так что расцепить обратно не будет
        # проблемой. теперь у нас один большой батч из всех слов:
        #                                             [N, S, C] --> [N * S, C]
        # важно: слова из-за паддинга тоже все одной длины. при этом многие
        #        пустые, т.е., состоят только из паддинга
        x = x.flatten(end_dim=1)

        # прогоняем через слой символьного эмбеддинга (обучаемый):
        # [N * S, C] --> [N * S, C, E]
        x = self._emb_l(x)
        x = self._emb_dropout(x)
        # сохраняем эту форму тоже
        x_e_shape = x.shape

        # создаём массив длин всех слов. для этого в список массивов длин слов
        # в предложениях добавляем нули в качестве длин слов, состоящих только
        # из паддинга, после чего сцепляем их вместе так же, как мы до этого
        # сцепили символы: [N, S] --> [N * S]
        lens0 = pad_sequence(lens, batch_first=True).flatten()

        # теперь маскируем слова нулевой длины:
        # 1. делаем маску длин слов: True, если длина не равна нулю (слово не
        # из паддинга)
        mask = lens0 != 0
        # 2. убираем нулевые слова из массива символьных эмбеддингов
        x_m = x[mask]
        # 3. убираем нулевые длины, чтобы массив длин соответствовал новому
        # массиву символьных эмбеддингов
        lens0_m = lens0[mask]

        # Добавим здесь три сверточных слоя с пулингом
        # NB! CNN принимает на вход тензор размерностью
        #     [batch_size, hidden_size, seq_len] 
        # CNN tensor input shape:
        #     [nonzero(N * S), E, C]
        # tensor after convolution shape:
        #     [nonzero(N * S), E, C - cnn_kernel_size + 1]
        # tensor after pooling:
        #     [nonzero(N * S), E, (C - cnn_kernel_size + 1)
        #                       - (pool_kernel_size - 1)]
        # example:
        ## N=32, E=300, C=45
        ## CNN (kernel_size=5) [32, 300, 41]
        ## pooling (kernel_size=8) [32, 300, 34]
        x_m = x_m.transpose(1, 2)

        x_ms = []
        for conv_l in self._conv_ls:
            if conv_l.kernel_size[0] <= max_len:
                #x_ms.append(F.relu(F.adaptive_max_pool1d(conv_l(x_m),
                #                                         output_size=2)))
                x_ms.append(conv_l(x_m))

        x_m = torch.cat(x_ms, dim=-1)
        # сейчас x_m имеет форму [N * S, E, pool_concat]. нам нужно привести
        # её к виду [N * S, E]. нам не нужно транспонировать его обратно, мы
        # просто берём среднее с учётом текущего расположения измерений
        #x_m = torch.mean(x_m, -1)
        x_m = F.relu(torch.max(x_m, -1)[0])

        # в этой точке нам нужна x_e_shape: forma x после
        # прохождения слоя эмбеддинга. причём измерение символов нам
        # надо убрать, т.е., перейти от [N * S, C, E] к [N * S, E].
        # при этом на месте пустых слов нам нужны нули. создадим
        # новый тензор:
        x = torch.zeros(x_e_shape[0], x_e_shape[2],
                        dtype=torch.float, device=device)
        # 3. сейчас x_m имеет ту же форму, что и перед запаковыванием,
        # но измерения символов уже нет. помещаем его в то же место x,
        # откуда брали (используем ту же маску)
        x[mask] = x_m
        # сейчас у нас x в форме [N * S, E]. переводим его в форму
        # эмбеддингов слов [N, S, E]:
        x = x.view(*x_shape[:-1], -1)

        return x

    def extra_repr(self):
        return '{}, {}, pad_idx={}, kernels={}'.format(
            self.alphabet_size, self.emb_dim, self.pad_idx, self.kernels
        ) if self.emb_dim else \
        '{}, external embedding layer, kernels={}'.format(
            self.alphabet_size, self.kernels
        )


class Highway(nn.Module):
    """ 
    Highway layer for Highway Networks as described in
    https://arxiv.org/abs/1505.00387 and https://arxiv.org/abs/1507.06228
    articles.

    Applies H(x)*T(x) + x*(1 - T(x)) transformation, where:
    .. H(x) - affine trainsform followed by a non-linear activation. The layer
           that we make Highway around;
    .. T(x) - transform gate: affine transform followed by a sigmoid
           activation;
    .. * - element-wise multiplication.

    Args:
        dim: size of each input and output sample.
        H_layer: H(x) layer. If ``None`` (default), affine transform is used.
        H_activation: non-linear activation after H(x). If ``None`` (default),
            then, if H_layer is ``None``, too, we apply F.relu; otherwise,
            activation function is not used.
    """
    __constants__ = ['H_layer', 'H_activation', 'dim']

    def __init__(self, dim, H_layer=None, H_activation=None):
        super().__init__()

        self._H = H_layer if H_layer else nn.Linear(dim, dim)
        self._H_activation = H_activation if H_activation else \
                             F.relu

        self._T = nn.Linear(dim, dim)
        self._T_activation = torch.sigmoid
        nn.init.constant_(self._T.bias, -1)

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, seq_len, emb_size]
        :return: tensor with shape [batch_size, seq_len, emb_size]
        """
        gate = self._T_activation(self._T(x))
        hx = self._H(x)
        if self._H_activation:
            hx = self._H_activation(hx)

        return hx * gate + x * (1 - gate)

    def extra_repr(self):
        return '{}, H_layer={}, H_activation={}'.format(
            self._T.dim, self._H, self._H_activation
        )


class HighwayNetwork(nn.Module):
    """ 
    Highway Network is described in
    https://arxiv.org/abs/1505.00387 and https://arxiv.org/abs/1507.06228 and
    it's formalation is: H(x)*T(x) + x*(1 - T(x)), where:
    .. H(x) - affine trainsformation followed by a non-linear activation;
    .. T(x) - transformation gate: affine transformation followed by a sigmoid
           activation;
    .. * - element-wise multiplication.

    There are some variations of it, so we implement more universal 
    architectute: U(x)*H(x)*T(x) + x*C(x), where:
    .. U(x) - user defined layer that we make Highway around; By default,
           U(x) = I (identity matrix);
    .. C(x) - carry gate: generally, affine transformation followed by a sigmoid
           activation. By default, C(x) = 1 - T(x).

    Args:
        in_features: number of features in input.
        out_features: number of features in output. If ``None`` (default),
            **out_features** = **in_features**.
        U_layer: layer that implements U(x). Default is ``None``. If U_layer
            is callable, it will be used to create the layer; elsewise, we'll
            use it as is (if **num_layers** > 1, we'll copy it). Note that
            number of input features of U_layer must be equal to
            **out_features** if **num_layers** > 1.
        U_init_: callable to inplace init weights of **U_layer**.
        U_dropout: if non-zero, introduces a Dropout layer on the outputs of
            U(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        H_features: number of input features of H(x). If ``None`` (default),
            H_features = in_features. If ``0``, don't use H(x).
        H_activation: non-linear activation after H(x). If ``None``, then no
            activation function is used. Default is ``F.relu``.
        H_dropout: if non-zero, introduces a Dropout layer on the outputs of
            H(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        gate_type: a type of the transform and carry gates:
            'generic' (default): C(x) = 1 - T(x);
            'independent': use both independent C(x) and T(x);
            'T_only': don't use carry gate: C(x) = I;
            'C_only': don't use carry gate: T(x) = I;
            'none': C(x) = T(x) = I.
        global_highway_input: if ``True``, we treat the input of all the
            network as the highway input of every layer. Thus, we use T(x)
            and C(x) only once. If **global_highway_input** is ``False``
            (default), every layer receives the output of the previous layer
            as the highway input. So, T(x) and C(x) use different weights
            matrices in each layer.
        num_layers: number of highway layers.
    """
    __constants__ = ['H_activation', 'H_dropout', 'H_features', 'U_dropout',
                     'U_init', 'U_layer', 'gate_type', 'global_highway_input',
                     'last_dropout', 'out_dim', 'num_layers']

    def __init__(self, in_features, out_features=None,
                 U_layer=None, U_init_=None, U_dropout=0,
                 H_features=None, H_activation=F.relu, H_dropout=0,
                 gate_type='generic', global_highway_input=False,
                 num_layers=1):
        super().__init__()

        if out_features is None:
            out_features = in_features
        if H_features is None:
            H_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.U_layer = U_layer
        self.U_init_ = U_init_
        self.U_dropout = U_dropout
        self.H_features = H_features
        self.H_activation = H_activation
        self.H_dropout = H_dropout
        self.gate_type = gate_type
        self.global_highway_input = global_highway_input
        self.num_layers = num_layers

        if U_layer:
            self._U = U_layer() if callable(U_layer) else U_layer
            if U_init_:
                U_init_(U_layer)
        else:
            self._U = None
        self._H = nn.Linear(H_features, out_features) if H_features else None
        if self.gate_type not in ['C_only', 'none']:
            self._T = nn.Linear(in_features, out_features)
            nn.init.constant_(self._T.bias, -1)
        if self.gate_type not in ['generic', 'T_only', 'none']:
            self._C = nn.Linear(in_features, out_features)
            nn.init.constant_(self._C.bias, 1)

        self._U_do = \
            nn.Dropout(p=U_dropout) if self._U and U_dropout else None
        self._H_do = \
            nn.Dropout(p=H_dropout) if self._H and H_dropout else None

        self._H_activation = H_activation
        self._T_activation = torch.sigmoid
        self._C_activation = torch.sigmoid

        if self.num_layers > 1:
            self._Us = nn.ModuleList() if U_layer else None
            self._Hs = nn.ModuleList() if H_features else None
            if not self.global_highway_input:
                if self.gate_type not in ['C_only', 'none']:
                    self._Ts = nn.ModuleList()
                if self.gate_type not in ['generic', 'T_only', 'none']:
                    self._Cs = nn.ModuleList()

            for i in range(self.num_layers - 1):
                if self._Us is not None:
                    U = U_layer() if callable(U_layer) else deepcopy(U_layer)
                    if U_init_:
                       U_init_(U)
                    self._Us.append(U)
                if self._Hs is not None:
                    self._Hs.append(nn.Linear(out_features, out_features))
                if not self.global_highway_input:
                    if self.gate_type not in ['C_only', 'none']:
                        T = nn.Linear(in_features, out_features)
                        nn.init.constant_(T.bias, -1)
                        self._Ts.append(T)
                    if self.gate_type not in ['generic', 'T_only', 'none']:
                        C = nn.Linear(in_features, out_features)
                        nn.init.constant_(C.bias, -1)
                        self._Cs.append(C)

    def forward(self, x, x_hw, *U_args, **U_kwargs):
        """
        :param x: tensor with shape [batch_size, seq_len, emb_size]
        :param x_hw: tensor with shape [batch_size, in_features, emb_size]
            if ``None``, x is used
        :return: tensor with shape [batch_size, seq_len, emb_size]
        """
        if x_hw is None:
            x_hw = x

        if self._U:
            x = self._U(x, *U_args, **U_kwargs)
            if self._U_do:
                x = self._U_do(x)
        if self._H:
            x = self._H(x)
        if self._H_activation:
            x = self._H_activation(x)
            if self._H_do:
                x = self._H_do(x)

        if self.gate_type == 'generic':
            x_t = self._T_activation(self._T(x_hw))
            x = x_t * x
            x_hw = (1 - x_t) * x_hw
        elif self.gate_type not in ['C_only', 'none']:
            x = self._T_activation(self._T(x_hw)) * x
        elif self.gate_type not in ['T_only', 'none']:
            x_hw = self._C_activation(self._C(x_hw)) * x_hw

        x += x_hw

        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                if self._Us:
                    x = self._Us[i](x, *U_args, **U_kwargs)
                    if self._U_do:
                        x = self._U_do(x)
                if self._Hs:
                    x = self._Hs[i](x)
                if self._H_activation:
                    x = self._H_activation(x)
                    if self._H_do:
                        x = self._H_do(x)

                if not self.global_highway_input:
                    if self.gate_type == 'generic':
                        x_t = self._T_activation(self._Ts[i](x_hw))
                        x = x_t * x
                        x_hw = (1 - x_t) * x_hw
                    elif self.gate_type not in ['C_only', 'none']:
                        x = self._T_activation(self._Ts[i](x_hw)) * x
                    elif self.gate_type not in ['T_only', 'none']:
                        x_hw = self._C_activation(self._Cs[i](x_hw)) * x_hw

                x += x_hw
                if not self.global_highway_input:
                    x_hw = x

        return x

    def extra_repr(self):
        return (
            '{}, {}, U_layer={}, U_init_={}, U_dropout={}, '
            'H_features={}, H_activation={}, H_dropout={}, '
            "gate_type='{}', global_highway_input={}, num_layers={}"
        ).format(self.in_features, self.out_features,
                 None if not self.U_layer else
                 '<callable>' if callable(self.U_layer) else
                 '<layer>' if isinstance(self.U_layer, nn.Module) else
                 '<ERROR>',
                 None if not self.U_init_ else
                 '<callable>' if callable(self.U_init_) else
                 '<ERROR>', self.U_dropout,
                 self.H_features,
                 None if not self.H_activation else
                 '<callable>' if callable(self.H_activation) else
                 '<ERROR>', self.H_dropout,
                 self.gate_type, self.global_highway_input, self.num_layers)


class HBiLSTM_layer(nn.Module):
    """Highway LSTM implementation, modified from
    from https://github.com/bamtercelboo/pytorch_Highway_Networks/blob/master/models/model_HBiLSTM.py.
    [Original Article](https://arxiv.org/abs/1709.06436).

    Args:
        in_features:        number of features in input.
        out_features:       number of features in output.
        lstm_hidden_dim:    hidden dim for LSTM layer.
        lstm_num_layers:    number of LSTM layers.
        lstm_dropout:       dropout between 2+ LSTM layers.
        init_weight:        whether to init bilstm weights as xavier_uniform_
        init_weight_value:  bilstm weight initialization `gain` will be defined as 
            `np.sqrt(init_weight_value)`
    """

    def __init__(self, in_features, out_features,
                 lstm_hidden_dim, lstm_num_layers,
                 lstm_dropout=0.0, init_weight=True, 
                 init_weight_value=2.0):
        super().__init__()

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = lstm_dropout
        self.bilstm = nn.LSTM(in_features, self.hidden_dim, num_layers=self.num_layers, bias=True, 
                              bidirectional=True, dropout=self.dropout)
        
        if init_weight:
            nn.init.xavier_uniform_(self.bilstm.all_weights[0][0], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[0][1], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[1][0], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[1][1], gain=np.sqrt(init_weight_value))

        if self.bilstm.bias is True:
            a = np.sqrt(2/(1 + 600)) * np.sqrt(3)
            nn.init.uniform_(self.bilstm.all_weights[0][2], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[0][3], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[1][2], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[1][3], -a, a)

        self.convert_layer = nn.Linear(in_features=self.hidden_dim * 2,
                                       out_features=self.in_features, bias=True)
        nn.init.constant_(self.convert_layer.bias, -1)

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_dim*2, bias=True)
        nn.init.constant_(self.fc1.bias, -1)

        self.gate_layer = nn.Linear(in_features=self.in_features, out_features=self.hidden_dim*2, bias=True)
        nn.init.constant_(self.gate_layer.bias, -1)

    def forward(self, x, hidden, lens):
        # handle the source input x
        source_x = x
        
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        x, hidden = self.bilstm(x, hidden)
        x, _ = pad_packed_sequence(x)
        
        normal_fc = torch.transpose(x, 0, 1)

        x = x.permute(1, 2, 0)

        # normal layer in the formula is H
        source_x = torch.transpose(source_x, 0, 1)

        # the first way to convert 3D tensor to the Linear

        source_x = source_x.contiguous()
        information_source = source_x.view(source_x.size(0) * source_x.size(1), source_x.size(2))
        information_source = self.gate_layer(information_source)
        information_source = information_source.view(source_x.size(0), source_x.size(1), information_source.size(1))

        # transformation gate layer in the formula is T
        transformation_layer = torch.sigmoid(information_source)
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)

        # the information_source compare to the source_x is for the same size of x,y,H,T
        allow_carry = torch.mul(information_source, carry_layer)
        # allow_carry = torch.mul(source_x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)

        information_flow = information_flow.contiguous()
        information_convert = information_flow.view(information_flow.size(0) * information_flow.size(1),
                                                    information_flow.size(2))
        information_convert = self.convert_layer(information_convert)
        information_convert = information_convert.view(information_flow.size(0), information_flow.size(1),
                                                       information_convert.size(1))

        information_convert = torch.transpose(information_convert, 0, 1)
        return information_convert, hidden


# HighWay recurrent model
class HighwayBiLSTM(nn.Module):
    """Highway LSTM model implementation, modified from
    from https://github.com/bamtercelboo/pytorch_Highway_Networks/blob/master/models/model_HBiLSTM.py.
    [Original Article](https://arxiv.org/abs/1709.06436).

    Args:
        hw_num_layers:      number of highway biLSTM layers.
        in_features:        number of features in input.
        out_features:       number of features in output.
        lstm_hidden_dim:    hidden dim for LSTM layer.
        lstm_num_layers:    number of LSTM layers.
        lstm_dropout:       dropout between 2+ LSTM layers.
        init_weight:        whether to init bilstm weights as xavier_uniform_
        init_weight_value:  bilstm weight initialization `gain` will be defined as 
            `np.sqrt(init_weight_value)`
        batch_first:        True if input.size(0) == batch_size.
        
    """

    def __init__(self, hw_num_layers, lstm_hidden_dim, lstm_num_layers, 
                 in_features, out_features, lstm_dropout,
                 init_weight=True, init_weight_value=2.0, batch_first=True):
        super().__init__()

        self.hw_num_layers = hw_num_layers
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.batch_first = batch_first

        # multiple HighWay layers List
        self.highway = nn.ModuleList(
                        [HBiLSTM_layer(in_features, out_features, 
                                 lstm_hidden_dim, lstm_num_layers, 
                                 lstm_dropout=lstm_dropout, 
                                 init_weight=True, init_weight_value=2.0) 
                         for _ in range(hw_num_layers)]
        )
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        if self.output_layer.bias is True:
            a = np.sqrt(2/(1 + in_features)) * np.sqrt(3)
            nn.init.uniform_(self.output_layer, -a, a)

    def init_hidden(self, num_layers, batch_size, device):
        # the first is the hidden h
        # the second is the cell c
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).to(device),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).to(device))

    def forward(self, x, lens):
        device = next(self.parameters()).device

        self.hidden = self.init_hidden(self.num_layers, x.size(0), device)

        if self.batch_first:
            x = x.transpose(0, 1)     # to (seq_len, batch_size, emb_dim)

        for current_layer in self.highway:
            x, self.hidden = current_layer(x, self.hidden, lens)

        x = torch.transpose(x, 0, 1)
        x = torch.tanh(x)
        output_layer = self.output_layer(x)

        return output_layer
