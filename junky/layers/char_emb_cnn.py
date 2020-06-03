# -*- coding: utf-8 -*-
# junky lib: layers.CharEmbeddingCNN
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a CharEmbeddingCNN layer implementation for PyTorch models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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
