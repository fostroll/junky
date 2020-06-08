# -*- coding: utf-8 -*-
# junky lib: layers.CharEmbeddingRNN
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a CharEmbeddingRNN layer implementation for PyTorch models.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
                               pad_sequence


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
            # слова (получим форму [N, S, E]). т.е., теперь у нас в x
            # будут эмбеддинги слов
            x = x.view(*x_shape, -1).sum(-2)

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
