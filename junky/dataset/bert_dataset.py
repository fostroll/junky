# -*- coding: utf-8 -*-
# junky lib: BertDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for word-level input.
"""
from junky import CPU, absmax_torch, pad_sequences_with_tensor
from junky.dataset.base_dataset import BaseDataset
from torch import Tensor, float32, int64, tensor


class BertDataset(BaseDataset):
    """
    `torch.utils.data.Dataset` for word-level input with contextual
    embeddings.

    Args:
        model: one of the token classification models from the `transformers`
            package. It should be created with config containing
            output_hidden_states=True. NB: Don't forget to set model in the
            eval mode before use it with this class.
        tokenizer: the tokenizer from `transformers` package corresponding to
            `model` chosen.
        int_tensor_dtype: dtype for int tensors: torch.dtype.
        sentences: sequences of words: list([list([str])]). If not ``None``,
            they will be transformed and saved. NB: All the sentences must
            not be empty.
        all other args are params for the `.transpose()` method. They are used
            only if sentences is not ``None``.
    """
    def __init__(self, model, tokenizer, int_tensor_dtype=int64,
                 sentences=None, max_len=None, batch_size=32, hidden_ids=0,
                 aggregate_hiddens_op='mean', aggregate_subtokens_op='max'):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.int_tensor_dtype = int_tensor_dtype
        if sentences:
            self.transform(sentences, max_len=max_len,
                           batch_size=batch_size, hidden_ids=hidden_ids,
                           aggregate_hiddens_op=aggregate_hiddens_op,
                           aggregate_subtokens_op=aggregate_hiddens_op,
                           save=True)

    def _pull_xtrn(self):
        xtrn = [self.model, self.tokenizer]
        self.model, self.tokenizer = None, None
        return xtrn

    def _push_xtrn(self, xtrn):
        self.model, self.tokenizer = xtrn

    @staticmethod
    def _aggregate_hidden_states(hidden_states, layer_ids=None,
                                 aggregate_op='mean'):
        if isinstance(layer_ids, int):
            hiddens = hidden_states[layer_ids]

        else:
            if layer_ids is None:
                hiddens = hidden_states
            else:
                hiddens = [hidden_states[x] for x in layer_ids]

            if aggregate_op == 'cat':
                hiddens = torch.cat(hiddens, dim=-1)
            elif aggregate_op == 'max':
                hiddens = absmax_torch(hiddens, dim=0)
            elif aggregate_op == 'mean':
                hiddens = torch.mean(hiddens if isinstance(hiddens, Tensor) else
                                     torch.stack(hiddens, dim=0),
                                     dim=0)
            elif aggregate_op == 'sum':
                hiddens = torch.sum(hiddens if isinstance(hiddens, Tensor) else
                                     torch.stack(hiddens, dim=0),
                                     dim=0)
            else:
                RuntimeError(
                  'ERROR: unknown aggregate_op '
                  "(choose one of ['cat', 'max', 'mean', 'sum'])"
                )
        return hiddens

    def transform(self, sentences, max_len=None, batch_size=None,
                  hidden_ids=0, aggregate_hiddens_op='mean',
                  aggregate_subtokens_op='max', save=True, append=False):
        """Convert *sentences* of words to the sequences of the corresponding
        contextual vectors and adjust their format for Dataset.

        *max_len* is a param for tokenizer. We'll transform lines of any
            length, but the quality is higher if *max_len* is greater.

        *batch_size* affects only on the execution time. Greater is faster,
            but big *batch_size* may be cause of CUDA Memory Error. If ``None``
            (default), we'll try to convert all *sentences* with one batch.

        *hidden_ids*: hidden score layers that we need to aggregate. Allowed
            int or tuple of ints. If ``None``, we'll aggregate all the layers.

        *aggregate_hidden_op*: how to aggregate hidden scores. The ops
            allowed: 'cat', 'max', 'mean', 'sum'. For 'max' method we take
            into account the absolute values of the compared items (absmax
            method).

        *aggregate_subtokens_op*: how to aggregate subtokens vectors to form
            only one vector for each input token. The ops allowed: ``None``,
            'max', 'mean', 'sum'. For 'max' method we take into account the
            absolute values of the compared items (absmax method).

        If save is ``True``, we'll keep the converted sentences as the
        `Dataset` source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save*=True.

        The result is depend on *aggregate_subtokens_op* param. If it is
        ``None``, the result keeps for each token the tensor with stacked
        vectors for all its subtokens. Otherwise, if any
        *aggregate_subtokens_op* is used, the each sentence will be converted
        to only one tensor of shape [<sentence length>, <vector size>]."""

        # overlap zone
        OVERLAP_SHIFT_COEF = .5
        OVERLAP_BORDER = 2

        if not max_len:
            max_len = self.tokenizer.max_len
        assert max_len >= 16, 'ERROR: max len must be >= 16'
        assert max_len <= self.tokenizer.max_len, \
               'ERROR: max len must be <= {}'.format(self.tokenizer.max_len)
        valid_ops = ['cat', 'max', 'mean', 'sum']
        assert aggregate_hiddens_op in valid_ops, \
               'ERROR: unknown aggregate_hidden_op (choose one of {})' \
                   .format(valid_ops)
        valid_ops = [None, 'max', 'mean', 'sum']
        assert aggregate_subtokens_op in valid_ops, \
               'ERROR: unknown aggregate_subtokens_op (choose one of {})' \
                   .format(valid_ops)
        if self.data and append:
            assert (
                isinstance(self.data[0], list)
            and aggregate_subtokens_op is None
            ) or (
                isinstance(self.data[0], Tensor)
            and aggregate_subtokens_op is not None
            ), "ERROR: can't append data created with inconsistent " \
               'aggregate_subtokens_op'
        device = next(self.model.parameters()).device
        max_len -= 2  # for [CLS] and [SEP]
        if batch_size is None:
            batch_size = len(sentences)

        shift = int(max_len * OVERLAP_SHIFT_COEF)

        # tokenize each token separately by the BERT tokenizer
        tokenized_sentences = [[self.tokenizer.tokenize(x) for x in x]
                                   for x in sentences]
        # number of subtokens in tokens of tokenized_sentences
        num_subtokens = [[len(x) for x in x] for x in tokenized_sentences]
        # for each subtoken we keep index of its token
        sub_to_kens = [[
            x for x in [[i] * x for i, x in enumerate(x)]
              for x in x
        ] for x in num_subtokens]
        print_list(sub_to_kens)
        # for each token we keep its start in flatten tokenized_sentences
        token_starts = []
        for sent in sub_to_kens:
            starts, prev_idx = [], -1
            for i, idx in enumerate(sent):
                if idx != prev_idx:
                    starts.append(i)
                    prev_idx = idx
            token_starts.append(starts)

        # flattening tokenized_sentences
        tokenized_sentences = [[x for x in x for x in x]
                                   for x in tokenized_sentences]
        # lengths of flattened tokenized_sentences
        sent_lens = [len(x) for x in tokenized_sentences]

        def process_long_sentences (sents, sent_lens, sub_to_kens,
                                    token_starts):
            overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
            overlap_token_starts, overmap = [], [], [], [], []
            for i, (sent, sent_len, token_ids, sub_ids) \
                    in enumerate(zip(sents, sent_lens, sub_to_kens,
                                     token_starts)):
                if sent_len > max_len:
                    # находим индекс токена для сабтокена с шифтом
                    pos = token_ids[shift]
                    # вычитаем токен нулевого сабтокена, получаем индекс
                    # токена относительно текущего начала
                    pos_ = pos - token_ids[0]
                    if not pos_:
                        pos, pos_ = token_ids[sub_ids[1]], 1
                    # находим индекс сабтокена: из индекса сабтокена
                    # найденного токена вычитаем индекс текущего нулевого
                    # сабтокена
                    start = sub_ids[pos_] - sub_ids[0]
                    if start > max_len:
                        raise RuntimeError(
                            ('ERROR: too long token in sentence {}, '
                             'token {} (longer than max_len)')
                                 .format(i, pos)
                        )
                    overlap_sents.append(sent[start:])
                    overlap_sent_lens.append(sent_len - start)
                    overlap_sub_to_kens.append(token_ids[start:])
                    overlap_token_starts.append(sub_ids[pos_:])
                    overmap.append((i, pos_))
                    end = token_ids[max_len] - token_ids[0]
                    end = sub_ids[end] - sub_ids[0]
                    sent[end:] = []
            return overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
                   overlap_token_starts, overmap

        overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
        overlap_token_starts, overmap = process_long_sentences(
            tokenized_sentences, sent_lens,
            sub_to_kens, token_starts
        )
        num_sents = len(tokenized_sentences)
        overmap = {i + num_sents: (orig_i, pos)
                       for i, (orig_i, pos) in enumerate(overmap)}

        while overlap_sents:
            tokenized_sentences += overlap_sents
            sent_lens += overlap_sent_lens
            sub_to_kens += overlap_sub_to_kens
            token_starts += overlap_token_starts
            prev_num_sents = num_sents
            num_sents = len(tokenized_sentences)
            overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
            overlap_token_starts, overmap_ = process_long_sentences(
                overlap_sents, overlap_sent_lens,
                overlap_sub_to_kens, overlap_token_starts
            )
            for i, (orig_i, pos) in enumerate(overmap_):
                prev_map = overmap[orig_i + prev_num_sents]
                overmap[i + num_sents] = (prev_map[0], prev_map[1] + pos)

#        a = []
#        print(overmap)
#        for i, sent in enumerate(tokenized_sentences):
#            if i in overmap:
#                j, pos = overmap[i]
#                 print(j, pos)
#                 print(token_starts[j])
#                start = token_starts[j][pos]
#                 print(start)
#                a[j][start:] = sent
#            else:
#                a.append(sent)
#        print('\nRESTORE:')
#        print_list(a)

        '''
        def process_long_sentences (sents, sent_lens, num_subtokens):
            overlap_sents, overlap_sent_lens, \
            overlap_num_subtokens, overmap = [], [], [], []
            for i, (sent, sent_len, sub_lens) \
                    in enumerate(zip(sents, sent_lens, num_subtokens)):
                if sent_len > max_len:
                    num, start, pos = 0, 0, 0
                    for j, num_ in enumerate(sub_lens):
                        new_num = num + num_
                        if not start and new_num > shift:
                            pos, start = (j, num) if j else (1, new_num)
                            if start > max_len:
                                raise RuntimeError(
                                    ('ERROR: too long token in sentence {}, '
                                     'token {} (longer than max_len)')
                                        .format(i, j)
                                )
                        if new_num > max_len:
                            overlap_sents.append(sent[start:])
                            overlap_sent_lens.append(sent_len - start)
                            overlap_num_subtokens.append(sub_lens[pos:])
                            overmap.append((i, pos))
                            sent[num:] = []
                            break
                        num = new_num
            return overlap_sents, overlap_sent_lens, overlap_num_subtokens, \
                   overmap

        overlap_sents, overlap_sent_lens, overlap_num_subtokens, overmap = \
            process_long_sentences(tokenized_sentences, sent_lens,
                                   num_subtokens)
        num_sents = len(tokenized_sentences)
        overmap = {i + num_sents: (orig_i, pos)
                       for i, (orig_i, pos) in enumerate(overmap)}

        while overlap_sents:
            tokenized_sentences += overlap_sents
            sent_lens += overlap_sent_lens
            num_subtokens += overlap_num_subtokens
            prev_num_sents = num_sents
            num_sents = len(tokenized_sentences)
            overlap_sents, overlap_sent_lens, overlap_num_subtokens,
            overmap_ = process_long_sentences(
                overlap_sents, overlap_sent_lens, overlap_num_subtokens
            )
            for i, (orig_i, pos) in enumerate(overmap_):
                prev_map = overmap[orig_i + prev_num_sents]
                overmap[i + num_sents] = (prev_map[0], prev_map[1] + pos)

#         a = []
#         for i, sent in enumerate(tokenized_sentences):
#             if i in overmap:
#                 j, pos = overmap[i]
#                 start = sum(num_subtokens[j][:pos])
#                 a[j][start:] = sent
#             else:
#                 a.append(sent)
#         print('\nRESTORE:')
#         print_list(a)
        '''
        encoded_sentences = [
            self.tokenizer.encode_plus(text=sent,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       pad_to_max_length=True,
                                       return_tensors='pt',
                                       return_attention_mask=True,
                                       return_overflowing_tokens=False)
                for sent in tokenized_sentences
        ]
        input_ids, attention_masks = zip(*[
            (x['input_ids'], x['attention_mask'])
                for x in encoded_sentences
        ])

        data = []
        print(num_sents, len(input_ids), len(attention_masks))
        for batch_i in range(0, num_sents, batch_size):

            print(i)
            with torch.no_grad():
                hiddens = self.model(
                    torch.cat(
                        input_ids[batch_i:batch_i + batch_size],
                        dim=0
                    ).to(device),
                    token_type_ids=None,
                    attention_mask=torch.cat(
                        attention_masks[batch_i:batch_i + batch_size],
                        dim=0
                    ).to(device)
                )[-1]

                hiddens = self._aggregate_hidden_states(
                    hiddens, layer_ids=hidden_ids,
                    aggregate_op=aggregate_hiddens_op
                )

                for i, sent in enumerate(hiddens, start=batch_i):
                    if i in overmap:
                        j, over_pos_start = overmap[i]
                        over_pos_end = sub_to_kens[j][len(data[j])]
                        overlap = over_pos_end - over_pos_start
                        if overlap > OVERLAP_BORDER + OVERLAP_BORDER:
                            start1 = over_pos_start + OVERLAP_BORDER
                            end1 = over_pos_end - OVERLAP_BORDER
                            overlap = end1 - start1
                            half1 = overlap - int(overlap / 2)
                            half2 = overlap - half1
                            half = half1 + (1 if half1 == half2 else 0)
                            for k in range(half2):
                                coef = (k + 1) / half / 2
                                data[j][start1 + k] = \
                                    data[j][start1 + k] * (1 - coef) \
                                  + sent[OVERLAP_BORDER + k] * coef
                                k_ = overlap - k - k
                                data[j][start1 + k_] = \
                                    data[j][start1 + k_] * coef \
                                  + sent[OVERLAP_BORDER + k_] * (1 - coef)
                            if half1 != half2:
                                data[j][start1 + half1] = (
                                    data[j][start1 + half1]
                                  + sent[OVERLAP_BORDER + half1]
                                ) / 2
                            start = OVERLAP_BORDER + overlap
                        else:
                            start = token_starts[j][
                                over_pos_start + overlap - int(overlap / 2)
                            ]
                            
                        data[j] = torch.cat([data[j][:start], sent], dim=0)
                    else:
                        data.append(sent)

        print(len(num_subtokens))
        print('d', len(data))
        for i, token_lens in enumerate(num_subtokens):
            token_lens = num_subtokens[i]
            '''
        for i in range(len(data)):
            token_lens = num_subtokens[i]
            '''
            start = 0
            sent = data[i]
            sent_ = []
            for token_len in token_lens:
                end = start + token_len
                vecs = sent[start:end]
                if aggregate_subtokens_op != None:
                    vecs = self._aggregate_hidden_states(
                        vecs, layer_ids=None,
                        aggregate_op=aggregate_subtokens_op
                    )
                sent_.append(vecs)
                start = end

            data[i] = sent_ if aggregate_subtokens_op == None else \
                      torch.stack(sent_, dim=0)

        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def _frame_collate(self, batch, pos, with_lens=True,
                       with_token_lens=True):
        """The method to use with `junky.dataset.FrameDataset`.

        :param pos: position of the data in *batch*.
        :type pos: int
        :with_lens: return lentghs of data.
        :with_token_lens: return lengths of tokens of the data.
        :return: depends on keyword args.
        :rtype: if the `.transform()` method was called with
            *aggregate_subtokens_op*=None:
                tuple(list([torch.tensor]), lens:torch.tensor,
                      token_lens:list([torch.tensor]))
            otherwise: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        device = CPU
        pad = 0.

        if isinstance(batch[0][pos], Tensor):
            if batch[0][pos].is_cuda:
                device = batch[0][pos].get_device()
            lens = [tensor([len(x[pos]) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)] if with_lens else []
            x = pad_sequences_with_tensor([x[pos] for x in batch],
                                          padding_tensor=pad)

        else:
            for x in batch:
                x_ = x[pos]
                if x_:
                    if x_[0].is_cuda:
                        device = x_[0].get_device()
                        tensor_dtype = x_[0].dtype
                    break
            lens = [tensor([len(x[pos]) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)] if with_lens else []
            if with_token_lens:
                lens.append([tensor([len(x) for x in x[pos]], device=device,
                                    dtype=self.int_tensor_dtype)
                                 for x in batch])
            x = pad_array_torch([x[pos] for x in batch], padding_value=pad,
                                device=device, dtype=tensor_dtype)

        return (x, *lens) if lens else x

    def _collate(self, batch):
        """The method to use with `DataLoader`.

        :rtype: if the `.transform()` method was called with
            *aggregate_subtokens_op*=None:
                tuple(list([torch.tensor]), lens:torch.tensor,
                      token_lens:list([torch.tensor]))
            otherwise: tuple(list([torch.tensor]), lens:torch.tensor)
        
        """
        device = CPU
        pad = 0.

        if isinstance(batch[0], Tensor):
            if batch[0].is_cuda:
                device = batch[0].get_device()
            lens = [tensor([len(x) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)]
            x = pad_sequences_with_tensor(batch, padding_tensor=pad)

        else:
            for x in batch:
                if x:
                    if x[0].is_cuda:
                        device = x[0].get_device()
                        tensor_dtype = x[0].dtype
                    break
            lens = [tensor([len(x) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)]
            lens.append([tensor([len(x) for x in x], device=device,
                                dtype=self.int_tensor_dtype) for x in batch])
            x = pad_array_torch(batch, padding_value=pad,
                                device=device, dtype=tensor_dtype)

        return x, *lens
