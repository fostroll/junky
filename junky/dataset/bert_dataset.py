# -*- coding: utf-8 -*-
# junky lib: dataset.BertDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for word-level input.
"""
import logging
import sys
#logging.basicConfig(level=logging.ERROR)
#if not sys.warnoptions:
#    import warnings
#    warnings.simplefilter('ignore')
#    os.environ['PYTHONWARNINGS'] = 'ignore'
from junky import CPU, absmax_torch, pad_array_torch, \
                  pad_sequences_with_tensor
from junky.dataset.base_dataset import BaseDataset
from time import time
import torch
from torch import Tensor, tensor
from tqdm import tqdm

# to suppress transformers' warnings
#logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
#logging.getLogger('pytorch_pretrained_bert.tokenization').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


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
        All other args are params for the `.transform()` method. They are used
            only if *sentences* is not ``None``. You can use any args but
            `save` that is set to `True`.
    """
    overlap_shift = .5
    overlap_border = 2
    use_batch_max_len = True
    sort_dataset = True

    @property
    def vec_size(self):
        return self.data[0].shape[-1] if self.data else \
               self.model.config.hidden_size

    def __init__(self, model, tokenizer, int_tensor_dtype=torch.int64,
                 sentences=None, **kwargs):
        assert model.config.output_hidden_states, \
               'ERROR: BERT model was created with ' \
               '`output_hidden_states=False` option. To use it in ' \
               'BertDataset, the model must be created with ' \
               '`output_hidden_states=True`'
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.int_tensor_dtype = int_tensor_dtype
        if sentences:
            self.transform(sentences, save=True, **kwargs)

    def _pull_xtrn(self):
        xtrn = self.model, self.tokenizer
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

            if aggregate_op == 'absmax':
                hiddens = absmax_torch(hiddens, dim=0)
            elif aggregate_op == 'cat':
                hiddens = torch.cat(hiddens, dim=-1)
            elif aggregate_op == 'first':
                hiddens = hiddens[0]
            elif aggregate_op == 'last':
                hiddens = hiddens[-1]
            elif aggregate_op == 'max':
                hiddens = torch.max(hiddens, dim=0)[0]
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
                  "(choose one of ['absmax', 'cat', 'max', 'mean', 'sum'])"
                )
        return hiddens

    def transform(self, sentences, max_len=None, batch_size=64,
                  hidden_ids=0, aggregate_hiddens_op='mean',
                  aggregate_subtokens_op='absmax', to=None,
                  save=True, append=False, loglevel=1):
        """Convert *sentences* of words to the sequences of the corresponding
        contextual vectors and adjust their format for Dataset.

        *max_len* is a param for tokenizer. We'll transform lines of any
            length, but the quality is higher if *max_len* is greater.
            ``None`` (default) or `0` means the maximum for the model
            (usually, `512`).

        *batch_size* affects only on the execution time. Greater is faster,
            but big *batch_size* may be cause of CUDA Memory Error. If
            ``None`` or `0`, we'll try to convert all *sentences* with one
            batch.

        *hidden_ids*: hidden score layers that we need to aggregate. Allowed
            int or tuple of ints. If ``None``, we'll aggregate all the layers.

        *aggregate_hidden_op*: how to aggregate hidden scores. The ops
            allowed: 'absmax', 'cat', 'max', 'mean', 'sum'. For the 'absmax'
            method we take into account absolute values of the compared items.

        *aggregate_subtokens_op*: how to aggregate subtokens vectors to form
            only one vector for each input token. The ops allowed: ``None``,
            'absmax', 'first', 'last', 'max', 'mean', 'sum'. For the 'absmax'
            method we take into account absolute values of the compared items.

        If you want to get the result placed on some exact device, specify the
        device with *to* param. If *to* is ``None`` (defautl), data will be
        placed to the very device that `bs.model` is used.

        If *save* is ``True``, we'll keep the converted sentences as the
        `Dataset` source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is ``True``.

        *loglevel* can be set to `0`, `1` or `2`. `0` means no output.

        The result is depend on *aggregate_subtokens_op* param. If it is
        ``None``, then for each token we keeps in the result a tensor with
        stacked vectors for all its subtokens. Otherwise, if any
        *aggregate_subtokens_op* is used, each sentence will be converted to
        exactly one tensor of shape [<sentence length>, <vector size>]."""

        if not max_len:
            max_len = self.tokenizer.max_len
        assert max_len >= 16, 'ERROR: max len must be >= 16'
        assert max_len <= self.tokenizer.max_len, \
               'ERROR: max len must be <= {}'.format(self.tokenizer.max_len)
        valid_ops = ['absmax', 'cat', 'max', 'mean', 'sum']
        assert aggregate_hiddens_op in valid_ops, \
               'ERROR: unknown aggregate_hidden_op (choose one of {})' \
                   .format(valid_ops)
        valid_ops = [None, 'absmax', 'first', 'last', 'max', 'mean', 'sum']
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
        max_len_ = max_len - 2  # for [CLS] and [SEP]

        overlap_border = max_len if self.overlap_border is None else \
                         0 if self.overlap_border < 0 else \
                         self.overlap_border
        shift = int(max(min(max_len_, self.overlap_shift)
                        if self.overlap_shift >= 1 else
                    (max_len_ * self.overlap_shift), 1))

        _src = sentences
        if loglevel >= 2:
            print('Tokenizing')
            _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)
        time0 = time()
        # tokenize each token separately by the BERT tokenizer
        tokenized_sentences = \
            [[x for x in self.tokenizer.tokenize(x) or ['[UNK]']]
                 for x in _src] \
                if sentences and isinstance(sentences[0], str) else \
            [[self.tokenizer.tokenize(x) or ['[UNK]'] for x in x] \
                 for x in _src]
        if time() - time0 < 2 and loglevel == 1:
            loglevel = 0

        # number of subtokens in tokens of tokenized_sentences
        num_subtokens = [[len(x) for x in x] for x in tokenized_sentences]
        # for each subtoken we keep index of its token
        sub_to_kens = [[
            x for x in [[i] * x for i, x in enumerate(x)]
              for x in x
        ] for x in num_subtokens]
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

        ## sort tokenized sentences by lenght
        ######
        if self.sort_dataset:
            sent_lens, sorted_sent_ids = zip(*sorted(
                [(x, i) for i, x in enumerate(sent_lens)],
                reverse=True
            ))
            sent_lens = list(sent_lens)
            tokenized_sentences = [
                tokenized_sentences[i] for i in sorted_sent_ids
            ]
            sub_to_kens = [
                sub_to_kens[i] for i in sorted_sent_ids
            ]
            token_starts = [
                token_starts[i] for i in sorted_sent_ids
            ]
            ######

        def process_long_sentences(sents, sent_lens, sub_to_kens,
                                   token_starts):
            overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
            overlap_token_starts, overmap = [], [], [], [], []
            for i, (sent, sent_len, token_ids, sub_ids) \
                    in enumerate(zip(sents, sent_lens, sub_to_kens,
                                     token_starts)):
                if sent_len > max_len_:
                    # находим индекс токена для сабтокена с шифтом
                    pos = token_ids[shift]
                    # вычитаем токен нулевого сабтокена, получаем индекс
                    # токена относительно текущего начала
                    pos_ = pos - token_ids[0]
                    if not pos_:
                        pos, pos_ = token_ids[sub_ids[1] - sub_ids[0]], 1
                    # находим индекс сабтокена: из индекса сабтокена
                    # найденного токена вычитаем индекс текущего нулевого
                    # сабтокена
                    start = sub_ids[pos_] - sub_ids[0]
                    if start > max_len_:
                        raise RuntimeError(
                            ('ERROR: too long token (longer than '
                             'effective max_len):\n{}')
                                 .format(sent[:start])
                        )
                    overlap_sents.append(sent[start:])
                    overlap_sent_lens.append(sent_len - start)
                    overlap_sub_to_kens.append(token_ids[start:])
                    overlap_token_starts.append(sub_ids[pos_:])
                    overmap.append((i, pos_))
                    end = token_ids[max_len_] - token_ids[0]
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

        splitted_sent_lens = [len(x) for x in tokenized_sentences]

        if not batch_size:
            batch_size = num_sents

        data = []
        _src = range(0, num_sents, batch_size)
        if loglevel:
            print('Vectorizing')
            _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)

        for batch_i in _src:
            batch_max_len = min(
                max(splitted_sent_lens[batch_i:batch_i + batch_size]) + 2,
                max_len
            ) if self.use_batch_max_len else max_len

            encoded_sentences = [
                self.tokenizer.encode_plus(text=sent,
                                           add_special_tokens=True,
                                           max_length=batch_max_len,
                                           pad_to_max_length=True,
                                           #truncation=True,
                                           return_tensors='pt',
                                           return_attention_mask=True,
                                           return_overflowing_tokens=False)
                    for sent in tokenized_sentences[batch_i:batch_i
                                                  + batch_size]
            ]
            input_ids, attention_masks = zip(*[
                (x['input_ids'], x['attention_mask'])
                    for x in encoded_sentences
            ])

            with torch.no_grad():
                hiddens = self.model(
                    torch.cat(input_ids, dim=0).to(device),
                    token_type_ids=None,
                    attention_mask=torch.cat(attention_masks, dim=0)
                                        .to(device)
                )[-1]

                hiddens = self._aggregate_hidden_states(
                    hiddens, layer_ids=hidden_ids,
                    aggregate_op=aggregate_hiddens_op
                )

                if to:
                    hiddens = hiddens.to(to)

                for i, sent in enumerate(hiddens, start=batch_i):
                    sent_len = splitted_sent_lens[i]
                    if i in overmap:
                        j, over_pos_start = overmap[i]
                        over_pos_end = sub_to_kens[j][len(data[j])]
                        overlap = over_pos_end - over_pos_start
                        if overlap > overlap_border * 2:
                            start1 = over_pos_start + overlap_border
                            end1 = over_pos_end - overlap_border
                            overlap = end1 - start1
                            half2 = overlap // 2
                            half1 = overlap - half2
                            half = half1 + (1 if half1 == half2 else 0)
                            for k in range(half2):
                                coef = (k + 1) / half / 2
                                data[j][start1 + k] = \
                                    data[j][start1 + k] * (1 - coef) \
                                  + sent[overlap_border + k] * coef
                                k_ = overlap - k - k
                                data[j][start1 + k_] = \
                                    data[j][start1 + k_] * coef \
                                  + sent[overlap_border + k_] * (1 - coef)
                            if half1 != half2:
                                data[j][start1 + half1] = (
                                    data[j][start1 + half1]
                                  + sent[overlap_border + half1]
                                ) / 2
                            end1 = token_starts[j][end1]
                        else:
                            start2 = overlap - overlap // 2
                            end1 = token_starts[j][over_pos_start + start2]

                        start2 = end1 - token_starts[j][over_pos_start]
                        data[j] = torch.cat([
                            data[j][:end1], sent[1 + start2:1 + sent_len]
                        ], dim=0)
                    else:
                        data.append(sent[1:1 + sent_len])

        ## sort data in original order
        ######
        if self.sort_dataset:
            _, data = zip(*sorted(
                [(i, x) for i, x in enumerate(data)],
                key=lambda x: sorted_sent_ids[x[0]]
            ))
            data = list(data)
        ######

        _src = num_subtokens
        if loglevel:
            print('Reordering')
            _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)
        for i, token_lens in enumerate(_src):
            token_lens = num_subtokens[i]
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

    def _collate(self, batch, with_lens=True, with_token_lens=True):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`.

        :with_lens: return lengths of data.
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

        if isinstance(batch[0], Tensor):
            if batch[0].is_cuda:
                device = batch[0].get_device()
            lens = [tensor([len(x) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)] if with_lens else []
            x = pad_sequences_with_tensor(batch, padding_tensor=pad)

        else:
            for x in batch:
                if x:
                    if x[0].is_cuda:
                        device = x[0].get_device()
                    tensor_dtype = x[0].dtype
                    break
            lens = [tensor([len(x) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)] if with_lens else []
            if with_token_lens:
                lens.append([tensor([len(x) for x in x], device=device,
                                    dtype=self.int_tensor_dtype)
                                 for x in batch])
            x = pad_array_torch(batch, padding_value=pad,
                                device=device, dtype=tensor_dtype)

        return (x, *lens)
