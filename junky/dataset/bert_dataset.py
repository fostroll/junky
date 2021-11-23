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
    overlap_shift = .5  # Defines the overlap's `shift` from the sentence's
    # start. We count it in words, so, if sentence has `9` words, the shift
    # will be `int(.5 * 9)`, i.e., `4`. The minimum value for `shift` is `1`.
    # If you set `ds.overlap_shift` > `1`, we will treat it as absolute value
    # (but reduce it to `max_len` if your `ds.overlap_shift` would be greater.
    overlap_border = 2  # The overlap is processed as follows. The left zone
    # of width equal to `ds.overlap_border` is taken from the earlier part of
    # the sentence; the right zone - from the later. The zone between borders
    # is calculated as weighted sum of both parts. The weights are
    # proportional to the distance to the middle of the zone: the beginning
    # has dominance to the left from the middle, the ending has dominance to
    # the right. In the very middle (if it exists), both weights are equal to
    # `.5`. If you set `ds.overlap_border` high enough (greater than
    # `(max_len - shift) / 2`) or `None`, it will be set to the middle of the
    # overlap zone. Thus, weighted algorithm will be dwindled.  Also note that
    # weights are applied to tokens (not subtokens). I.e. all subtokens of any
    # particular token have equal weights when summing.
    use_batch_max_len = True  # Either we want to use the length of the
    # longest sentence in the batch instead of the `max_len` param of
    # `.transform()`. We use it only if that length is less than `max_len`,
    # and as result, with high *max_len*, we have a substantial speed increase
    # without any quality change or resulting data.
    sort_dataset = True  # Do we want to sort the dataset before feeding it to
    # `ds.model`. With high **max_len** it highly increases processing speed,
    # and affects resulting data only because of different sentences' grouping
    # (deviation is about `1e-7`).

    @property
    def vec_size(self):
        if self.data:
            if isinstance(self.data[0], list):
                res = self.data[0][0].shape[-1]
            else:
                res = self.data[0].shape[-1]
        else:
            res = self.model.config.hidden_size
        return res

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
                  with_grad=False, save=True, append=False, loglevel=1):
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
            'absmax', 'append', 'first', 'last', 'max', 'mean', 'sum'. For the
            'absmax' method we take into account absolute values of the
            compared items.

        If you want to get the result placed on some exact device, specify the
        device with *to* param. If *to* is ``None`` (defautl), data will be
        placed to the very device that `ds.model` is used.

        *with_grad*: calculate gradients during forward propagation through
        self.model. Default is ``False``.

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

        tokenizer_max_len = self.tokenizer.max_len \
                                if hasattr(self.tokenizer, 'max_len') else \
                            self.tokenizer.model_max_len \
                                if hasattr(self.tokenizer,
                                           'model_max_len') else \
                            512

        if not max_len:
            max_len = tokenizer_max_len
        assert max_len >= 16, 'ERROR: max len must be >= 16'
        assert max_len <= tokenizer_max_len, \
               'ERROR: max len must be <= {}'.format(tokenizer_max_len)
        valid_ops = ['absmax', 'cat', 'max', 'mean', 'sum']
        assert aggregate_hiddens_op in valid_ops, \
               'ERROR: unknown aggregate_hidden_op (choose one of {})' \
                   .format(valid_ops)
        valid_ops = [None, 'absmax', 'append', 'first', 'last', 'max', 'mean',
                     'sum']
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
        # how many subtokens should be overlapped, e.g.:
        # max_len = 100, shift = 50, array = [0, 1, 2, ..., 120]
        # then array1 = [0, 1, 2, ..., 99], array2 = [50, 51, 52, ..., 120]
        # for every sentence it is rounded to tokens' borders (including)

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
        ] for x in num_subtokens]  # [[0, 0, 0, 1, 2, 3, 3, 4, ...], ...]
        # for each token we keep its start in flatten tokenized_sentences
        token_starts = []  # [[0, 3, 4, 5, 7, ...], ...]
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

        need_sort_dataset = self.sort_dataset \
                        and self.use_batch_max_len \
                        and len(tokenized_sentences) > batch_size
        ## sort tokenized sentences by lenght
        ######
        if need_sort_dataset:
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
                                   token_starts, first_sent_idx):
            overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
            overlap_token_starts, overmap = [], [], [], [], []
            for sent_idx, (sent, sent_len, sub_token_ids, token_sub_ids) \
                    in enumerate(zip(sents, sent_lens, sub_to_kens,
                                     token_starts)):
                if sent_len > max_len_:
                    # индекс первого сабтокена в начальном слове фрагмента
                    # и индекс токена для первого сабтокена во фрагменте
                    sub0_token, token0_sub = \
                        sub_token_ids[0], token_sub_ids[0]
                    # индекс начального токена для области пересечения
                    over_tok = sub_token_ids[max_len_ - shift]
                    # вычитаем токен нулевого сабтокена, получаем индекс
                    # токена относительно текущего начала (если фразу уже
                    # делили, то это будет не 0)
                    over_tok_ = over_tok - sub0_token
                    # если в результате получаем 0, то переносим на
                    # следующий токен
                    if over_tok_ == 0:
                        over_tok += 1
                        over_tok_ = 1
                    # находим индекс сабтокена: из индекса сабтокена
                    # найденного токена вычитаем индекс текущего нулевого
                    # сабтокена
                    over_sub_ = token_sub_ids[over_tok_] - token0_sub
                    if over_sub_ > max_len_:
                        # it can appear only when over_tok_ == 1
                        raise RuntimeError(
                            ('ERROR: too long token (longer than '
                            f'effective max_len):\n{sent[:over_sub_]}')
                        )
                    overlap_sents.append(sent[over_sub_:])
                    overlap_sent_lens.append(sent_len - over_sub_)
                    overlap_sub_to_kens.append(sub_token_ids[over_sub_:])
                    overlap_token_starts.append(token_sub_ids[over_tok_:])
                    # индекс начального токена для отрезаемого фрагмента
                    cut_tok = sub_token_ids[max_len_]
                    # относительный индекс начального токена для отрезаемого
                    # фрагмента
                    cut_tok_ = cut_tok - sub0_token
                    cut_sub_ = token_sub_ids[cut_tok_] - token0_sub
                    sent[cut_sub_:] = []  # cut long sentence
                    overmap.append((first_sent_idx + sent_idx,
                                    over_tok, cut_tok))
            return overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
                   overlap_token_starts, overmap

        overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
        overlap_token_starts, overmap = process_long_sentences(
            tokenized_sentences, sent_lens, sub_to_kens,
            token_starts, 0
        )
        num_sents = len(tokenized_sentences)
        overmap = {num_sents + i: x for i, x in enumerate(overmap)}
        # overlap_sents: cutted parts of tokenized_sentences
        # overlap_sent_lens: their lens in subtokens
        # overlap_sub_to_kens: [sub -> token] map, indexes are absolute!
        # overlap_token_starts: [token_start -> first sub] map, absolute!
        # overmap: {sent_no: (orig_sent_no, orig_start_token, orig_end_token)}
        #     (orig_end_token is not included)

        while overlap_sents:
            tokenized_sentences += overlap_sents
            sent_lens += overlap_sent_lens
            sub_to_kens += overlap_sub_to_kens
            token_starts += overlap_token_starts
            overlap_sents, overlap_sent_lens, overlap_sub_to_kens, \
            overlap_token_starts, overmap_ = process_long_sentences(
                overlap_sents, overlap_sent_lens, overlap_sub_to_kens,
                overlap_token_starts, num_sents
            )
            num_sents = len(tokenized_sentences)
            for sent_idx, (orig_sent_idx, over_tok, cut_tok) in enumerate(overmap_):
                orig_sent_idx_ = overmap.get(orig_sent_idx)
                if orig_sent_idx_:
                    orig_sent_idx = orig_sent_idx_[0]
                overmap[num_sents + sent_idx] = \
                    (orig_sent_idx, over_tok, cut_tok)

        splitted_sent_lens = [len(x) for x in tokenized_sentences]

        if not batch_size:
            batch_size = num_sents

        data = []
        _src = range(0, num_sents, batch_size)
        if loglevel:
            print('Vectorizing')
            _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)

        ####### workaround for transformers v.4
        test_enc = self.tokenizer.encode_plus(
            text='', add_special_tokens=True, max_length=3,
            padding='max_length', return_tensors=None,
            return_token_type_ids=False, return_attention_mask=True
        )
        cls_id, sep_id, pad_id = test_enc['input_ids']
        att_mask1, _, att_mask0 = test_enc['attention_mask']
        #######

        for batch_i in _src:
            ####### workaround for transformers v.4
            '''
            batch_max_len = min(
                max(splitted_sent_lens[batch_i:batch_i + batch_size]) + 2,
                max_len
            ) if self.use_batch_max_len else max_len

            encoded_sentences = [
                self.tokenizer.encode_plus(text=sent,
                                           add_special_tokens=True,
                                           max_length=batch_max_len,
                                           #truncation=True,
                                           #pad_to_max_length=True,
                                           padding='longest',
                                           return_tensors='pt',
                                           return_attention_mask=True,
                                           return_overflowing_tokens=False)
                    for sent in tokenized_sentences[batch_i:batch_i
                                                          + batch_size]
            ]
            encoded_sentences = []
            for sent in tokenized_sentences[batch_i:batch_i + batch_size]:
                try:
                    self.tokenizer.encode_plus(text=sent,
                                               add_special_tokens=True,
                                               max_length=batch_max_len,
                                               #truncation=True,
                                               #pad_to_max_length=True,
                                               padding='longest',
                                               return_tensors='pt',
                                               return_attention_mask=True,
                                               return_overflowing_tokens=False)
                except TypeError as e:
                    print('batch_max_len =', batch_max_len)
                    print('type(sent) =', type(sent))
                    print('sent = [{}]'.format(sent))
                    raise e'''

            batch_max_len = min(
                max(splitted_sent_lens[batch_i:batch_i + batch_size]),
                max_len - 2
            ) if self.use_batch_max_len else max_len - 2

            input_ids = tensor(
                [([cls_id]
                + self.tokenizer.convert_tokens_to_ids(sent)
                + [sep_id] + [pad_id] * (batch_max_len - len(sent)))
                     for sent in tokenized_sentences[batch_i:batch_i
                                                           + batch_size]],
                dtype=self.int_tensor_dtype
            )
            attention_mask = (input_ids != pad_id).type(self.int_tensor_dtype)
            '''
            input_ids, attention_masks = zip(*[
                (x['input_ids'], x['attention_mask'])
                    for x in encoded_sentences
            ])'''
            #######

            with torch.set_grad_enabled(with_grad):
                hiddens = self.model(
                    #torch.cat(input_ids, dim=0).to(device),
                    input_ids.to(device),
                    token_type_ids=None,
                    #attention_mask=torch.cat(attention_masks, dim=0)
                    #                    .to(device)
                    attention_mask=attention_mask.to(device)
                )[-1]

                hiddens = self._aggregate_hidden_states(
                    hiddens, layer_ids=hidden_ids,
                    aggregate_op=aggregate_hiddens_op
                )

                if to:
                    hiddens = hiddens.to(to)
                data_device = hiddens.device

                # sub_to_kens: [sub -> token] map, indexes are absolute!
                # token_starts: [token -> first sub] map, absolute!
                # overmap: {sent_no: (orig_sent_no, orig_start_token,
                #                     orig_end_token)}
                #     (orig_end_token is not included)
                for sent_idx, sent in enumerate(hiddens, start=batch_i):
                    sent_fin = 1 + splitted_sent_lens[sent_idx]
                    if sent_idx in overmap:
                        orig_sent_idx, over_tok, cut_tok = overmap[sent_idx]
                        orig_data = data[orig_sent_idx]
                        overlap = cut_tok - over_tok  # overlap is in tokens
                        # overlap_border: how many tokens on the rims we don't
                        # touch
                        if overlap > overlap_border * 2:
                            over_tok_ = over_tok + overlap_border
                                # change starts here
                            cut_tok_ = cut_tok - overlap_border
                                # change ends here (not including)
                            overlap = cut_tok_ - over_tok_
                                # length of changeable area
                            sent_over_tok_ = overlap_border
                                # start of the eff part of the sent
                            sent_cut_tok_ = sent_over_tok_ + overlap
                                # end of the eff part of the sent
                            step = 1 / (overlap + 1)
                            token_starts_ = token_starts[orig_sent_idx]
                            over_sub_ = token_starts_[over_tok_]
                            cut_sub_ = token_starts_[cut_tok_]
                            weights = (
                                torch.tensor(sub_to_kens[orig_sent_idx]
                                                        [over_sub_:cut_sub_],
                                             device=data_device)
                              - over_tok_ + 1
                            ) * step
                            weights.unsqueeze_(1)
                            sent_token_starts_ = token_starts[sent_idx]
                            sent_token0_start_ = sent_token_starts_[0]
                            sent_over_sub_ = \
                                1 + sent_token_starts_[sent_over_tok_] \
                                  - sent_token0_start_
                            sent_cut_sub_ = \
                                1 + sent_token_starts_[sent_cut_tok_] \
                                  - sent_token0_start_
                            # weighs are used to merge the overlap zone
                            overlap_data = \
                                orig_data[over_sub_:cut_sub_] * (1
                                                               - weights) \
                              + sent[sent_over_sub_:sent_cut_sub_] * weights

                        else:
                            over_sub_ = overlap_border
                            sent_cut_sub_ = 1 + over_sub_
                            overlap_data = None

                        data[orig_sent_idx] = torch.cat((
                            orig_data[:over_sub_],
                            *([] if overlap_data is None else [overlap_data]),
                            sent[sent_cut_sub_:sent_fin]
                        ), dim=0)

                    else:
                        data.append(sent[1:sent_fin])

        ## sort data in original order
        ######
        if need_sort_dataset:
            _, data = zip(*sorted(
                [(i, x) for i, x in enumerate(data)],
                key=lambda x: sorted_sent_ids[x[0]]
            ))
            data = list(data)
        ######

        if aggregate_subtokens_op != 'append':
            _src = num_subtokens
            if loglevel:
                print('Reordering')
                _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)
            for i, token_lens in enumerate(_src):
                #token_lens = num_subtokens[i]
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

    def _collate(self, batch, with_lens=True, with_token_lens=True,
                 append_subtokens=False):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`.

        :with_lens: return lengths of data.
        :with_token_lens: return lengths of tokens of the data (only allowed
            if `.transform()` was called with `aggregate_subtokens_op=None`).
        :append_subtokens: line up tokens if `.transform()` was called with
            `aggregate_subtokens_op=None`. The result will be exactly as if
            `aggregate_subtokens_op='append'` but you can get token lens.
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
            lens = [tensor([len([x for x in x for x in x]) for x in batch],
                           device=device, dtype=self.int_tensor_dtype)] \
                       if with_lens and append_subtokens else \
                   [tensor([len(x) for x in batch], device=device,
                           dtype=self.int_tensor_dtype)] \
                       if with_lens else \
                   []
            if with_token_lens:
                lens.append([tensor([len(x) for x in x], device=device,
                                    dtype=self.int_tensor_dtype)
                                 for x in batch])
            if append_subtokens:
                x = pad_sequences_with_tensor(
                    [torch.vstack([x for x in x for x in x]) for x in batch],
                    padding_tensor=pad
                )
            else:
                x = pad_array_torch(batch, padding_value=pad,
                                    device=device, dtype=tensor_dtype)

        return (x, *lens)
