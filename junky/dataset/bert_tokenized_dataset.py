# -*- coding: utf-8 -*-
# junky lib: dataset.BertDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset that prepare text
sentences for `transformers.BertModel` input.
"""
from junky.dataset.base_dataset import BaseDataset
import logging
import sys
from torch import int64, tensor
from tqdm import tqdm

logging.getLogger('transformers').setLevel(logging.ERROR)


class BertTokenizedDataset(BaseDataset):
    """
    `torch.utils.data.Dataset` for `transformers.BertModel` input.

    Args:
        tokenizer: the tokenizer from `transformers` package corresponding to
            `model` chosen.
        int_tensor_dtype: dtype for int tensors: torch.dtype.
        sentences (`list([str])|list([list([str])])`): If not ``None``, they
            will be transformed and saved. NB: All the sentences must not be
            empty.
        All other args are params for the `.transform()` method. They are used
            only if *sentences* is not ``None``. You can use any args but
            `save` that is set to `True`.
    """
    def __init__(self, tokenizer, int_tensor_dtype=int64,
                 sentences=None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.int_tensor_dtype = int_tensor_dtype
        if sentences:
            self.transform(sentences, save=True, **kwargs)

    def _pull_xtrn(self):
        xtrn = self.tokenizer
        self.tokenizer = None
        return xtrn

    def _push_xtrn(self, xtrn):
        self.tokenizer = xtrn

    def transform(self, sentences, add_special_tokens=True, is_pretokenized=False,
                  max_len=None, save=True, append=False):
        """Convert text *sentences* to the `transformers.BertModel` input.
        Already tokenized sentences are also allowed but fill be joined before
        tokenizing with space character.

        *max_len*, *add_special_tokens* and *is_pretokenized* are params for
            the tokenizer. *max_len* ``None`` (default) or `0` means the
            highest number of subtokens for the model (usually, `512`).

        If *save* is ``True``, we'll keep the converted sentences as the
        Dataset source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""

        if not max_len:
            max_len = self.tokenizer.max_len
        assert max_len >= 16, 'ERROR: max len must be >= 16'
        assert max_len <= self.tokenizer.max_len, \
               'ERROR: max len must be <= {}'.format(self.tokenizer.max_len)

        data = [
            self.tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                max_length=max_len,
                is_pretokenized=is_pretokenized,
                pad_to_max_length=False,
                return_tensors=None,
                return_attention_mask=True,
                return_overflowing_tokens=False
            ) for sent in tqdm(
                iterable=sentences
                    if sentences and (isinstance(sentences[0], str)
                                   or is_pretokenized == False) else
                [' '.join(x for x in x) for x in sentences],
                mininterval=2, file=sys.stdout
            )
        ]
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def _collate(self, batch, with_lens=True):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`.

        :with_lens: return lengths of data.
        :return: depends on keyword args.
        :rtype: tuple(
            dict({'input_ids': tensor([batch x batch max len]),
                  'token_type_ids': tensor([batch x batch max len]),
                  'attention_mask': tensor([batch x batch max len])}),
            lens:torch.tensor
        )
        """
        batch_ = {}
        [batch_.setdefault(k, []).append(v)
             for x in batch
             for k, v in x.items()]
        lens = [tensor([len(x['input_ids']) for x in batch],
                       dtype=self.int_tensor_dtype)] if with_lens else []
        return (self.tokenizer.pad(batch_, padding='longest',
                                   return_tensors='pt'),
                *lens)
