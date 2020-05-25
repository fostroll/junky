# -*- coding: utf-8 -*-
# junky lib: TokenDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for token-level input.
"""
from junky import make_token_dict
from junky.dataset import BaseDataset
from torch import Tensor, int64, tensor
from torch.nn.utils.rnn import pad_sequence


class TokenDataset(BaseDataset):
    """
    torch.utils.data.Dataset for token-level input.

    Args:
        sentences: sequences of tokens: list([list([str])]).
        unk_token: add a token for tokens that are not present in the dict:
            str.
        pad_token: add a token for padding: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        transform: if ``True``, transform and save `sentences`.
        skip_unk, keep_empty: params for the `transform()` method.
        int_tensor_dtype: dtype for int tensors: torch.dtype
        batch_first: if ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
    """
    def __init__(self, sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, transform=False, skip_unk=False,
                 keep_empty=False, int_tensor_dtype=int64, batch_first=False):
        super().__init__()
        self.int_tensor_dtype = int_tensor_dtype
        self.batch_first = batch_first
        self.fit(sentences, unk_token=unk_token, pad_token=pad_token,
                 extra_tokens=extra_tokens)
        if transform:
            self.transform(sentences, skip_unk=skip_unk,
                           keep_empty=keep_empty, save=True)

    def fit(self, sentences, unk_token=None, pad_token=None,
            extra_tokens=None):
        """Recreate the internal dict.

        :param sentences: sequences of tokens.
        :type sentences: list([list([str])])
        :param unk_token: add a token for tokens that are not present in the
            dict.
        :type unk_token: str
        :param pad_token: add a token for padding.
        :type pad_token: str
        :param extra_tokens: add tokens for any other purposes.
        :type extra_tokens: list([str])
        """
        if unk_token:
            extra_tokens = (extra_tokens if extra_tokens else []) \
                         + [unk_token]
        self.transform_dict, self.pad, extra = \
            make_token_dict(sentences,
                            pad_token=pad_token,
                            extra_tokens=extra_tokens)
        self.unk = extra[-1] if unk_token else None
        self.reconstruct_dict = {y: x for x, y in self.transform_dict.items()}

    def token_to_idx(self, token, skip_unk=True):
        """Convert a token to its index. If the token is not present in the
        internal dict, return index of unk token or None if it's not
        defined."""
        return self.transform_dict[token] \
                   if token in self.transform_dict else \
               self.unk if not skip_unk and self.unk else \
               None

    def idx_to_token(self, idx, skip_unk=True):
        """Convert an index to the corresponding token. If the index is not
        present in the internal dict, return unk token or None if it's not
        defined."""
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        return self.reconstruct_dict[idx] \
                   if idx in self.reconstruct_dict else \
               self.reconstruct_dict[self.unk] \
                   if not skip_unk and self.unk else \
               ''

    def transform_tokens(self, tokens, skip_unk=False):
        """Convert a token or a list of tokens to the corresponding
        index|list of indices. If skip_unk is ``True``, unknown tokens will be
        skipped."""
        return self.token_to_idx(tokens, skip_unk=skip_unk) \
                   if isinstance(tokens, str) else \
               [self.token_to_idx(t, skip_unk=skip_unk) for t in tokens]

    def reconstruct_tokens(self, ids, skip_unk=False):
        """Convert an index or a list of indices to the corresponding
        token|list of tokens. If skip_unk is ``True``, unknown indices will be
        skipped."""
        data = self.idx_to_token(ids, skip_unk=skip_unk) \
                   if isinstance(ids, int) else \
               [self.idx_to_token(i, skip_unk=skip_unk) for i in ids]
        return data

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
        """Convert sentences of token to the sequences of the corresponding
        indices and adjust its format for Dataset. If *skip_unk* is ``True``,
        unknown tokens will be skipped. If *keep_empty* is ``False``, we'll
        remove sentences that have no data after converting.

        If save is ``True``, we'll keep the converted sentences as the Dataset
        source."""
        data = [tensor([
            i for i in s if keep_empty or i
        ], dtype=self.int_tensor_dtype) for s in [
            self.transform_tokens(s, skip_unk=skip_unk)
                for s in sentences
        ] if keep_empty or s]
        if save:
            self.data = data
        else:
            return data

    def reconstruct(self, sequences, skip_unk=False, keep_empty=False):
        """Convert sequences of indices in Dataset format to the sentences
        of the corresponding tokens. If *skip_unk* is ``True``, unknown
        indices will be skipped. If *keep_empty* is ``False``, we'll remove
        sentences that have no data after converting."""
        return [[
            t for t in s if keep_empty or t
        ] for s in [
            self.reconstruct_tokens(s[0], skip_unk=skip_unk)
                for s in sequences
        ] if keep_empty or s]

    def fit_transform(self, sentences, unk_token=None, pad_token=None,
                      extra_tokens=None, skip_unk=False, keep_empty=False,
                      save=True):
        """Just a serial execution `fit()` and `transform()` methods."""
        self.fit(sentences, unk_token=unk_token, pad_token=pad_token,
                 extra_tokens=extra_tokens)
        return self.transform(sentences, skip_unk=skip_unk,
                              keep_empty=keep_empty, save=save)

    def frame_collate(self, batch, pos, with_lens=True):
        """The method to use with junky.dataset.FrameDataset.

        :param pos: position of the data in *batch*.
        :type pos: int
        :with_lens: return lentghs of data.
        :return: depends of keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        lens = [tensor([len(x[pos]) for x in batch],
                       dtype=self.int_tensor_dtype)] if with_lens else []
        x = pad_sequence([x[pos] for x in batch],
                         batch_first=self.batch_first,
                         padding_value=self.pad)
        return (x, *lens) if lens else x

    def collate(self, batch):
        """The method to use with torch.utils.data.DataLoader

        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        lens = tensor([len(x) for x in batch], dtype=self.int_tensor_dtype)
        x = pad_sequence(batch, batch_first=self.batch_first,
                         padding_value=self.pad)
        return x, lens
