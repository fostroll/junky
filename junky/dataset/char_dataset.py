# -*- coding: utf-8 -*-
# junky lib: dataset.CharDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for character-level input.
"""
from junky import CPU, make_alphabet, pad_array_torch
from junky.dataset.base_dataset import BaseDataset
from torch import Tensor, int64, tensor
from torch.nn.utils.rnn import pad_sequence


class CharDataset(BaseDataset):
    """
    torch.utils.data.Dataset for character-level input.

    Args:
        sentences: sequences of tokens: list([list([str])]).
        unk_token: add a token for characters that are not present in the
            internal dict: str.
        pad_token: add a token for padding: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        allowed_chars: if not None, all charactes not from *allowed_chars*
            will be removed: str|list([str]).
        exclude_chars: if not None, all charactes from *exclude_chars* will
            be removed: str|list([str]).
        int_tensor_dtype: dtype for int tensors: torch.dtype.
        transform: if ``True``, transform and save `sentences`.
        skip_unk, keep_empty: params for the `.transform()` method.
    """
    def __init__(self, sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, allowed_chars=None, exclude_chars=None,
                 int_tensor_dtype=int64, transform=False, skip_unk=False,
                 keep_empty=False):
        super().__init__()
        self.int_tensor_dtype = int_tensor_dtype
        self.fit(sentences, unk_token=unk_token, pad_token=pad_token,
                 extra_tokens=extra_tokens, allowed_chars=allowed_chars,
                 exclude_chars=exclude_chars)
        if transform:
            self.transform(sentences, skip_unk=skip_unk,
                           keep_empty=keep_empty, save=True)

    def fit(self, sentences, unk_token=None, pad_token=None,
            extra_tokens=None, allowed_chars=None, exclude_chars=None):
        """Recreate the internal dict.

        :param sentences: sequences of tokens.
        :type sentences: list([list([str])])
        :param unk_token: add a token for characters that are not present in
            the dict.
        :type unk_token: str
        :param pad_token: add a token for padding.
        :type pad_token: str
        :param extra_tokens: add tokens for any other purposes.
        :type extra_tokens: list([str])
        :param allowed_chars: if not None, all charactes not from
            *allowed_chars* will be removed.
        :type allowed_chars: str|list([str])
        :param exclude_chars: if not None, all charactes from *exclude_chars*
            will be removed.
        :type exclude_chars: str|list([str])
        """
        if unk_token:
            extra_tokens = (extra_tokens if extra_tokens else []) \
                         + [unk_token]
        self.transform_dict, self.pad, extra = \
            make_alphabet(sentences,
                          pad_char=pad_token,
                          extra_chars=extra_tokens,
                          allowed_chars=allowed_chars,
                          exclude_chars=exclude_chars)
        self.unk = extra[-1] if unk_token else None
        self.reconstruct_dict = {y: x for x, y in self.transform_dict.items()}

    def char_to_idx(self, char, skip_unk=False):
        """Convert a character to its index. If the character is not present
        in the internal dict, return index of unk token or None if it's not
        defined or *skip_unk* is `True`."""
        return self.transform_dict[char] \
                   if char in self.transform_dict else \
               self.unk if not skip_unk and self.unk is not None else \
               None

    def idx_to_char(self, idx, skip_unk=False, skip_pad=True):
        """Convert an index to the corresponding character. If the index is
        not present in the internal dict, return unk token or empty string if
        it's not defined or *skip_unk* is ``True``. If *skip_pad* is ``True``,
        padding index will be replaced to empty string, too."""
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        return '' if skip_pad and idx == self.pad else (
            self.reconstruct_dict[idx] if idx in self.reconstruct_dict else
            '' if skip_unk or self.unk is None else
            self.reconstruct_dict[self.unk]
        )

    def token_to_ids(self, token, skip_unk=False):
        """Convert a token or a `list` of characters to the list of indices of
        its chars. If some characters are not present in the internal dict,
        we'll use the index of unk token for them, or empty strings if it's
        not defined or *skip_unk* is ``True``. If *skip_pad* is ``True``,
        padding indices will be replaced to empty string, too.

        :type token: str|list([char])
        :rtype: list([int])
        """
        return [i for i in [
            self.char_to_idx(c, skip_unk=skip_unk) for c in token
        ] if not skip_unk or i is not None]

    def ids_to_token(self, ids, skip_unk=False, skip_pad=True, aslist=False):
        """Convert a list of indices to the corresponding token or a list of
        characters. If some indices are not present in the internal dict,
        we'll use unk token for them, or None if it's not defined.

        :param aslist: if ``True``, we want list of characters instead of
            token as the result.
        """
        data = [c for c in [
            self.idx_to_char(i, skip_unk=skip_unk) for i in ids
        ] if not skip_unk or c]
        return data if aslist else ''.join(data)

    def transform_tokens(self, tokens, skip_unk=False):
        """Convert a token or a sequence of tokens to the corresponding list
        or a sequence of lists of indices. If skip_unk is ``True``, unknown
        tokens will be skipped."""
        return self.token_to_ids(tokens, skip_unk=skip_unk) \
                   if isinstance(tokens, str) else \
               [self.token_to_ids(t, skip_unk=skip_unk) for t in tokens]

    def reconstruct_tokens(self, ids, skip_unk=False, skip_pad=True,
                           aslist=False):
        """Convert a list of indices or a sequence of lists of indices to the
        corresponding token|sequence of tokens. If skip_unk is ``True``,
        unknown indices will be skipped (or replaced to empty strings, if
        *aslist* is ``True``). If *skip_pad* is ``True``, padding indices
        also will be removed or replaced to empty strings.

        :param aslist: if ``True``, we want lists of characters instead of
            tokens in the result.
        """
        return self.ids_to_token(ids, skip_unk=skip_unk, skip_pad=skip_pad,
                                 aslist=aslist) \
                   if ids and isinstance(ids[0], int) else \
               [self.ids_to_token(i, skip_unk=skip_unk, skip_pad=skip_pad,
                                  aslist=aslist)
                    for i in ids]

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True, append=False):
        """Convert *sentences* of tokens to the sequences of the lists of the
        indices corresponding to token's chars and adjust their format for
        Dataset. If *skip_unk* is ``True``, unknown chars will be skipped.
        If *keep_empty* is ``False``, we'll remove tokens and sentences that
        have no data after converting.

        If save is ``True``, we'll keep the converted sentences as the Dataset
        source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""
        data = [[
            tensor(i, dtype=self.int_tensor_dtype)
                for i in s if keep_empty or i is not None
        ] for s in [
            self.transform_tokens(s, skip_unk=skip_unk)
                for s in sentences
        ] if keep_empty or s]
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def reconstruct(self, sequences, skip_unk=False, skip_pad=True,
                    keep_empty=False, aslist=False):
        """Convert *sequences* of the lists of the indices in Dataset format
        to the sentences of the corresponding tokens. If *skip_unk* is
        ``True``, unknown indices will be skipped. If *skip_pad* is ``True``,
        padding will be removed. If *keep_empty* is ``False``, we'll remove
        sentences that have no data after converting.

        :param aslist: if ``True``, we want lists of characters instead of
            tokens in the result.
        """
        return [[
            [c for c in t if keep_empty or c is not None] if aslist else t
                for t in s if keep_empty or t
        ] for s in [
            self.reconstruct_tokens(s, skip_unk=skip_unk, skip_pad=skip_pad,
                                    aslist=aslist)
                for s in sequences
        ] if keep_empty or s]

    def fit_transform(self, sentences, unk_token=None, pad_token=None,
                      extra_tokens=None, skip_unk=False, keep_empty=False,
                      save=True):
        """Just a serial execution `.fit()` and `.transform()` methods."""
        self.fit(sentences, unk_token=unk_token, pad_token=pad_token,
                 extra_tokens=extra_tokens)
        return self.transform(sentences, skip_unk=skip_unk,
                              keep_empty=keep_empty, save=save)

    def _collate(self, batch, with_lens=True, with_token_lens=True):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`.

        :with_lens: return lengths of data.
        :with_token_lens: return lengths of tokens of the data.
        :return: depends on keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor,
                      token_lens:list([torch.tensor]))
        """
        assert self.pad is not None, \
               ('ERROR: pad_token must be defined if you want to use {} in '
                'DataLoader').format(self.__class__.__name__)
        device = CPU
        for x in batch:
            if x:
                if x[0].is_cuda:
                    device = x[0].get_device()
                break
        lens = [tensor([len(x) for x in batch], device=device,
                       dtype=self.int_tensor_dtype)] if with_lens else []
        if with_token_lens:
            lens.append([tensor([len(x) for x in x], device=device,
                                dtype=self.int_tensor_dtype) for x in batch])
        batch = self._to(batch, CPU)
        x = pad_array_torch(batch, padding_value=self.pad,
                            device=device, dtype=self.int_tensor_dtype)
        return (x, *lens) if lens else x
