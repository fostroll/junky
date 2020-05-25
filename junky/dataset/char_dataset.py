# -*- coding: utf-8 -*-
# junky lib: CharDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for character-level input.
"""
from junky import make_alphabet, pad_array_torch
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    """
    torch.utils.data.Dataset for character-level input.

    Args:
        sentences: sequences of tokens: list([list([str])]).
        unk_token: add a token for tokens that are not present in the dict:
            str.
        pad_token: add a token for padding: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        allowed_chars: if not None, all charactes not from *allowed_chars*
            will be removed: str|list([str])
        exclude_chars: if not None, all charactes from *exclude_chars* will
            be removed: str|list([str])
        transform: if ``True``, transform and save `sentences`.
        skip_unk, keep_empty: params for the `transform()` method.
        batch_first: if ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
        min_len: if specified, collate will pad sentences in `batch` that
            are shorter than `min_len`: int.
    """
    def __init__(self, sentences, unk_token=None, pad_token=None,
                 extra_tokens=None, allowed_chars=None, exclude_chars=None,
                 transform=False, skip_unk=False, keep_empty=False,
                 batch_first=False, min_len=None):
        super().__init__()
        self.batch_first = batch_first
        self.min_len = min_len
        self.fit(sentences, unk_token=unk_token, pad_token=pad_token,
                 extra_tokens=extra_tokens, allowed_chars=allowed_chars,
                 exclude_chars=exclude_chars)
        if transform:
            self.transform(sentences, skip_unk=skip_unk,
                           keep_empty=keep_empty, save=True)
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
        defined."""
        return self.transform_dict[char] \
                   if char in self.transform_dict else \
               self.unk if not skip_unk and self.unk else \
               None

    def idx_to_char(self, idx, skip_unk=False):
        """Convert an index to the corresponding character. If the index is
        not present in the internal dict, return unk token or None if it's not
        defined."""
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        return self.reconstruct_dict[idx] \
                   if idx in self.reconstruct_dict else \
               self.reconstruct_dict[self.unk] \
                   if not skip_unk and self.unk else \
               ''

    def token_to_ids(self, token, skip_unk=False):
        """Convert a token to the list of indices of its chars. If some
        characters are not present in the internal dict, we'll use the index
        of unk token for them, or None if it's not defined.

        :type token: str|list([char])
        :rtype: list([int])
        """
        return [i for i in [
            self.char_to_idx(c, skip_unk=skip_unk) for c in token
        ] if not skip_unk or i is not None]

    def ids_to_token(self, ids, skip_unk=False, aslist=False):
        """Convert a list of an indices to the list of corresponding
        characters. If some indices are not present in the internal dict,
        we'll use unk token for them, or None if it's not defined."""
        data = [c for c in [
            self.idx_to_char(i, skip_unk=skip_unk) for i in ids
        ] if not skip_unk or c]
        return data if aslist else ''.join(data)

    def transform_tokens(self, tokens, skip_unk=False):
        """Convert a token or a list of tokens to the corresponding
        index|list of indices. If skip_unk is ``True``, unknown tokens will be
        skipped."""
        return self.token_to_ids(tokens, skip_unk=skip_unk) \
                   if isinstance(tokens, str) else \
               [self.token_to_ids(t, skip_unk=skip_unk) for t in tokens]

    def reconstruct_tokens(self, ids, skip_unk=False, aslist=False):
        """Convert a list of indices or a sequence of lists of indices to the
        corresponding token|sequence of tokens. If skip_unk is ``True``,
        unknown indices will be skipped."""
        return self.ids_to_token(ids, skip_unk=skip_unk, aslist=aslist) \
                   if ids and isinstance(ids[0], int) else \
               [self.ids_to_token(i, skip_unk=skip_unk, aslist=aslist)
                    for i in ids]

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
        """Convert sentences of token to the sequences of the lists of the
        indices corresponding to token's chars and adjust its format for
        Dataset. If *skip_unk* is ``True``, unknown chars will be skipped.
        If *keep_empty* is ``False``, we'll remove tokens and sentences that
        have no data after converting.

        If save is ``True``, we'll keep the converted sentences as the Dataset
        source."""
        data = [[
            tensor(i) for i in s if keep_empty or i
        ] for s in [
            self.transform_tokens(s, skip_unk=skip_unk)
                for s in sentences
        ] if keep_empty or s]
        if save:
            self.data = data
        else:
            return data

    def reconstruct(self, sequences, skip_unk=False, keep_empty=False,
                    aslist=False):
        """Convert sequences of the lists of the indices in Dataset format to
        the sentences of the corresponding tokens. If *skip_unk* is ``True``,
        unknown indices will be skipped. If *keep_empty* is ``False``, we'll
        remove sentences that have no data after converting."""
        return [[
            t for t in s if keep_empty or t
        ] for s in [
            self.reconstruct_tokens(s, skip_unk=skip_unk, aslist=aslist)
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

    def frame_collate(self, batch, pos, with_lens=True,
                          with_token_lens=True):
        """The method to use with junky.dataset.FrameDataset.

        :param pos: position of the data in *batch*.
        :type pos: int
        :with_lens: return lentghs of data.
        :with_token_lens: return lengths of tokens of the data.
        :return: depends of keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor,
                      token_lens:list([torch.tensor]))
        """
        lens = [tensor([len(x[pos]) for x in batch])] if with_lens else []
        if with_token_lens:
            lens.append([tensor([len(x) for x in x[pos]]) for x in batch])
        if self.min_len is not None:
            batch.append(([tensor([self.pad])] * self.min_len))
        x = pad_array_torch([x[pos] for x in batch],
                            padding_value=self.pad)
        if self.min_len is not None:
            x = x[:-1]
        return (x, *lens) if lens else x

    def collate(self, batch):
        """The method to use with torch.utils.data.DataLoader

        :rtype: tuple(list([torch.tensor]), lens:torch.tensor,
                      token_lens:list([torch.tensor]))
        """
        lens = tensor([len(x) for x in batch])
        token_lens = [tensor([len(x) for x in x]) for x in batch]
        if self.min_len is not None:
            batch.append(([tensor([self.pad])] * self.min_len))
        x = pad_array_torch(batch, padding_value=self.pad)
        if self.min_len is not None:
            x = x[:-1]
        return x, lens, token_lens

    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
        """Get `DataLoader` for this class. All params are the params of
        `DataLoader`. Only *dataset* and *collate_fn* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate, **kwargs)
