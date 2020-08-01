# -*- coding: utf-8 -*-
# junky lib: dataset.TokenDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for token-level input.
"""
from junky import CPU, make_token_dict
from junky.dataset.base_dataset import BaseDataset
from torch import Tensor, int64, tensor


class LabelDataset(BaseDataset):
    """
    torch.utils.data.Dataset for token-level input.

    Args:
        sentences: sequences of tokens: list([list([str])]).
        unk_token: add a token for tokens that are not present in the
            internal dict: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        int_tensor_dtype: dtype for int tensors: torch.dtype.
        transform: if ``True``, transform and save `sentences`.
        skip_unk, keep_empty: params for the `.transform()` method.
    """
    def __init__(self, labels, unk_token=None, extra_tokens=None, 
                 int_tensor_dtype=int64, transform=False, skip_unk=False,
                 keep_empty=False):
        super().__init__()
        self.int_tensor_dtype = int_tensor_dtype
        self.fit(labels, unk_token=unk_token, extra_tokens=extra_tokens)
        if transform:
            self.transform(labels, skip_unk=skip_unk, keep_empty=keep_empty,
                           save=True)

    def fit(self, labels, unk_token=None, extra_tokens=None):
        """Recreate the internal dict.

        :param sentences: sequences of tokens.
        :type sentences: list([list([str])])
        :param unk_token: add a token for tokens that are not present in the
            dict.
        :type unk_token: str
        :param extra_tokens: add tokens for any other purposes.
        :type extra_tokens: list([str])
        """
        if unk_token:
            extra_tokens = (extra_tokens if extra_tokens else []) \
                         + [unk_token]
        self.transform_dict, _, extra = \
            make_token_dict([labels], extra_tokens=extra_tokens)
        self.unk = extra[-1] if unk_token else None
        self.reconstruct_dict = {y: x for x, y in self.transform_dict.items()}

    def label_to_idx(self, label, skip_unk=False):
        """Convert a token to its index. If the token is not present in the
        internal dict, return index of unk token or None if it's not
        defined."""
        return self.transform_dict[label] \
                   if label in self.transform_dict else \
               self.unk if not skip_unk and self.unk is not None else \
               None

    def idx_to_label(self, idx, skip_unk=False):
        """Convert an index to the corresponding token. If the index is not
        present in the internal dict, return unk token or empty string if it's
        not defined or *skip_unk* is ``True``. If *skip_pad* is ``True``,
        padding index will be replaced to empty string, too."""
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        return self.reconstruct_dict[idx] \
                   if idx in self.reconstruct_dict else \
               '' if skip_unk or self.unk is None else \
               self.reconstruct_dict[self.unk]

    def transform(self, labels, skip_unk=False, keep_empty=False,
                  save=True, append=False):
        """Convert *sentences* of tokens to the sequences of the corresponding
        indices and adjust their format for Dataset. If *skip_unk* is
        ``True``, unknown tokens will be skipped. If *keep_empty* is
        ``False``, we'll remove sentences that have no data after converting.

        If *save* is ``True``, we'll keep the converted sentences as the
        Dataset source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""
        data = tensor([i for i in (self.label_to_idx(l, skip_unk=skip_unk)
                                       for l in labels)
                           if keep_empty or i is not None],
                      dtype=self.int_tensor_dtype)
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def reconstruct(self, ids, skip_unk=False, keep_empty=False):
        """Convert *sequences* of indices in Dataset format to the sentences
        of the corresponding tokens. If *skip_unk* is ``True``, unknown
        indices will be skipped. If *skip_pad* is ``True``, padding will be
        removed. If *keep_empty* is ``False``, we'll remove sentences that
        have no data after converting."""
        return [l for l in (self.idx_to_label(i, skip_unk=skip_unk)
                                for i in ids)
                    if keep_empty or l]

    def fit_transform(self, labels, unk_token=None, extra_tokens=None,
                      skip_unk=False, keep_empty=False, save=True):
        """Just a serial execution `.fit()` and `.transform()` methods."""
        self.fit(labels, unk_token=unk_token, extra_tokens=extra_tokens)
        return self.transform(labels, skip_unk=skip_unk,
                              keep_empty=keep_empty, save=save)

    def _collate(self, batch):
        return tensor(batch, dtype=self.int_tensor_dtype)
