# -*- coding: utf-8 -*-
# junky lib: dataset.LabelDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for label-level input.
"""
from junky import CPU, make_token_dict
from junky.dataset.base_dataset import BaseDataset
from torch import Tensor, int64, stack, tensor, zeros


class LabelDataset(BaseDataset):
    """
    torch.utils.data.Dataset for label-level input.

    Args:
        labels: list([str])|list[list([str])]).
        unk_label: add a value for labels that are not present in the internal
            dict: str.
        extra_labels: add label values for any other purposes: list([str]).
        tensor_dtype: dtype for tensors' data: torch.dtype. Default is torch.int64
        transform: if `True`, transform and save `labels`.
        skip_unk, keep_empty: params for the `.transform()` method.
    """
    def __init__(self, labels, unk_label=None, extra_labels=None, 
                 tensor_dtype=int64, transform=False, skip_unk=False,
                 keep_empty=False):
        super().__init__()
        self.tensor_dtype = tensor_dtype
        self.fit(labels, unk_label=unk_label, extra_labels=extra_labels)
        if transform:
            self.transform(labels, skip_unk=skip_unk, keep_empty=keep_empty,
                           save=True)

    def fit(self, labels, unk_label=None, extra_labels=None):
        """Recreates the internal dict.

        :param labels: sequences of label values.
        :type labels: list([str])|list[list([str])]).
        :param unk_label: add a value for labels that are not present in the
            dict.
        :type unk_label: str
        :param extra_labels: add label values for any other purposes.
        :type extra_labels: list([str])
        """
        if unk_label:
            extra_labels = (extra_labels if extra_labels else []) \
                         + [unk_label]
        self.transform_dict, _, extra = make_token_dict(
             labels if labels and (isinstance(labels[0], list)
                                or isinstance(labels[0], tuple)) else
             [labels],
             extra_tokens=extra_labels
        )
        self.unk = extra[-1] if unk_label else None
        self.reconstruct_dict = {y: x for x, y in self.transform_dict.items()}

    def label_to_idx(self, label, skip_unk=False):
        """Converts a label value to its index. If the value is not present in
        the internal dict, return index of unk label or None if it's not
        defined."""
        res = None
        if isinstance(label, list) or isinstance(label, tuple):
            ids = [self.label_to_idx(x, skip_unk=skip_unk) for x in label]
            res = zeros((len(self.transform_dict),), dtype=self.tensor_dtype)
            res[[ids]] = 1
        else:
            res = tensor(self.transform_dict[label],
                         dtype=self.tensor_dtype) \
                      if label in self.transform_dict else \
                  self.unk if not skip_unk and self.unk is not None else \
                  None
        return res

    def idx_to_label(self, idx, skip_unk=False):
        """Converts an index to the corresponding label value. If the index is
        not present in the internal dict, return unk label or empty string if
        it's not defined or *skip_unk* is `True`."""
        res = None
        if isinstance(idx, list) or isinstance(idx, tuple):
            idx = tensor(idx)
        if isinstance(idx, Tensor):
            try:
                dim = len(idx.shape)
                assert dim <= 1
                if dim == 0:
                    idx = idx.item()
                if dim == 1:
                    assert not any((idx != 0) * (idx != 1))
                    res = [self.idx_to_label(x.item(), skip_unk=skip_unk)
                               for x in (idx == 1).nonzero(as_tuple=True)[0]]
            except AssertionError:
                raise ValueError(
                    'ERROR: Only scalars and multi-hot vectors are allowed.'
                )
        if res is None:
            res = self.reconstruct_dict[idx] \
                      if idx in self.reconstruct_dict else \
                  '' if skip_unk or self.unk is None else \
                  self.reconstruct_dict[self.unk]
        return res

    def transform(self, labels, skip_unk=False, keep_empty=False,
                  save=True, append=False):
        """Converts *labels* of str type to the sequences of the corresponding
        indices and adjust their format for Dataset. If *skip_unk* is
        `True`, unknown labels will be skipped. If *keep_empty* is `False`,
        we'll remove rows that have no data after converting.

        Corresponding indices are represented as is (int numbers) if *labels*
        are of list([str]) type. Elsewise, if *labels* are of list of
        list([str]) type (each row may contain several label values), the
        indices are represented as multi-hot vectors.

        The type of indices representation is exactly the tensor_dtype
        specified in constructor.

        If *save* is `True`, we'll keep the converted labels as the
        Dataset source.

        If *append* is `True`, we'll append the converted labels to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        `True`."""
        data = [i for i in (self.label_to_idx(l, skip_unk=skip_unk)
                                for l in labels)
                  if keep_empty or i is not None]
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def reconstruct(self, ids, skip_unk=False, keep_empty=False):
        """Converts sequences of indices in Dataset format to the rows of the
        corresponding label values.
        
        If *skip_unk* is `True`, unknown indices will be skipped.
        If *keep_empty* is `False`, we'll remove labels that are empty after
        converting."""
        return [l for l in (self.idx_to_label(i, skip_unk=skip_unk)
                                for i in ids)
                  if keep_empty or l]

    def fit_transform(self, labels, unk_label=None, extra_labels=None,
                      skip_unk=False, keep_empty=False, save=True):
        """Just a serial execution `.fit()` and `.transform()` methods."""
        self.fit(labels, unk_label=unk_label, extra_labels=extra_labels)
        return self.transform(labels, skip_unk=skip_unk,
                              keep_empty=keep_empty, save=save)

    def _collate(self, batch):
        #return tensor(batch, dtype=self.tensor_dtype)
        return stack(batch, dim=0)
