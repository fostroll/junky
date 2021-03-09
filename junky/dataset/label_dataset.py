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
from torch import Tensor, float32, int64, stack, tensor, zeros


class LabelDataset(BaseDataset):
    """
    torch.utils.data.Dataset for label-level input.

    Args:
        labels: list([str])|list[list([str])]).
        unk_label: add a value for labels that are not present in the internal
            dict: str.
        extra_labels: add label values for any other purposes: list([str]).
        tensor_dtype: dtype for tensors' data: torch.dtype. Default depends on
            labels: for just list it is torch.int64 wereas for list of lists
            it is torch.float32.
        transform: if ``True``, transform and save `labels`.
        skip_unk, keep_empty: params for the `.transform()` method.
    """
    def __init__(self, labels, unk_label=None, extra_labels=None, 
                 tensor_dtype=None, transform=False, skip_unk=False,
                 keep_empty=False):
        super().__init__()
        self.tensor_dtype = tensor_dtype
        self.fit(labels, unk_label=unk_label, extra_labels=extra_labels)
        if transform:
            self.transform(labels, skip_unk=skip_unk, keep_empty=keep_empty,
                           save=True)

    def fit(self, labels, unk_label=None, extra_labels=None):
        """Recreate the internal dict.

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
        """Convert a label value to its index. If the value is not present in
        the internal dict, return index of unk label or None if it's not
        defined."""
        return self.transform_dict[label] \
                   if label in self.transform_dict else \
               self.unk if not skip_unk and self.unk is not None else \
               None

    def idx_to_label(self, idx, skip_unk=False):
        """Convert an index to the corresponding label value. If the index is
        not present in the internal dict, return unk label or empty string if
        it's not defined or *skip_unk* is ``True``. If *skip_pad* is ``True``,
        padding index will be replaced to empty string, too."""
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        return self.reconstruct_dict[idx] \
                   if idx in self.reconstruct_dict else \
               '' if skip_unk or self.unk is None else \
               self.reconstruct_dict[self.unk]

    def transform(self, labels, skip_unk=False, keep_empty=False,
                  save=True, append=False):
        """Convert *labels* of str type to the sequences of the corresponding
        indices and adjust their format for Dataset. If *skip_unk* is
        ``True``, unknown labels will be skipped. If *keep_empty* is
        ``False``, we'll remove rows that have no data after converting.

        Corresponding indices are represented as is (int numbers) if *labels*
        are of list([str]) type. Elsewise, if *labels* are of list of
        list([str]) type (each row may contain several label values), the
        indices are represented as multi-hot vectors.

        The type of indices representation is exactly the tensor_dtype
        specified in constructor. If ``None`` specified (default), torch.int64
        is used for singleton-type labels and torch.float32 in the multi-hot
        case.

        If *save* is ``True``, we'll keep the converted labels as the
        Dataset source.

        If *append* is ``True``, we'll append the converted labels to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""

        if labels and (isinstance(labels[0], list)
                    or isinstance(labels[0], tuple)):
            if not self.tensor_dtype:
                self.tensor_dtype = float32
            data = []
            for labs in labels:
                d = zeros((len(self.transform_dict),),
                          dtype=self.tensor_dtype)
                for i in (self.label_to_idx(l, skip_unk=skip_unk)
                              for l in labs):
                    d[i] = 1
                data.append(d)
        else:
            if not self.tensor_dtype:
                self.tensor_dtype = int64
            data = [tensor(i, dtype=self.tensor_dtype)
                        for i in (self.label_to_idx(l, skip_unk=skip_unk)
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
        """Convert *sequences* of indices in Dataset format to the rows of the
        corresponding label values. If *skip_unk* is ``True``, unknown indices
        will be skipped. If *skip_pad* is ``True``, padding will be removed.
        If *keep_empty* is ``False``, we'll remove labels that are empty after
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
