# -*- coding: utf-8 -*-
# junky lib: dataset.DummyDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset with constant output.
"""
from junky.dataset.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    """
    torch.utils.data.Dataset with constant output.

    Args:
        output_obj: the object that will be returned with every invoke.
            Default is `None`.
        data: an array-like object that support the `len(data)` method or just
            int value that is treated as the length of that object.
    """
    def __init__(self, output_obj=None, data=None):
        super().__init__()
        delattr(self, 'data')
        self.size = 0
        self.value = output_obj
        if data:
            self.transform(data, save=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.value

    def _pull_data(self):
        data = (self.size, self.value)
        self.data, self.value = 0, None
        return data

    def _push_data(self, data):
        self.size, self.value = data

    def transform(self, data, save=True, append=False):
        """Treats the length of *data* as the size of the internal data array.
        If *data* is of `int` type, just keeps that value as the size.

        If *save* is ``True``, we'll keep the size as the size of the Dataset
        source.

        If *append* is ``True``, we'll increase the size of the Dataset source
        by the size of *data*."""
        size = len(data)
        if save:
            if append:
                self.size += size
            else:
                self.size = size
        else:
            return [self.value] * size
