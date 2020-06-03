# -*- coding: utf-8 -*-
# junky lib: dataset.BaseDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides base functionality for junky.dataset.*Dataset classes.
"""
from copy import deepcopy
import pickle
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    """
    Base class for junky.dataset.*Dataset classes.
    """
    def __init__(self):
        super().__init__()
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _pull_data(self):
        data = None
        if hasattr(self, 'data'):
            data = self.data
            self.data = []
        return data

    def _push_data(self, data):
        if hasattr(self, 'data'):
            self.data = data

    def _pull_xtrn(self):
        return None

    def _push_xtrn(self, xtrn):
        pass

    def _clone_or_save(self, with_data=True, file_path=None):
        data, o = None, None
        if not with_data:
            data = self._pull_data()
        xtrn = self._pull_xtrn()
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, 2)
        else:
            o = deepcopy(self)
        if xtrn is not None:
            self._push_xtrn(xtrn)
            if o is not None:
                o._push_xtrn(xtrn)
        if data:
            self._push_data(data)
        return o if o is not None else xtrn

    def clone(self, with_data=True):
        """Clone this object. If *with_data* is ``False``, the `data` attr of
        the new object will be empty."""
        return self._clone_or_save(with_data=with_data)

    def save(self, file_path, with_data=True):
        """Save this object to *file_path*. If *with_data* is ``False``, the
        `data` attr of the new object will be empty."""
        return self._clone_or_save(with_data=with_data, file_path=file_path)

    @staticmethod
    def load(file_path, xtrn=None):
        """Load object from *file_path*. You should pass the *xtrn* object
        that you received as result of the `.save()` method call for this
        object."""
        with open(file_path, 'rb') as f:
            o = pickle.load(f)
        if xtrn is not None:
            o._push_xtrn(xtrn)
        return o

    @classmethod
    def _to(cls, o, *args, **kwargs):
        if isinstance(o, Tensor):
            o = o.to(*args, **kwargs)
        elif isinstance(o, Module):
            o.to(*args, **kwargs)
        elif isinstance(o, list):
            for i in range(len(o)):
                o[i] = cls._to(o[i], *args, **kwargs)
        return o

    def to(self, *args, **kwargs):
        """Invoke the `.to()` method for all object of `torch.Tensor` or 
        `torch.nn.Module` type."""
        self.data = self._to(self.data, *args, **kwargs)

    def _frame_collate(self, batch, pos):
        """The stub method to use with `junky.dataset.FrameDataset`.

        :param pos: position of the data in *batch*.
        :type pos: int
        """
        return [x[pos] for x in batch]

    def _collate(self, batch):
        """The stub method to use with `DataLoader`."""
        return batch

    def create_loader(self, batch_size=32, shuffle=False, num_workers=0,
                      **kwargs):
        """Create `DataLoader` for this object. All params are the params of
        `DataLoader`. Only *dataset* and *collate_fn* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self._collate, **kwargs)
