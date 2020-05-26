# -*- coding: utf-8 -*-
# junky lib: BaseDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides base functionality for junky.dataset.*Dataset classes.
"""
from copy import deepcopy
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

    def _create_empty(self):
        """Create empty instance of the current class. Is invoked by
        `.clone()`. You must override this method if constructor of your class
        has positional args."""
        return self.__class__()

    def clone(self, with_data=True):
        """Clone this object. If *with_data* is ``False``, the `data` attr of
        the new object will be empty.
        """
        o = self._create_empty()
        for name, val in self.__dict__.items():
            setattr(o, name, val.clone(with_data=with_data)
                                 if isinstance(val, BaseDataset) else
                             deepcopy(val)
                                 if name != 'data' or with_data else
                             [])
        return o

    def frame_collate(self, batch, pos):
        """The stub method to use with `junky.dataset.FrameDataset`.

        :param pos: position of the data in *batch*.
        :type pos: int
        """
        return [x[pos] for x in batch]

    def collate(self, batch):
        """The stub method to use with `DataLoader`."""
        return batch

    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
        """Get `DataLoader` for this class. All params are the params of
        `DataLoader`. Only *dataset* and *collate_fn* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate, **kwargs)
