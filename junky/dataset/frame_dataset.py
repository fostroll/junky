# -*- coding: utf-8 -*-
# junky lib: FrameDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
A frame for use several torch.utils.data.Dataset together.
"""
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset


class FrameDataset(Dataset):
    """
    A frame for use several objects of `junky.dataset.*Dataset` conjointly.
    All datasets must have the data of equal length.
    """
    def __init__(self):
        super().__init__()
        self.datasets = OrderedDict()

    def __len__(self):
        """Returns the lentgh of the first added dataset. Note, that all
        datasets must be of equal length."""
        return len(next(iter(self.datasets.values()))[0]) \
                   if self.datasets else \
               0

    def __getitem__(self, idx):
        """Returns a tuple of outputs of all added datasets in order of
        addition."""
        return tuple(x for x in self.datasets.values()
                       for x in (x[0][idx]
                                     if isinstance(x[0][idx], tuple) else
                                 [x[0][idx]]))

    def add(self, name, dataset, **collate_kwargs):
        """Add *dataset* with specified *name*.

        :param **collate_kwargs: keyword arguments for the *dataset*'s
            `frame_collate()` method.
        """
        assert name not in self.datasets, \
               "ERROR: dataset '{}' was already added".format(name)
        assert len(dataset) > 0, "ERROR: can't add empty dataset"
        num_pos = len(dataset[0]) if isinstance(dataset[0], tuple) else 1
        self.datasets[name] = [dataset, num_pos, collate_kwargs]

    def remove(self, name):
        """Remove *dataset* with specified *name*."""
        del self.datasets[name]

    def list(self):
        """Print names of the added datasets in order of addition."""
        return tuple(self.datasets.keys())

    def collate(self, batch):
        """The method to use with torch.utils.data.DataLoader. It concatenates
        outputs of the added datasets in order of addition. All the dataset
        must have the method `.frame_collate(batch, pos, **kwargs)`, where
        *pos* is the first position of the corresponding dataset's data in the
        batch.
        """
        res, pos = [], 0
        for ds in self.datasets.values():
            res_ = ds[0].frame_collate(batch, pos, **ds[2])
            res += res_ if isinstance(res_, tuple) else [res_]
            pos += ds[1]
        return tuple(res)

    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
        """Get `DataLoader` for this class. All params are the params of
        `DataLoader`. Only *dataset* and *pad_collate* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.collate, **kwargs)
