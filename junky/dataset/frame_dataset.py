# -*- coding: utf-8 -*-
# junky lib: FrameDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
A frame for use several torch.utils.data.Dataset together.
"""
from collections import OrderedDict
from junky.dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset


class FrameDataset(BaseDataset):
    """
    A frame for use several objects of `junky.dataset.*Dataset` conjointly.
    All datasets must have the data of equal length.
    """
    def __init__(self):
        super().__init__()
        delattr(self, 'data')
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
        """Remove dataset with specified *name*."""
        del self.datasets[name]

    def get(self, name):
        """Get dataset with specified *name*.

        :return: dataset, collate_kwargs
        """
        return self.datasets[name][0], self.datasets[name][2]

    def list(self):
        """Print names of the added datasets in order of addition."""
        return tuple(self.datasets.keys())

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
        """Invoke `.transform(sentences, skip_unk, keep_empty, save)` method
        for all nested `Dataset` objects.

        If save is ``False``, we'll return the stacked result of objects'
        return."""
        data = tuple(x[0].transform(sents, skip_unk=skip_unk,
                                    keep_empty=keep_empty, save=save)
                         for x in self.datasets.values())
        if not save:
            return tuple(data)

    def collate(self, batch):
        """The method to use with torch.utils.data.DataLoader. It concatenates
        outputs of the added datasets in order of addition. All the dataset
        must have the method `.frame_collate(batch, pos, **kwargs)`, where
        *pos* is the first position of the corresponding dataset's data in the
        batch.
        """
        res, pos = [], 0
        for ds, num_pos, kwargs in self.datasets.values():
            res_ = ds.frame_collate(batch, pos, **kwargs)
            res += res_ if isinstance(res_, tuple) else [res_]
            pos += num_pos
        return tuple(res)
