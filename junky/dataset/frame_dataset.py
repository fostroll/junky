# -*- coding: utf-8 -*-
# junky lib: dataset.FrameDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
A frame for use several torch.utils.data.Dataset together.
"""
from collections import OrderedDict
from junky.dataset.base_dataset import BaseDataset


class FrameDataset(BaseDataset):
    """
    A frame for use several objects of `junky.dataset.Dataset` conjointly.
    All the datasets must have the data of equal length.
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

    def _pull_data(self):
        data = {}
        for name, ds in self.datasets.items():
            data[name] = ds[0]._pull_data()
        return data

    def _push_data(self, data):
        for name, d in data.items():
            self.datasets[name][0]._push_data(d)

    def _pull_xtrn(self):
        xtrn = {}
        for name, ds in self.datasets.items():
            xtrn[name] = ds[0]._pull_xtrn()
        return xtrn

    def _push_xtrn(self, xtrn):
        for name, x in xtrn.items():
            self.datasets[name][0]._push_xtrn(x)

    def add(self, name, dataset, **collate_kwargs):
        """Add *dataset* with a specified *name*.

        :param **collate_kwargs: keyword arguments for the *dataset*'s
            `._frame_collate()` method.
        """
        assert name not in self.datasets, \
               "ERROR: dataset '{}' was already added".format(name)
        assert len(dataset) > 0, "ERROR: can't add empty dataset"
        num_pos = len(dataset[0]) if isinstance(dataset[0], tuple) else 1
        self.datasets[name] = dataset, num_pos, collate_kwargs

    def remove(self, name):
        """Remove dataset with a specified *name*."""
        del self.datasets[name]

    def get(self, name):
        """Get dataset with a specified *name*.

        :return: dataset, collate_kwargs
        """
        return self.datasets[name][0], self.datasets[name][2]

    def get_dataset(self, name):
        """Get dataset with a specified *name*.

        :return: dataset
        """
        return self.datasets[name][0]

    def list(self):
        """Get names of the nested datasets in order of their addition."""
        return tuple(self.datasets.keys())

    def to(self, *args, **kwargs):
        """Invoke the `.to()` method for all object of `torch.Tensor` or
        `torch.nn.Module` type."""
        [x[0].to(*args, **kwargs) for x in self.datasets.values()]

    def transform(self, sentences, names=None, save=True, append=False,
                  part_kwargs=None, **kwargs):
        """Invoke `.transform()` methods for nested `Dataset` objects.

        *names* is a list of datasets `.transform()` methods of which will be
        called.

        *save*, *append* and **kwargs will be transfered to any nested
        `.transform()` methods.

        *part_kwargs* is a `dict` of format: {<name>: kwargs, ...}, where one
        can specify separate keyword args for `.transform()` metod of certain
        nested `Dataset` objects.

        If *save* is ``False``, we'll return the stacked result of objects'
        returns."""
        if isinstance(names, str):
            names = [names]
        data = tuple(
            y[0].transform(sentences, save=save, append=append, **kwargs,
                           **(part_kwargs[x]
                                  if part_kwargs and x in part_kwargs else
                              {}))
                for x, y in self.datasets.items()
                    if names is None
                    or x in names
        )
        if not save:
            return data

    def _frame_collate(self, batch, pos, names=None):
        """The method to use with junky.dataset.FrameDataset.

        :param pos: position of the data in *batch*.
        :type pos: int
        """
        if isinstance(names, str):
            names = [names]
        res = []
        for name, (ds, num_pos, kwargs) in self.datasets.items():
            if names is None or name in names:
                res_ = ds._frame_collate(batch, pos, **kwargs)
                res += res_ if isinstance(res_, tuple) else [res_]
                pos += num_pos
        return tuple(res)

    def _collate(self, batch, names=None):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`. It concatenates outputs of the added datasets
        in order of addition. All the dataset must have the method
        `.frame_collate(batch, pos)`, where *pos* is the first position of the
        corresponding dataset's data in the batch."""
        return self._frame_collate(batch, 0, names=names)
