# -*- coding: utf-8 -*-
# junky lib: dataset.WordCatDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Concatenate outputs of datasets the sources of which are of `torch.Tensor`
type.
"""
from itertools import accumulate
from junky.dataset import FrameDataset
import torch


class WordCatDataset(FrameDataset):
    """
    Concatenate outputs of datasets the sources of which are of `torch.Tensor`
    type.
    """
    @property
    def vec_size(self):
        return sum(x[0].vec_size for x in self.datasets) \
                   if self.datasets else \
               0

    def __getitem__(self, idx):
        """Returns a tuple of outputs of all added datasets in order of
        addition."""
        return torch.cat([x for x in self.datasets.values()
                            for x in (x[0][idx]
                                          if isinstance(x[0][idx], tuple) else
                                      [x[0][idx]])], dim=-1)

    def add(self, name, dataset):
        """Add *dataset* with a specified *name*.

        :param **collate_kwargs: keyword arguments for the *dataset*'s
            `._frame_collate()` method.
        """
        assert name not in self.datasets, \
               "ERROR: dataset '{}' was already added".format(name)
        self.datasets[name] = dataset,

    def get(self, name):
        """Get dataset with a specified *name*.

        :return: tuple(dataset,)
        """
        return self.datasets[name][0],

    def _frame_collate(self, batch, pos, with_lens=True):
        """The method to use with junky.dataset.FrameDataset.

        :param pos: position of the data in *batch*.
        :type pos: int
        :with_lens: return lengths of data.
        :return: depends on keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        vec_sizes = [x[0].vec_size for x in self.datasets.values()]
        if vec_sizes:
            vec_sizes[0] = (0, vec_sizes[0])
        vec_sizes = list(accumulate(vec_sizes, lambda x, y: (x[1], x[1] + y)))

        x = [y[0]._frame_collate([x[pos][..., z[0]:z[1]] for x in batch], 0,
                              with_lens=with_lens and i == 0)
                 for i, (y, z) in enumerate(zip(self.datasets.values(),
                                            vec_sizes))]
        if with_lens:
            lens = x[0][1]
            x[0] = x[0][0]
        else:
            lens = []
        x = torch.cat(x, dim=-1)
        return (x, *lens) if lens else x

    def _collate(self, batch):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`. It concatenates outputs of the added datasets
        in order of addition. All the dataset must have the method
        `.frame_collate(batch, pos)`, where *pos* is the first position of the
        corresponding dataset's data in the batch."""
        return self._frame_collate(batch, 0)
