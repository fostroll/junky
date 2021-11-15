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
import sys
from torch import Tensor, load as torch_load, save as torch_save
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    """
    Base class for junky.dataset.*Dataset classes.

    Args:
        data: any list of data to save as Dataset source.
    """
    def __init__(self, data=None):
        super().__init__()
        if data:
            self.transform(data if data else [], append=False)
        else:
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

    def _clone_or_save(self, with_data=True, file_path=None, method='torch'):
        data, o = None, None
        if not with_data:
            data = self._pull_data()
        xtrn = self._pull_xtrn()
        if file_path:
            with open(file_path, 'wb') as f:
                if method == 'pickle':
                    pickle.dump(self, f, 2)
                elif method == 'torch':
                    torch_save(self, f)
                else:
                    raise ValueError(f'ERROR: Unknown method "{method}"')
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

    def save(self, file_path, with_data=True, method='torch'):
        """Save the object to *file_path*. If *with_data* is ``False``, the
        `data` attribute of the saved object will be empty. The param *method*
        can be either 'torch' (default) or 'pickle'.
        """
        return self._clone_or_save(with_data=with_data, file_path=file_path,
                                   method=method)

    @staticmethod
    def load(file_path, xtrn=None, method=None):
        """Load object from *file_path*. You should pass the *xtrn* object
        that you received as result of the `.save()` method call for this
        object. The param *method* can be either 'torch' or 'pickle'. If the
        *method* is ``None`` (default), we detect it by trial and error."""
        with open(file_path, 'rb') as f:
            if method is None:
                try:
                    o = pickle.load(f)
                except pickle.UnpicklingError:
                    o = torch_load(f)
            elif method == 'pickle':
                o = pickle.load(f)
            elif method == 'torch':
                o = torch_load(f)
            else:
                raise ValueError(f'ERROR: Unknown method "{method}"')
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
        """Invokes `.to(*args, **kwargs)` for all the elements of the Dataset
        source that have `torch.Tensor` or `torch.nn.Model` type. All the
        params are transferred as is."""
        self.data = self._to(self.data, *args, **kwargs)

    def transform(self, data, append=False):
        """Just save any list of *data* as Dataset source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced."""
        if append:
            self.data += data
        else:
            self.data = data

    def _frame_collate(self, batch, pos, **kwargs):
        """The method to use with `junky.dataset.FrameDataset`.

        :param pos: position of the data in *batch*.
        :type pos: int
        :param **kwargs: params for _collate().
        :return: depends on keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        return self._collate([x[pos] for x in batch], **kwargs)

    def _collate(self, batch):
        """The stub method to use with `DataLoader` and
        `.transform_collate()`."""
        return batch

    def create_loader(self, batch_size=32, shuffle=False, num_workers=0,
                      **kwargs):
        """Create `DataLoader` for this object. All params are the params of
        `DataLoader`. Only *dataset* and *collate_fn* can't be changed."""
        return DataLoader(self, batch_size=batch_size or len(self),
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self._collate, **kwargs)

    def transform_collate(self, sentences, batch_size=32,
                          transform_kwargs=None, collate_kwargs=None,
                          loglevel=0):
        """Sequentially makes batches from **sentences** and call
        `.transform(batch, save=False, **transform_kwargs)` and
        `._collate(batch, **collate_kwargs)` methods for them."""
        if transform_kwargs is None:
            transform_kwargs = {}
        if collate_kwargs is None:
            collate_kwargs = {}
        batch = []
        _src = sentences
        if loglevel >= 1:
            print('Processing corpus')
            _src = tqdm(iterable=_src, mininterval=2, file=sys.stdout)
        for sentence in _src:
            batch.append(sentence)
            if len(batch) == batch_size:
                res = self.transform(batch, **transform_kwargs, save=False)
                yield self._collate(list(zip(*res)
                                        if isinstance(res, tuple) else
                                    res), **collate_kwargs)
                batch = []
        if batch:
            res = self.transform(batch, **transform_kwargs, save=False)
            yield self._collate(list(zip(*res)
                                    if isinstance(res, tuple) else
                                res), **collate_kwargs)
