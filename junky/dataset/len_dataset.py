# -*- coding: utf-8 -*-
# junky lib: dataset.LenDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset with lengths of input data
rows as output.
"""
from junky.dataset.base_dataset import BaseDataset


class LenDataset(BaseDataset):
    """
    torch.utils.data.Dataset with lengths of input data rows as output.

    Args:
        data: list([list([...])]).
    """
    def __init__(self, data=None):
        super().__init__()
        if data:
            self.transform(data, save=True)

    def transform(self, data, save=True, append=False):
        """Store lengths of *data* rows as the internal data array.

        If *save* is ``True``, we'll keep the converted sentences as the
        Dataset source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""
        data = [len(x) for x in data]
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data
