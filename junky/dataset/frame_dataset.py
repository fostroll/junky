# -*- coding: utf-8 -*-
# junky lib: FrameDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides torch.utils.data.Dataset for word-level input.
"""
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """
    torch.utils.data.Dataset for word-level input.

    Args:
        emb_model: dict or any other object that allow the syntax
            `vector = emb_model[word]` and `if word in emb_model:`
        sentences: sequences of words: list([list([str])]).
        unk_token: add a token for words that are not present in the dict:
            str.
        unk_vec_norm: 
        pad_token: add a token for padding: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        batch_first: if ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
    """
    def __init__(self, dataset, *datasets):
        super().__init__()
        len_ = len(dataset)
        for ds in datasets:
            assert len(ds) == len_,
                   'ERROR: all datasets must have equal length'
        self.datasets = (dataset,) + datasets

    def __len__(self):
        return len(self.datasets[0].data)

    def __getitem__(self, idx):
        return self.dataset[0].data[idx]

    def pad_collate(self, batch):
        """The method to use with torch.utils.data.DataLoader
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        batch_ =  [[] for _ in self.datasets]
        for x in batch:
            for y, b in zip(x, batch_):
                b.append((y,))

        res = []
        for ds, b in zip(ds, batch_):
            res_ = ds.pad_collate(b)
            res.append([r for r in res_] if isinstance(res_, tuple) else
                       [res_])

        return tuple(res)

    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
        """Get `DataLoader` for this class. All params are the params of
        `DataLoader`. Only *dataset* and *pad_collate* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.pad_collate, **kwargs)
