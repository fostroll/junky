# -*- coding: utf-8 -*-
# junky lib: WordDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for word-level input.
"""
from junky import get_rand_vector, pad_sequences_with_tensor
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class WordDataset(Dataset):
    """
    `torch.utils.data.Dataset` for word-level input.

    Args:
        emb_model: dict or any other object that allow the syntax
            `vector = emb_model[word]` and `if word in emb_model:`
        unk_token: add a token for words that are not present in the dict:
            str.
        unk_vec_norm: 
        pad_token: add a token for padding: str.
        extra_tokens: add tokens for any other purposes: list([str]).
        sentences: sequences of words: list([list([str])]). If not ``None``,
            they will be transformed and saved.
        skip_unk, keep_empty: params for the `transform()` method.
        batch_first: if ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
    """
    def __init__(self, emb_model, vec_size, vec_dtype=float,
                 unk_token=None, unk_vec_norm=1e-2,
                 pad_token=None, pad_vec_norm=0.,
                 extra_tokens=None, extra_vec_norm=1e-2,
                 sentences=None, skip_unk=False, keep_empty=False,
                 batch_first=False):
        super().__init__()
        self.emb_model = emb_model
        self.extra_model = {
            t: get_rand_vector((vec_size,), extra_vec_norm)
                for t in extra_tokens
        } if extra_tokens else \
        {}
        if unk_token:
            self.unk = self.extra_model[unk_token] = \
                get_rand_vector((vec_size,), unk_vec_norm)
        else:
            self.unk = None
        if pad_token:
            self.pad = self.extra_model[pad_token] = \
                get_rand_vector((vec_size,), pad_vec_norm)
            self.pad_tensor = tensor(self.pad)
        else:
            elf.pad = None
        self.batch_first = batch_first
        if sentences:
            self.transform(sentences, skip_unk=skip_unk,
                           keep_empty=keep_empty, save=True)
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def word_to_vec(self, word, skip_unk=True):
        """Convert a token to its vector. If the token is not present in the
        model, return vector of unk token or None if it's not defined."""
        return self.extra_model[word] if word in self.extra_model else \
               self.emb_model[word] if word in self.emb_model else \
               self.unk if not skip_unk and self.unk else \
               None

    def transform_words(self, words, skip_unk=False):
        """Convert a token or a list of words to the corresponding
        vector|list of vectors. If skip_unk is ``True``, unknown words will be
        skipped."""
        return self.word_to_vec(words, skip_unk=skip_unk) \
                   if isinstance(words, str) else \
               [self.word_to_vec(w, skip_unk=skip_unk) for w in words]

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
        """Convert sentences of words to the sentences of the corresponding
        vectors. If *skip_unk* is ``True``, unknown words will be skipped.
        If *keep_empty* is ``False``, we'll remove sentences that have no data
        after converting.

        If save is ``True``, we'll keep the converted sentences as the
        `Dataset` source."""
        data = [tensor([
            v for v in s if keep_empty or v is not None
        ]) for s in [
            self.transform_words(s, skip_unk=skip_unk)
                for s in sentences
        ] if keep_empty or s]
        if save:
            self.data = data
        else:
            return data

    def frame_pad_collate(self, batch, idx, with_lens=True):
        """The method to use with `junky.dataset.FrameDataset`.

        :param idx: index of the data in *batch*.
        :type idx: int
        :with_lens: return lentghs of data.
        :return: depends of keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        lens = [tensor([len(x[idx]) for x in batch])] if with_lens else []
        x = pad_sequences_with_tensor([x[idx] for x in batch],
                                      batch_first=True,
                                      padding_tensor=self.pad_tensor)
        return x, *lens

    def pad_collate(self, batch):
        """The method to use with `DataLoader`.

        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        lens = tensor([len(x) for x in batch])
        x = pad_sequences_with_tensor(batch, batch_first=True,
                                      padding_tensor=self.pad_tensor)
        return x, lens

    def get_loader(self, batch_size=32, shuffle=False, num_workers=0,
                   **kwargs):
        """Get `DataLoader` for this class. All params are the params of
        `DataLoader`. Only *dataset* and *pad_collate* can't be changed."""
        return DataLoader(self, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.pad_collate, **kwargs)
