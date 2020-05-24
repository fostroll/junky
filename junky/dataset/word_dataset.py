# -*- coding: utf-8 -*-
# junky lib: WordDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides torch.utils.data.Dataset for word-level input.
"""
from junky import get_rand_vector
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class WordDataset(Dataset):
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
    def __init__(self, emb_model, vec_size, vec_dtype=float,
                 instr_vec_norm=1e-2, unk_token=None, unk_vec_norm=1e-2,
                 pad_token=None, pad_vec_norm=0., extra_tokens=None,
                 extra_vec_norm=1e-2, batch_first=False):
        super().__init__()
        self.extra_model = {}
        self.emb_model = emb_model
        if extra_tokens:
            for token in extra_tokens:
                self.extra_model[token] = \
                    tensor(get_rand_vector((vec_size,), extra_vec_norm))
        if unk_token:
            self.unk = self.extra_model[unk_token] = \
                tensor(get_rand_vector((vec_size,), unk_vec_norm))
        else:
            self.unk = None
        if pad_token:
            self.pad = self.extra_model[pad_token] = \
                tensor(get_rand_vector((vec_size,), pad_vec_norm))
        else:
            elf.pad = None
        self.batch_first = batch_first
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def word_to_vec(self, word, skip_unk=True):
        """Convert a token to its vector. If the token is not present in the
        model, return vector of unk token or None if it's not defined."""
        return vector(self.emb_model[word]) if word in self.emb_model else \
               self.unk if not skip_unk and self.unk else \
               None

    def transform_words(self, words, skip_unk=False):
        """Convert a token or a list of words to the corresponding
        vector|list of vectors. If skip_unk is ``True``, unknown words will be
        skipped."""
        return tensor(self.word_to_vec(words, skip_unk=skip_unk)) \
                   if isinstance(words, str) else \
               tensor(self.word_to_vec(w, skip_unk=skip_unk) for w in words)

    def transform(self, sentences, skip_unk=False, keep_empty=False,
                  save=True):
        """Convert sentences of words to the sentences of the corresponding
        vectors. If *skip_unk* is ``True``, unknown words will be skipped.
        If *keep_empty* is ``False``, we'll remove sentences that have no data
        after converting.

        If save is ``True``, we'll keep the converted sentences as the Dataset
        source."""
        data = [(tensor(
            v for v in s if keep_empty or v is not None
        ),) for s in [
            self.transform_words(s, skip_unk=skip_unk)
                for s in sentences
        ] if keep_empty or s]
        if save:
            self.data = data
        else:
            return data

    def pad_collate(self, batch):
        """The method to use with torch.utils.data.DataLoader
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
    def pad_collate(batch):
        lens = tensor([len(x[0]) for x in batch])
        x = pad_sequences_with_tensor(
            [x[0] for x in batch], batch_first=True,
            padding_tensor=wd_test.pad
        )
        return x, lens
