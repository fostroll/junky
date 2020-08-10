# -*- coding: utf-8 -*-
# junky lib: dataset.WordDataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementation of torch.utils.data.Dataset for word-level input.
"""
from copy import deepcopy
#from gensim.models.keyedvectors import FastTextKeyedVectors
from junky import CPU, get_rand_vector, pad_sequences_with_tensor
from junky.dataset.base_dataset import BaseDataset
from torch import Tensor, float32, int64, tensor
from torch.nn.utils.rnn import pad_sequence


class WordDataset(BaseDataset):
    """
    `torch.utils.data.Dataset` for word-level input.

    Args:
        emb_model: dict or any other object that allow the syntax
            `vector = emb_model[word]` and `if word in emb_model:`.
        vec_size: the length of the word's vector.
        unk_token: add a token for words that are not present in the internal
            dict: str.
        unk_vec_norm: the norm of the vector for *unk_token*: float.
        pad_token: add a token for padding: str.
        pad_vec_norm: the norm of the vector for *pad_token*: float.
        extra_tokens: add tokens for any other purposes: list([str]).
        extra_vec_norm: the norm of the vectors for *extra_tokens*: float.
        float_tensor_dtype: dtype for float tensors: torch.dtype.
        int_tensor_dtype: dtype for int tensors: torch.dtype.
        sentences: sequences of words: list([list([str])]). If not ``None``,
            they will be transformed and saved.
        check_lower, skip_unk, keep_empty: params for the `.transform()`
        method.
    """
    #@property
    #def vec_size(self):
    #    return self.data[0].shape[-1] if self.data else \
    #           self.emb_model.vec_size
    #               if hasattr(self.emb_model, 'vec_size') else \
    #           self.emb_model.config.hidden_size

    def __init__(self, emb_model, vec_size,
                 unk_token=None, unk_vec_norm=1.,
                 pad_token=None, pad_vec_norm=0.,
                 extra_tokens=None, extra_vec_norm=1.,
                 float_tensor_dtype=float32, int_tensor_dtype=int64,
                 sentences=None, check_lower=True, skip_unk=False,
                 keep_empty=False):
        super().__init__()
        self.emb_model = emb_model
        self.vec_size = vec_size
        self.float_tensor_dtype = float_tensor_dtype
        self.int_tensor_dtype = int_tensor_dtype
        self.extra_model = {
            #t: emb_model[t]
            #       if t in (emb_model.vocab
            #                    if isinstance(emp_model,
            #                                  FastTextKeyedVectors) else
            #                emb_model) else
            #   get_rand_vector((vec_size,), extra_vec_norm)
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
            self.pad_tensor = tensor(self.pad, dtype=float_tensor_dtype)
        else:
            self.pad = None
            self.pad_tensor = None
        if sentences:
            self.transform(sentences, check_lower=check_lower,
                           skip_unk=skip_unk, keep_empty=keep_empty,
                           save=True)

    def _pull_xtrn(self):
        xtrn = self.emb_model
        self.emb_model = {}
        return xtrn

    def _push_xtrn(self, xtrn):
        self.emb_model = xtrn

    def word_to_vec(self, word, check_lower=True, force_lower=False,
                    skip_unk=True):
        """Convert a token to its vector. If the token is not present in the
        model, return vector of unk token or ``None`` if it's not defined.

        If *check_lower* is ``True`` (default), try to find vector of
        `word.lower()` if vector of *word* is absent.

        If *force_lower* is ``True``, convert *word* to lower case before
        looking its vectors up. Default is ``False``.

        If *skip_unk* is ``True``, unknown words will be skipped."""
        return self.extra_model[word] if word in self.extra_model else \
               self.emb_model[word.lower()] \
                   if force_lower and word.lower() in self.emb_model else \
               self.emb_model[word] if word in self.emb_model else \
               self.emb_model[word.lower()] \
                   if not force_lower \
                  and check_lower and word.lower() in self.emb_model else \
               self.unk if not skip_unk and self.unk is not None else \
               None

    def transform_words(self, words, check_lower=True, force_lower=False,
                        skip_unk=False):
        """Convert a word or a list of words to the corresponding
        vector|list of vectors.

        If *check_lower* is ``True`` (default), try to find a vector of a
        lower cased word if vector of a word is absent.

        If *force_lower* is ``True``, convert words to lower case before
        looking their vectors up. Default is ``False``.

        If *skip_unk* is ``True``, unknown words will be skipped."""
        return self.word_to_vec(words, check_lower=check_lower,
                                force_lower=force_lower, skip_unk=skip_unk) \
                   if isinstance(words, str) else \
               [self.word_to_vec(w, check_lower=check_lower,
                                 skip_unk=skip_unk) for w in words]

    def transform(self, sentences, check_lower=True, force_lower=False,
                  skip_unk=False, keep_empty=False, save=True, append=False):
        """Convert *sentences* of words to the sequences of the corresponding
        vectors and adjust their format for Dataset. If *skip_unk* is
        ``True``, unknown words will be skipped. If *keep_empty* is ``False``,
        we'll remove sentences that have no data after converting.

        If *check_lower* is ``True`` (default), try to find a vector of a
        lower cased word if vector of a word is absent.

        If *force_lower* is ``True``, convert sentences to lower case before
        looking their vectors up. Default is ``False``.

        If save is ``True``, we'll keep the converted sentences as the
        `Dataset` source.

        If *append* is ``True``, we'll append the converted sentences to the
        existing Dataset source. Elsewise (default), the existing Dataset
        source will be replaced. The param is used only if *save* is
        ``True``."""
        data = [tensor([
            v for v in s if keep_empty or v is not None
        ], dtype=self.float_tensor_dtype) for s in [
            self.transform_words(s, check_lower=check_lower,
                                 force_lower=force_lower, skip_unk=skip_unk) \
                for s in sentences
        ] if keep_empty or s]
        if save:
            if append:
                self.data += data
            else:
                self.data = data
        else:
            return data

    def _collate(self, batch, with_lens=True):
        """The method to use with `torch.utils.data.DataLoader` and
        `.transform_collate()`.

        :with_lens: return lengths of data.
        :return: depends on keyword args.
        :rtype: tuple(list([torch.tensor]), lens:torch.tensor)
        """
        assert self.pad_tensor is not None, \
               ('ERROR: pad_token must be defined if you want to use {} in '
                'DataLoader').format(self.__class__.__name__)
        device = batch[0].get_device() if batch[0].is_cuda else CPU
        lens = [tensor([len(x) for x in batch], device=device,
                       dtype=self.int_tensor_dtype)] if with_lens else []
        x = pad_sequences_with_tensor(batch, padding_tensor=self.pad_tensor)
        return (x, *lens) if lens else x
