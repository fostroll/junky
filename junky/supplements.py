# -*- coding: utf-8 -*-
# junky lib: supplements
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides supplement methods to use in PyTorch pipeline.
"""
from gensim.models import keyedvectors
import junky
import numpy as np
import torch


def make_word_embeddings(vocab, vectors=None, unk_token=None, pad_token=None,
                         with_layer=True, layer_freeze=True, **layer_kwargs):
    """Create or adjust Word to Index dict for embedding layer.

    :param vocab: array of words or dict of words with their indicies.
    :type vocab: list([word])|dict({word: int})
                             |dict({word: gensim.models.keyedvectors.Vocab})
    :param vectors: array of word vectors
    :type vectors: numpy.ndarray
    :param unk_token: add a token for unknown token.
    :type unk_token: str
    :param pad_token: add a token for padding.
    :type pad_token: str
    :param with_layer: if True, torch.nn.Embedding layer will be created from
        *vectors*.
    :param layer_freeze: If True, layer weights does not get updated in the
        learning process.
    :param **layer_kwargs: any other keyword args for
        torch.nn.Embedding.from_pretrained() method.
    :return: Word to Index dict; *vectors* (possibly, updated); the index of
        the unknown token; the index of the padding token; embedding layer.
    :rtype: tuple(dict({word: int}), int, int, torch.nn.Embedding)
    """
    assert vectors is None or len(vocab) == vectors.shape[0], \
           'ERROR: vocab and vectors must be of equal size'
    assert vectors is None or vectors.shape[0] != 0, \
           'ERROR: vectors must not be empty'
    assert vectors is not None or not (unk_token and pad_token), \
           'ERROR: unk_token and pad_token can be used only if vectors ' \
           'is not None'
    assert vectors is not None or not with_layer, \
           'ERROR: with_layer can be True only if vectors is not None'

    if isinstance(vocab, dict):
        val = next(iter(vocab.values()))
        if isinstance(val, keyedvectors.Vocab):
            vocab = {x: y.index for x, y in vocab.items()}
        elif not isinstance(val, int):
            vocab = None
    else:
        val = next(iter(vocab))
        if isinstance(val, int):
            vocab = {y: x for x, y in enumerate(vocab)}
        else:
            vocab = None
    assert vocab, 'ERROR: vocab of the incorrect type'

    def add_token(vocab, vectors, token):
        if token:
            idx = vocab[token] = len(vocab)
            vectors = junky.add_mean_vector(vectors, scale=.01)
        else:
            idx = None
        return vectors, idx

    vectors, unk_idx = add_token(vocab, vectors, unk_token)
    vectors, pad_idx = add_token(vocab, vectors, pad_token)

    emb_layer = torch.nn.Embedding.from_pretrained(
        embeddings=torch.from_numpy(vectors),
        padding_idx=pad_idx,
        freeze=layer_freeze,
        **layer_kwargs
    ) if with_layer else \
    None
    return vocab, vectors, unk_idx, pad_idx, emb_layer

def make_alphabet(sentences, pad_char=None, allowed_chars=None,
                  exclude_chars=None):
    """Make alphabet from the given corpus of tokenized *sentences*.

    :param sentences: tokenized sentences.
    :type sentences: list([list([str])])
    :param pad_char: add a token for padding.
    :type pad_char: str
    :param allowed_chars: if not None, all charactes not from *allowed_chars*
        will be removed.
    :type allowed_chars: None|str|list([str])
    :param exclude_chars: if not None, all charactes from *exclude_chars* will
        be removed.
    :type exclude_chars: None|str|list([str])
    :return: the alphabet created and the index of the padding character (it's
        always the last index, if pad_char is not None).
    :rtype: tuple(dict({char: int}), int)
    """
    abc = {
        x: i for i, x in enumerate(sorted(set(
            x for x in sentences
                if (not allowed_chars or x in allowed_chars)
               and (not exclude_chars or x not in exclude_chars)
                for x in x for x in x
        )))
    }
    if pad_char:
        assert pad_char not in abc, \
           "ERROR: char '{}' is already in vocabulary".format(pad_char)
        pad_idx = abc[pad_char] = len(abc)
    else:
        pad_idx = None

    return abc, pad_idx

def make_token_dict(sentences, pad_token=None):
    """Extract tokens from tokenized *sentences*, remove all duplicates, sort
    the resulting set and map all tokens onto ther indices.

    :param sentences: tokenized sentences.
    :type sentences: list([list([str])])
    :param pad_token: add a token for padding.
    :type pad_char: str
    :return: the dict created and the index of the padding token (it's
        always the last index, if pad_token is not None).
    :rtype: tuple(dict({char: int}), int)
    """
    t2idx = {
        x: i for i, x in enumerate(sorted(set(
            x for x in sentences for x in x
        )))
    }
    if pad_token:
        pad_idx = t2idx[token] = len(t2idx)
    else:
        pad_idx = None
    return t2idx, pad_idx

def get_conllu_fields(corpus=None, fields=None, word2idx=None, unk_token=None,
                      with_empty=False, silent=False):
    """Split corpus in CONLL-U format to separate lists of tokens and tags.

    :param corpus: the corpus in CONLL-U or Parsed CONLL-U format.
    :param fields: list of CONLL-U fields but 'FORM' to extract.
    :type fields: None|list
    :param word2idx: Word to Index dict. If not None, all words not from dict
        will be skipped or replacet to *unk_token*
    :type word2idx: dict({word: int})
    :param unk_token: replacement for tokens that not present in *word2idx*.
    :type unk_token: str
    :param with_empty: don't skip empty sentences.
    :param silent: suppress output.
    :return: splitted corpus
    :rtype: tuple(list([list([str|OrderedDict])]))
    """
    if isinstance(corpus, str):
        corpus = Conllu.load(corpus, **({'log_file': None} if silent else{}))
    elif callable(corpus):
        corpus = corpus()

    sents = tuple([] for _ in range(len(fields) + 1))

    for sent, _ in corpus:
        for i, field in enumerate(zip(*[
            (x['FORM'] if not word2idx or x['FORM'] in word2idx else
             unk_token,
             *([x[y] for y in fields] if fields else []))
                 for x in sent
                     if x['FORM'] and '-' not in x['ID']
                                  and (not word2idx or x['FORM'] in word2idx
                                                    or unk_token)
        ])):
            if field or with_empty:
                sents[i].append(field)

    return sents
