# -*- coding: utf-8 -*-
# junky lib: supplements
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides supplement methods to use in PyTorch pipeline.
"""
from collections.abc import Iterable
from gensim.models import keyedvectors
import junky
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                            recall_score
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def clear_stderr():
    if hasattr(tqdm, '_instances'):
        for instance in list(tqdm._instances):
            tqdm._decr_instances(instance)

def make_word_embeddings(vocab, vectors=None,
                         pad_token=None, extra_tokens=None,
                         with_layer=True, layer_freeze=True, **layer_kwargs):
    """Create or adjust Word to Index dict for embedding layer.

    :param vocab: array of words or dict of words with their indicies.
    :type vocab: list([word])|dict({word: int})
                             |dict({word: gensim.models.keyedvectors.Vocab})
    :param vectors: array of word vectors
    :type vectors: numpy.ndarray
    :param pad_token: add a token for padding.
    :type pad_token: str
    :param extra_tokens: add any tokens for other purposes.
    :type extra_tokens: list([str])
    :param with_layer: if True, torch.nn.Embedding layer will be created from
        *vectors*.
    :param layer_freeze: If True, layer weights does not get updated in the
        learning process.
    :param **layer_kwargs: any other keyword args for
        torch.nn.Embedding.from_pretrained() method.
    :return: Word to Index dict; *vectors* (possibly, updated); the index of
        the padding token (it's always the last index, if pad_token is not
        None); the indices of the extra tokens; embedding layer.
    :rtype: tuple(dict({word: int}), int, list([int])|None,
        torch.nn.Embedding)
    """
    assert vectors is None or len(vocab) == vectors.shape[0], \
           'ERROR: vocab and vectors must be of equal size'
    assert vectors is None or vectors.shape[0] != 0, \
           'ERROR: vectors must not be empty'
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

    def add_token(vectors, token):
        if token:
            assert token not in vocab, \
                   "ERROR: token '{}' is already in the vocab".format(token)
            idx = vocab[token] = len(vocab)
            if vectors is not None:
                vectors = junky.add_mean_vector(vectors, scale=.01)
        else:
            idx = None
        return vectors, idx

    if extra_tokens is not None:
        extra_idxs = []
        for t in extra_tokens:
            vectors, idx = add_token(vectors, t)
            extra_idxs.append(idx)
    else:
        extra_idxs = None
    vectors, pad_idx = add_token(vectors, pad_token)

    emb_layer = torch.nn.Embedding.from_pretrained(
        embeddings=torch.from_numpy(vectors),
        padding_idx=pad_idx,
        freeze=layer_freeze,
        **layer_kwargs
    ) if with_layer else \
    None

    return vocab, vectors, pad_idx, extra_idxs, emb_layer

def make_alphabet(sentences, pad_char=None, extra_chars=None,
                  allowed_chars=None, exclude_chars=None):
    """Make alphabet from the given corpus of tokenized *sentences*.

    :param sentences: tokenized sentences.
    :type sentences: list([list([str])])
    :param pad_char: add a token for padding.
    :type pad_char: str
    :param extra_chars: add tokens for other purposes.
    :type extra_chars: list([str])
    :param allowed_chars: if not None, all charactes not from *allowed_chars*
        will be removed.
    :type allowed_chars: str|list([str])
    :param exclude_chars: if not None, all charactes from *exclude_chars* will
        be removed.
    :type exclude_chars: str|list([str])
    :return: the alphabet created; the index of the padding token (it's always
        the last index, if pad_char is not None); the indices of the extra
        characters.
    :rtype: tuple(dict({char: int}), int, list([int])|None)
    """
    abc = {
        x: i for i, x in enumerate(sorted(set(
            x for x in sentences
                if (not allowed_chars or x in allowed_chars)
               and (not exclude_chars or x not in exclude_chars)
                for x in x for x in x
        )))
    }

    def add_char(char):
        if char:
            assert char not in abc, \
                   "ERROR: char '{}' is already in the alphabet".format(char)
            idx = abc[char] = len(abc)
        else:
            idx = None
        return idx

    if extra_chars is not None:
        extra_idxs = [add_char(c) for c in extra_chars]
    else:
        extra_idxs = None
    pad_idx = add_char(pad_char)

    return abc, pad_idx, extra_idxs

def make_token_dict(sentences, pad_token=None, extra_tokens=None):
    """Extract tokens from tokenized *sentences*, remove all duplicates, sort
    the resulting set and map all tokens onto ther indices.

    :param sentences: tokenized sentences.
    :type sentences: list([list([str])])
    :param pad_token: add a token for padding.
    :type pad_char: str
    :param extra_tokens: add any tokens for other purposes.
    :type extra_tokens: list([str])
    :return: the dict created and the index of the padding token (it's
        always the last index, if pad_token is not None); the indices of
        the extra tokens.
    :rtype: tuple(dict({char: int}), int, list([int])|None)
    """
    t2idx = {
        x: i for i, x in enumerate(sorted(set(
            x for x in sentences for x in x
        )))
    }

    def add_token(token):
        if token:
            assert token not in t2idx, \
                   "ERROR: token '{}' is already in the dict".format(token)
            idx = t2idx[token] = len(t2idx)
        else:
            idx = None
        return idx

    if extra_tokens is not None:
        extra_idxs = [add_token(t) for t in extra_tokens]
    else:
        extra_idxs = None
    pad_idx = add_token(pad_token)

    return t2idx, pad_idx, extra_idxs

def get_conllu_fields(corpus=None, fields=None, word2idx=None, unk_token=None,
                      with_empty=False, silent=False):
    """Split corpus in CONLL-U format to separate lists of tokens and tags.

    :param corpus: the corpus in CONLL-U or Parsed CONLL-U format.
    :param fields: list of CONLL-U fields but 'FORM' to extract.
    :type fields: list
    :param word2idx: Word to Index dict. If not None, all words not from dict
        will be skipped or replacet to *unk_token*
    :type word2idx: dict({word: int})
    :param unk_token: replacement for tokens that are not present in
        *word2idx*.
    :type unk_token: str
    :param with_empty: don't skip empty sentences.
    :param silent: suppress output.
    :return: splitted corpus
    :rtype: tuple(list([list([str|OrderedDict])]))
    """
    if fields is None:
        fields = []

    if not silent:
        clear_stderr()

    if isinstance(corpus, str):
        corpus = Conllu.load(corpus, **({'log_file': None} if silent else{}))
    elif callable(corpus):
        corpus = corpus()

    sents = tuple([] for _ in range(len(fields) + 1))

    for sent, _ in corpus:
        for i, field in enumerate(zip(*[
            (x['FORM'] if not word2idx or x['FORM'] in word2idx else
             unk_token,
             *[x[y] for y in fields])
                 for x in sent
                     if x['FORM'] and '-' not in x['ID']
                                  and (not word2idx or x['FORM'] in word2idx
                                                    or unk_token)
        ])):
            if field or with_empty:
                sents[i].append(field)

    return sents


class WordSeqDataset(torch.utils.data.Dataset):
    """
    Dataset for sequence tagging with word-level input.

    Args:
        x_data: sequences of word indices: list([list([int])]).
        y_data: sequences of label's indices: list([list([int])]).
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
        x_pad: index of the padding token for `x_data`.
        y_pad: index of the padding token for `y_data`.

    Output:
        x:list([torch.tensor]), x_lens:torch.tensor,
        y:list([torch.tensor]), y_lens:torch.tensor
    """
    def __init__(self, x_data, y_data, batch_first=False,
                 x_pad=0, y_pad=0):
        super().__init__()
        self.x_data = [torch.tensor(x) for x in x_data]
        self.y_data = [torch.tensor(y) for y in y_data]
        self.batch_first = batch_first
        self.x_pad = x_pad
        self.y_pad = y_pad

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def pad_collate(self, batch):
        x_lens = torch.tensor([len(x[0]) for x in batch])
        y_lens = torch.tensor([len(x[1]) for x in batch])

        x = pad_sequence([x[0] for x in batch], batch_first=self.batch_first,
                         padding_value=self.x_pad)
        y = pad_sequence([y[1] for y in batch], batch_first=self.batch_first,
                         padding_value=self.y_pad)

        return x, x_lens, y, y_lens


class CharSeqDataset(torch.utils.data.Dataset):
    """
    Dataset for sequence tagging with char-level input.

    Args:
        x_ch_data: sequences of sequences of char indices:
            list([list([list([int])])]).
        y_data: sequences of label's indices: list([list([int])]).
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
        x_ch_pad: index of the padding token for `x_ch_data`.
        y_pad: index of the padding token for `y_data`.

    Output:
        x_ch:list([list([torch.tensor])]), x_ch_lens:list([torch.tensor]),
        y:list([torch.tensor]), y_lens:torch.tensor
    """
    def __init__(self, x_ch_data, y_data, batch_first=False,
                 x_ch_pad=0, y_pad=0):
        super().__init__()
        self.x_ch_data = [
            [torch.tensor(x_ch) for x_ch in sent] for sent in x_ch_data
        ]
        self.y_data = [torch.tensor(y) for y in y_data]
        self.batch_first = batch_first
        self.x_ch_pad = x_ch_pad
        self.y_pad = y_pad

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_ch_data[idx], self.y_data[idx]

    def pad_collate(self, batch):
        x_ch_lens = [torch.tensor([len(x) for x in x[0]]) for x in batch]
        y_lens = torch.tensor([len(x[1]) for x in batch])

        if self.min_len is not None:
            batch.append(([torch.tensor([self.x_ch_pad])] * self.min_len,
                          torch.tensor([self.y_pad] * self.min_len)))

        x_ch = junky.pad_array_torch([x[0] for x in batch],
                                     padding_value=self.x_ch_pad)
        y = pad_sequence([x[1] for x in batch], batch_first=self.batch_first,
                         padding_value=self.y_pad)

        if self.min_len is not None:
            x_ch = x_ch[:-1]
            y = y[:-1]

        return x_ch, x_ch_lens, y, y_lens


class WordCharSeqDataset(Dataset):
    """
    Dataset for sequence tagging with both word- and char-level inputs.

    Args:
        x_data: sequences of word indices: list([list([int])]).
        x_ch_data: sequences of sequences of char indices:
            list([list([list([int])])]).
        y_data: sequences of label's indices: list([list([int])]).
        batch_first: If ``True``, then the input and output tensors are
            provided as `(batch, seq, feature)`. Otherwise (default),
            `(seq, batch, feature)`.
        x_pad: index of the padding token for `x_data`.
        x_ch_pad: index of the padding token for `x_ch_data`.
        y_pad: index of the padding token for `y_data`.
        min_len: min length of the resulting arrays. If *_data is shorter,
            they will be appended with *_pad.

    Output:
        x:list([torch.tensor]), x_lens:torch.tensor,
        x_ch:list([list([torch.tensor])]), x_ch_lens:list([torch.tensor]),
        y:list([torch.tensor]), y_lens:torch.tensor
    """
    def __init__(self, x_data, x_ch_data, y_data, batch_first=False,
                 x_pad=0, x_ch_pad=0, y_pad=0, min_len=None):
        super().__init__()
        self.x_data = [torch.tensor(x) for x in x_data]
        self.x_ch_data = [[torch.tensor(x_ch) for x_ch in sent]
                              for sent in x_ch_data]
        self.y_data = [torch.tensor(y) for y in y_data]
        self.batch_first = batch_first
        self.x_pad = x_pad
        self.x_ch_pad = x_ch_pad
        self.y_pad = y_pad
        self.min_len = min_len

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.x_ch_data[idx], self.y_data[idx]

    def pad_collate(self, batch):
        x_lens = torch.tensor([len(x[0]) for x in batch])
        x_ch_lens = [torch.tensor([len(x) for x in x[1]]) for x in batch]
        y_lens = torch.tensor([len(x[2]) for x in batch])

        if self.min_len is not None:
            batch.append((torch.tensor([self.x_pad] * self.min_len),
                          [torch.tensor([self.x_ch_pad])] * self.min_len,
                          torch.tensor([self.y_pad] * self.min_len)))

        x = pad_sequence([x[0] for x in batch], batch_first=self.batch_first,
                         padding_value=self.x_pad)
        x_ch = junky.pad_array_torch([x[1] for x in batch],
                                     padding_value=self.x_ch_pad)
        y = pad_sequence([x[2] for x in batch], batch_first=self.batch_first,
                         padding_value=self.y_pad)

        if self.min_len is not None:
            x = x[:-1]
            x_ch = x_ch[:-1]
            y = y[:-1]

        return x, x_lens, x_ch, x_ch_lens, y, y_lens


def train(device, loaders, model, criterion, optimizer,
          best_model_backup_method, log_prefix, train_dataset, test_dataset,
          pad_collate=None, epochs=100, bad_epochs=8, batch_size=32,
          control_metric='accuracy', with_progress=True):

    assert control_metric in ['accuracy', 'f1', 'loss'], \
           "ERROR: unknown control_metric '{}' ".format(control_metric) \
         + "(only 'accuracy', 'f1' and 'loss' are available)"

    train_loader = loaders[0] if loaders else \
                   DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=pad_collate if pad_collate else
                              train_dataset.pad_collate)
    test_loader = loaders[1] if loaders else \
                  DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=pad_collate if pad_collate else
                             test_dataset.pad_collate)

    train_losses, test_losses = [], []
    best_scores = float('-inf')
    best_test_golds, best_test_preds = [], []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    bad_epochs_ = 0

    if with_progress:
        clear_stderr()
    print_indent = ' ' * len(log_prefix)
    for epoch in range(epochs):
        train_losses_ = []

        progress_bar = tqdm(total=len(train_loader.dataset),
                            desc='Epoch {}'.format(epoch + 1)) \
                           if with_progress else \
                       None

        def to_device(data):
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, Iterable):
                data = type(data)(to_device(x) for x in data)
            return data

        model.train()
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad()
            pred, gold = model(*batch[:-2]), batch[-2]

            batch_loss = []
            for i in range(pred.size(0)):
                tmp_loss = criterion(pred[i], gold[i])
                batch_loss.append(tmp_loss)

            loss = torch.mean(torch.stack(batch_loss)) 
            loss.backward()
            optimizer.step()
            train_losses_.append(loss.item())

            if with_progress:
                progress_bar.set_postfix(
                    train_loss=np.mean(train_losses_[-500:])
                )
                progress_bar.update(batch[0].shape[0])

        if with_progress:
            progress_bar.close()

        mean_train_loss = np.mean(train_losses_)
        train_losses.append(mean_train_loss)

        test_losses_, test_golds, test_preds = [], [], []

        model.eval()
        for batch in test_loader:
            gold, gold_lens = batch[-2:]
            [test_golds.extend(y_[:len_])
                 for y_, len_ in zip(gold.numpy(), gold_lens)]

            batch = to_device(batch)
            with torch.no_grad():           
                pred, gold, gold_lens = model(*batch[:-2]), *batch[-2:]

            pred_values, pred_indices = pred.max(2)

            [test_preds.extend(y_[:len_])
                 for y_, len_ in zip(pred_indices.cpu().numpy(), gold_lens)]

            batch_loss = []
            for i in range(pred.size(0)):
                loss_ = criterion(pred[i], gold[i])
                batch_loss.append(loss_)

            loss = torch.mean(torch.stack(batch_loss))
            test_losses_.append(loss.item())

        mean_test_loss = np.mean(test_losses_)
        test_losses.append(mean_test_loss)

        accuracy = accuracy_score(test_golds, test_preds)
        precision = precision_score(test_golds, test_preds, average='macro')
        recall = recall_score(test_golds, test_preds, average='macro')
        f1 = f1_score(test_golds, test_preds, average='macro')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print('{}Epoch {}: \n'.format(log_prefix, epoch + 1)
            + '{}Losses: train = {:.8f}, test = {:.8f}\n'
                  .format(print_indent, mean_train_loss, mean_test_loss)
            + '{}Test: accuracy = {:.8f}\n'.format(print_indent, accuracy)
            + '{}Test: precision = {:.8f}\n'.format(print_indent, precision)
            + '{}Test: recall = {:.8f}\n'.format(print_indent, recall)
            + '{}Test: f1_score = {:.8f}'.format(print_indent, f1))

        scores = -mean_test_loss if control_metric == 'loss' else \
                 accuracy if control_metric == 'accuracy' else \
                 f1 if control_metric == 'f1' else \
                 None

        if scores > best_scores:
            best_scores = scores
            best_test_golds, best_test_preds = test_golds[:], test_preds[:]
            best_model_backup_method(model, scores)
            bad_epochs_ = 0
        else:
            bad_epochs_ += 1
            print('{}BAD EPOCHS: {}'.format(log_prefix, bad_epochs_))
            if bad_epochs_ >= bad_epochs:
                print('{}Maximum bad epochs exceeded. Process has stopped'
                          .format(log_prefix))
                break

        sys.stdout.flush()

    return {'train_losses': train_losses,
            'test_losses': test_losses,
            'best_scores': best_scores,
            'best_test_golds': best_test_golds,
            'best_test_preds': best_test_preds,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1s': f1s}
