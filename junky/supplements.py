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
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def clear_stderr():
    if hasattr(tqdm, '_instances'):
        for instance in list(tqdm._instances):
            tqdm._decr_instances(instance)

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
        pad_idx = t2idx[pad_token] = len(t2idx)
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


class WordSeqDataset(torch.utils.data.Dataset):
    """
    Dataset for sequence tagging with word-level input.

    Args:
        x_data: sequences of word indices: list([list([int])]).
        y_data: sequences of label's indices: list([list([int])]).

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
                              num_workers=0, shuffle=True,
                              collate_fn=pad_collate if pad_collate else
                              train_dataset.pad_collate)
    test_loader = loaders[1] if loaders else \
                  DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, shuffle=False,
                             collate_fn=pad_collate if pad_collate else
                             test_dataset.pad_collate)

    train_losses, test_losses = [], []
    best_res = float('-inf')
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
                for i in range(len(data)):
                    data[i] = to_device(data[i])
            return data

        model.train()
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad()
            pred = model(*batch[:-2])

            batch_loss = []
            for i in range(pred.size(0)):
                tmp_loss = criterion(pred[i], y[i])
                batch_loss.append(tmp_loss)

            loss = torch.mean(torch.stack(batch_loss)) 
            loss.backward()
            optimizer.step()
            train_losses_.append(loss.item())

            if with_progress:
                progress_bar.set_postfix(
                    train_loss=np.mean(train_losses_[-500:])
                )
                progress_bar.update(x.shape[0])

        if with_progress:
            progress_bar.close()

        mean_train_loss = np.mean(train_losses_)
        train_losses.append(mean_train_loss)

        test_losses_, test_golds, test_preds = [], [], []

        model.eval()
        for batch in dev_loader:

            [test_golds.extend(y_[:len_])
                 for y_, len_ in zip(batch[-2].numpy(), batch[-1])]

            batch = to_device(batch)

            with torch.no_grad():           
                pred = model(*batch[:-2])

            pred_values, pred_indices = pred.max(2)

            pred_cpu = pred_indices.cpu()
            [test_preds.extend(y_[:len_])
                 for y_, len_ in zip(pred_cpu.numpy(), batch[-1])]

            batch_loss = []
            for i in range(pred.size(0)):
                loss_ = criterion(pred[i], y[i])
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

        res = -mean_test_loss if control_metric == 'loss' else \
              accuracy if control_metric == 'accuracy' else \
              f1 if control_metric == 'f1' else \
              None

        if res > best_res:
            best_res = res
            best_test_golds, best_test_preds = test_golds[:], test_preds[:]
            best_model_backup_method(model, res)
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
            'best_res': best_res,
            'best_test_golds': best_test_golds,
            'best_test_preds': best_test_preds,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1s': f1s}
