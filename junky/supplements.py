# -*- coding: utf-8 -*-
# junky lib: supplements
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides supplement methods to use in PyTorch pipeline.
"""
from collections.abc import Iterable
from corpuscula import Conllu
from gensim.models import keyedvectors
import junky
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
                            recall_score
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def clear_tqdm():
    if hasattr(tqdm, '_instances'):
        for instance in list(tqdm._instances):
            tqdm._decr_instances(instance)
def clear_stderr():
    import sys
    print('WARNING: clear_stderr() is deprecated and is going to be removed '
          'in future releases. Use clear_tqdm() instead.', file=sys.stderr)
    clear_tqdm()

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
    """Split corpus in CoNLL-U format to separate lists of tokens and tags.

    :param corpus: the corpus in CoNLL-U or Parsed CoNLL-U format.
    :param fields: list of CoNLL-U fields but 'FORM' to extract.
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

    if isinstance(corpus, str):
        corpus = Conllu.load(corpus, **({'log_file': None} if silent else{}))
    elif callable(corpus):
        corpus = corpus()

    sents = tuple([] for _ in range(len(fields) + 1))

    for sent in corpus:
        if isinstance(sent, tuple):
            sent = sent[0]
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

    return sents if fields else sents[0]

def conllu_remove(corpus, remove=None):
    for sent in corpus:
        if remove:
            if isinstance(sent, tuple):
                sent = sent[0]
            sent = [
                x for x in sent if not [
                    1 for f, v in remove.items() if x[f] in
                        ([v] if isinstance(v, str) else v)
                ]
            ]
        yield sent

def extract_conllu_fields(corpus, fields=None, word2idx=None, unk_token=None,
                          with_empty=False, return_nones=False, silent=False):
    """Split corpus in CoNLL-U format to separate lists of tokens and tags.

    :param corpus: the corpus in CoNLL-U or Parsed CoNLL-U format.
    :param fields: list of CoNLL-U fields but 'FORM' to extract.
    :type fields: list
    :param word2idx: Word to Index dict. If not None, all words not from dict
        will be skipped or replacet to *unk_token*
    :type word2idx: dict({word: int})
    :param unk_token: replacement for tokens that are not present in
        *word2idx*.
    :type unk_token: str
    :param with_empty: don't skip empty sentences.
    :param silent: suppress output.
    :param return_nones: return indexes of filtered sentences and tokens
    :return: splitted corpus
    :rtype: tuple(list([list([str|OrderedDict])])), [ list([<empty sent idx]),
            list([tuple(<empty token sent idx>, <empty token idx>)]) ]
    """
    if fields is None:
        fields = []

    if isinstance(corpus, str):
        corpus = Conllu.load(corpus, **({'log_file': None} if silent else{}))
    elif callable(corpus):
        corpus = corpus()

    sents = tuple([] for _ in range(len(fields) + 1))
    empties, nones = [], []

    for i, sent in enumerate(corpus):
        if isinstance(sent, tuple):
            sent = sent[0]
        for j, field in enumerate(zip(*[
            (x['FORM'] if not word2idx or x['FORM'] in word2idx else
             unk_token,
             *[x[y] for y in fields])
                 for x in sent
                     if x['FORM'] and '-' not in x['ID']
                                  and (not word2idx or x['FORM'] in word2idx
                                                    or unk_token)
        ])):
            if field or with_empty:
                sents[j].append(field)
            else:
                if return_nones:
                    empties.append(i)
                break

        if return_nones:
            for j, x in enumerate(sent):
                if not(x['FORM'] and '-' not in x['ID']
                                 and (not word2idx or x['FORM'] in word2idx
                                                   or unk_token)):
                    nones.append((i, j))

    return (*sents, *((empties, nones) if return_nones else [])) \
               if fields or return_nones else \
           sents[0]

def embed_conllu_fields(corpus, fields, values, empties=None, nones=None):
    if empties:
        for i in empties:
            values.insert(i, [])
    if nones:
        for i, j in nones:
            values[i].insert(j, None)
    for sentence, vals in zip(corpus, values):
        sent = sentence[0] if isinstance(sentence, tuple) else sentence
        for token, val in zip(sent, vals):
            for field, val_ in [[fields, val]] \
                                   if isinstance(fields, str) else \
                               zip(fields, val):
                token[fields] = val_
        yield sentence

def train(device, loaders, model, criterion, optimizer, scheduler,
          best_model_backup_method, log_prefix, datasets=None,
          pad_collate=None, epochs=100, bad_epochs=8, batch_size=32,
          control_metric='accuracy', best_score=None, with_progress=True):

    assert control_metric in ['accuracy', 'f1', 'loss'], \
           "ERROR: unknown control_metric '{}' ".format(control_metric) \
         + "(only 'accuracy', 'f1' and 'loss' are available)"
    assert loaders or datasets, \
           'ERROR: You must pass a tuple of Dataloader or Dataset ' \
           'instances for train and test goals'

    train_loader = loaders[0] if loaders else \
                   datasets[0].create_loader(batch_size=batch_size,
                                             shuffle=True, num_workers=0) \
                       if callable(getattr(datasets[0],
                                           'create_loader', None)) else \
                   DataLoader(datasets[0], batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=pad_collate if pad_collate else
                              datasets[0].pad_collate)
    test_loader = loaders[1] if loaders else \
                  datasets[1].create_loader(batch_size=batch_size,
                                            shuffle=False, num_workers=0) \
                      if callable(getattr(datasets[1],
                                          'create_loader', None)) else \
                  DataLoader(datasets[1], batch_size=batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=pad_collate if pad_collate else
                             datasets[1].pad_collate)

    best_epoch = None
    if best_score is None:
        best_score = float('-inf')
    prev_score = best_score

    train_losses, test_losses = [], []
    best_test_golds, best_test_preds = [], []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    bad_epochs_ = 0

    if with_progress:
        clear_tqdm()
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
            pred, gold = model(*batch[:-1]), batch[-1]

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
            gold, gold_lens = batch[-1], batch[1]
            [test_golds.extend(y_[:len_])
                 for y_, len_ in zip(gold.cpu().numpy(), gold_lens)]

            batch = to_device(batch)
            with torch.no_grad():           
                pred = model(*batch[:-1])

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

        score = -mean_test_loss if control_metric == 'loss' else \
                accuracy if control_metric == 'accuracy' else \
                f1 if control_metric == 'f1' else \
                None

        if scheduler:
            scheduler.step(score) 

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_test_golds, best_test_preds = test_golds[:], test_preds[:]
            best_model_backup_method(model, score)
            bad_epochs_ = 0
        else:
            if score <= prev_score:
                bad_epochs_ += 1
            sgn = '{} {}'.format('==' if score == best_score else '<<',
                                 '<' if score < prev_score else
                                 '=' if score == prev_score else
                                 '>')
            print('{}BAD EPOCHS: {} ({})'
                      .format(log_prefix, bad_epochs_, sgn))
            if bad_epochs_ >= bad_epochs:
                print(('{}Maximum bad epochs exceeded. Process has stopped. '
                       'Best epoch: {}').format(log_prefix, best_epoch))
                break
        prev_score = score

        sys.stdout.flush()

    return {'train_losses': train_losses,
            'test_losses': test_losses,
            'best_score': best_score,
            'best_test_golds': best_test_golds,
            'best_test_preds': best_test_preds,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1s': f1s}
