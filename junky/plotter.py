# -*- coding: utf-8 -*-
# junky lib: plotter
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a bunch of tools to visualize train results.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
import matplotlib.style
from matplotlib import pyplot as plt
import seaborn as sns


def plot_losses(train_losses, test_losses, accuracies=None,
                plot_title='Train/Dev Loss', figsize=(7, 4),
                legend_labels=['train', 'dev', '1 - acc'], save_name=None):
    """Plots train and dev losses obtained during training.
    The plot image is saved to disk.
    args:
      train_losses:   list of floats, train losses throughout epochs.
      test_losses:    list of floats, test losses throughout epochs.
      accuracies:     optional, list of floats, accuracies throughout epochs.
                      Here, used to count (1 - accuracy). If `None`, not
                      plotted.
      plot_title:     plot title, `str`. Default value - 'Train/Dev Loss'.
      figsize:        the size of the figure plotted. Default size is `(5,3)`.
      save_name:      optional, if `None`, plot is not saved. 
                      Used as `fname` in `plt.savefig()`.
                      Default file extention is '.png', if other extention is needed, 
                      please specify extention in save_name as well. 
                      Example: save_name='plot.pdf'
    """
    mpl.style.use('default')
    plt.figure(figsize=figsize)
    plt.plot([None] + train_losses)
    plt.plot([None] + test_losses)
    if accuracies is not None:
        plt.plot([None] + [1 - x for x in accuracies])
    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend_labels, loc='upper right')
    plt.grid()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, n_classes,
                          pad_index=None, ymap=None, figsize=(20, 10),
                          show_total=['x', 'y'], show_zeros=True,
                          plot_title='Confusion Matrix', save_name=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:        true label of the data, with shape (nsamples,)
      y_pred:        prediction of the data, with shape (nsamples,)
      n_classes:     int: number of target classes. Is used to create 
                     index labels as (range(n_classes)).
                     If padding class was also used during train, and 
                     pad_index!=len(tag2index), i.e. is not the last element
                     in tag2index, add +1 to n_classes, so that pad_index 
                     will be ignored if not present in y_pred.                     
      pad_index:     if not None and not present in y_pred, pad_index 
                     will not be included in the plot.                  
      ymap:          dict: index -> tag, length == nclass.
                     if not `None`, map the labels & ys to s.
                     if `None`, range(1, n_classes+1) is used for labels.
      figsize:       tuple: the size of the figure plotted.
      show_total:    list of `str`. Where to display total number of 
                     class occurrences in the corpus: diagonal and/or axes.
                     Up to all from ['diag', 'x', 'y'] can be chosen.
                     Default = ['x', 'y']
      show_zeros:    bool: whether to show zeros in the confusion matrix.
      plot_title:    str: plot title, default title - 'Confusion Matrix'.
      save_name:     str: filename of figure file to save. 
                     if `None`, image is not saved to disk.
    """
    # if pad_index is not None and does not exist in y_pred, it's excluded
    # from labels
    mpl.style.use('default')

    labels = [i if pad_index is not None and i != pad_index
                                         and pad_index not in y_pred else
              i
              for i in range(0, n_classes)]
    if ymap is not None:
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                if 'diag' in show_total:
                    s = cm_sum[i]
                    annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
                else:
                    annot[i, j] = '%.2f%%\n%d' % (p, c)
            elif c == 0:
                if show_zeros:
                    annot[i, j] = '0'
                else:
                    annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d' % (p, c)

    total_labels = [str(i)+'\n'+str(n[0]) for i, n in zip(labels, cm_sum)]

    cm = pd.DataFrame(cm, 
                      index=total_labels if 'x' in show_total else labels, 
                      columns=total_labels if 'y' in show_total else labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)

    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.title(plot_title)
    plt.show()

def plot_metrics(metrics=[], 
                 labels=['accuracy', 'precision', 'recalls', 'f1_score'],
                 plot_title='Metrics', figsize=(7, 4), save_name=None):
    """Plots metrics obtained during training. 
    Default: ['accuracy', 'precision', 'recalls', 'f1_score'].
    The plot image is saved to disk.
    args:
      metrics:        tuple or list of metrics, where each metric is
                      a list of floats, len(metric)==num_epochs
      labels:         list of str, labels for metrics plotted.
      figsize:        tuple: the size of the figure plotted.
      plot_title:     str: plot title, default title - 'Metrics'.
      save_name:      str: filename of figure file to save. 
                      if `None`, image is not saved to disk.
    """
    mpl.style.use('default')
    plt.figure(figsize=figsize)
    for metric in metrics:
        plt.plot([None] + metric)
    plt.grid()
    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend(labels, loc='lower right')
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
