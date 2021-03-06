<h2 align="center">junky lib: PyTorch utilities</h2>

## Plotter

The lib contains tools to plot useful statistics on trained *PyTorch* models.

### Table of Contents

1. [Plot Losses](#loss)
1. [Plot Metrics](#metrics)
1. [Plot Confusion Matrix](#confusion)
1. [Plot Sequence Lengths](#seqlen)

### Plot Losses <a name="loss"></a>

```python
import junky
junky.plot_losses(train_losses, test_losses, accuracies=None,
                  plot_title='Train/Dev Loss', figsize=(7, 4),
				  legend_loc='best',
                  legend_labels=['train', 'dev', '1 - acc'], save_name=None)
```

Plots train and dev losses obtained during training.

Also, can plot (1-accuracy) curve, if `accuracies` are specified.
The plot image is saved to disk.

Params:

**train_losses**:   list of floats, train losses throughout epochs. If `None`,
not plotted.

**test_losses**:    list of floats, test losses throughout epochs. If `None`,
not plotted.

**accuracies**:     optional, list of floats, accuracies throughout epochs.
Here, used to count `(1 - accuracy)`. If `None`, not plotted.

**plot_title**:     plot title, `str`. Default value - 'Train/Dev Loss'.

**figsize**:        the size of the figure plotted. Default size is `(10,6)`.

**legend_loc**:     location of the legend according to matplotlib. Default:
'best'. If invalid `legend_loc` is entered, matplotlib will place the legend
in the `best` location and show a warning with all possible valid loc
definitions.

**legend_labels**:  Labels to use on the legend. Default:
`['train', 'dev', '1 - acc']`.

**save_name**:      optional, if `None`, plot is not saved. Used as `fname` in
`plt.savefig()`. Default file extention is '.png', if other extention is
needed, please specify extention in save_name as well. *Example*:
`save_name='plot.pdf'`

### Plot Metrics <a name="metrics"></a>

```python
junky.plot_metrics(metrics=[], legend_loc='best',
                   labels=['accuracy', 'precision', 'recalls', 'f1_score'],
                   plot_title='Metrics', figsize=(7, 4), save_name=None)
```
Plots metrics obtained during training. The plot image is saved to disk.

Custom metrics can also be plotted - specify them in `metrics` and assign them
`labels`.

Params:

**metrics**:        tuple or list of metrics, where each metric is a list of
floats, `len(metric)==num_epochs`.

**legend_loc**:     location of the legend according to matplotlib. Default:
'best'. If invalid `legend_loc` is entered, matplotlib will place the legend
in the `best` location and show a warning with all possible valid loc
definitions.

**labels**:         list of str, labels for metrics plotted.

**figsize**:        `tuple`: the size of the figure plotted.

**plot_title**:     `str`: plot title, default title - 'Metrics'.

**save_name**:      `str`: filename of figure file to save. If `None`, image
is not saved to disk.

### Plot Confusion Matrix <a name="confusion"></a>

```python
junky.plot_confusion_matrix(y_true, y_pred, 
                            pad_index=None, ymap=None, figsize=(20, 10),
                            show_total=['x', 'y'], show_zeros=True,
							show_empty_tags=False,
							plot_title='Confusion Matrix', save_name=None)
```
Generate matrix plot of confusion matrix with pretty annotations.
The plot image is saved to disk.

Params:

**y_true**:        true label of the data, with shape `(nsamples,)`

**y_pred**:        prediction of the data, with shape `(nsamples,)`

**pad_index**:     if not None and not present in `y_pred`, pad_index will not
be included in the plot.

**ymap**:          dict: index -> tag. If not `None`, map the predictions to
their categorical labels. If `None`, `range(1, len(set(y_true+y_pred)` is used
for labels.

**figsize**:       tuple: the size of the figure plotted.

**show_total**:    list of `str`. Where to display total number of class
occurrences in the corpus: diagonal and/or axes. Up to all from
`['diag', 'x', 'y']` can be chosen. Default = `['x', 'y']` (horizontal and
vertical axes respectively). If `None`, total values are not displayed on the
plot.

**show_zeros**:    `bool`: whether to show zeros in the confusion matrix.

**show_empty_tags**:    only active when when `ymap` is specified. If `True`,
all tags, including those that weren't met neither in `y_true` or `y_pred` are
displayed on the plot (filled with `0` in both axes). NB! If `True`, `pad_idx`
will also be displayed even if specified. Default: `False`, 'empty' tags are
skipped.

**plot_title**:    `str`: plot title, default title - 'Confusion Matrix'.

**save_name**:     `str`: filename of figure file to save. If `None`, image is
not saved to disk.

### Plot Sequence Lengths <a name="seqlen"></a>

```python
plot_seqlens(corpus, bins=100, **plot_kwargs)
```
Plots histogram of pretokenized sequence lengths from the corpus.

Args:

**corpus** (`list([list([str])])`): pretokenized corpus.

**figsize** (`tuple(x, y)`): figure size

**\*\*plot_kwargs**: any other `pandas.DataFrame.hist` keyword arguments.