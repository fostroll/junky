<h2 align="center">junky lib: PyTorch utilities</h2>

## Plotter

The lib contains tools to plot useful statistics on trained *PyTorch* models.

#### Plot Losses

```python
import junky
junky.plot_losses(train_losses, test_losses, accuracies=None, 
                  plot_title='Train/Dev Loss', figsize=(7, 4), 
                  legend_labels=['train', 'dev', '1 - acc'], save_name=None)
```

Plots train and dev losses obtained during training.

Also, can plot (1-accuracy) curve, if `accuracies` are specified.
The plot image is saved to disk.

Params:

**train_losses**:   list of floats, train losses throughout epochs.

**test_losses**:    list of floats, test losses throughout epochs.

**accuracies**:     optional, list of floats, accuracies throughout epochs.
Here, used to count (1 - accuracy). If `None`, not plotted.

**plot_title**:     plot title, `str`. Default value - 'Train/Dev Loss'.

**figsize**:        the size of the figure plotted. Default size is `(10,6)`.

**legend_labels**:  Line labels to use on the plot. Default: `['train', 'dev', '1 - acc']`.

**save_name**:      optional, if `None`, plot is not saved. 
Used as `fname` in `plt.savefig()`. Default file extention is '.png', 
if other extention is needed, please specify extention in save_name as well. 
*Example*: ``save_name='plot.pdf'``

#### Plot Metrics

```python
junky.plot_metrics(metrics=[], 
                   labels=['accuracy', 'precision', 'recalls', 'f1_score'],
                   plot_title='Metrics', figsize=(7, 4), save_name=None)
```
Plots metrics obtained during training. The plot image is saved to disk.

Custom metrics can also be plotted - specify them in `metrics` and assign them `labels`.

Params:

**metrics**:        tuple or list of metrics, where each metric is 
a list of floats, `len(metric)==num_epochs`

**labels**:         list of str, labels for metrics plotted.

**figsize**:        `tuple`: the size of the figure plotted.

**plot_title**:     `str`: plot title, default title - 'Metrics'.

**save_name**:      `str`: filename of figure file to save. 
If `None`, image is not saved to disk.

#### Plot Confusion Matrix

```python
junky.plot_confusion_matrix(y_true, y_pred, n_classes,
                            pad_index=None, ymap=None, figsize=(20, 10),
                            show_total=['x', 'y'], show_zeros=True,
                            plot_title='Confusion Matrix', save_name=None)
```
Generate matrix plot of confusion matrix with pretty annotations.
The plot image is saved to disk.

Params:

**y_true**:        true label of the data, with shape (nsamples,)

**y_pred**:        prediction of the data, with shape (nsamples,)

**n_classes**:     int: number of target classes. Is used to create index labels 
as (range(n_classes)). If padding class was also used during train, and 
`pad_index!=len(tag2index)`, i.e. is not the last element in tag2index, 
add +1 to n_classes, so that `pad_index` will be ignored if not present in `y_pred`. 

**pad_index**:     if not None and not present in y_pred, pad_index will not be 
included in the plot.

**ymap**:          dict: index -> tag, length == nclass. if not `None`, 
map the labels & ys to s. if `None`, range(1, n_classes+1) is used for labels.

**figsize**:       tuple: the size of the figure plotted.

**show_total**:    list of `str`. Where to display total number of class occurrences 
in the corpus: diagonal and/or axes. Up to all from ['diag', 'x', 'y'] can be chosen.
Default = ['x', 'y'] (show total only on axes).

**show_zeros**:    `bool`: whether to show zeros in the confusion matrix.

**plot_title**:    `str`: plot title, default title - 'Confusion Matrix'.

**save_name**:     `str`: filename of figure file to save. If `None`, 
image is not saved to disk.
