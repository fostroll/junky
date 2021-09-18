<h2 align="center">junky lib: PyTorch utilities</h2>

## Trainer

This is just a trainer for *PyTorch* *models*. We use it in our projects for
not to copypaste identical blocks of code between training pipelines. Contain
the configuration part that can be serialized and saved and the trainer
itself.

#### The configuration

```python
from junky.trainer import TrainerConfig

config = TrainrConfig(save_dir, **kwargs)
```

Args:

**save_dir** (`str`): the directory where to save the best model.

**batch_labels_idx** (`int`; default is -1): labels position in the batch
received from loader.

**batch_lens_idx** (`int`; default is `2`): lengthts of sentences in the
batch. Used only in the postprocessing, so it will be ignored if
postprocess doesn't apply (see the `Trainer` class description).

**model_args** (`list`; default is `None`): positional arguments of the
`model.forward()` method in the train mode. It must be indices of the
positions in the batch.

**model_kwargs** (`dict`; default is None): keyword arguments of the
`model.forward()` method in the train mode. It's the dict of kwarg names
and corresponding positions in the batch.

***Example***: `model_args=[0, 1], model_kwargs={'labels': 2}` results in
invoking `model(batch[0], batch[1], labels=batch[2])`. Before this, all
the batch will be moved to the model's device.

***NB:*** if both **model_args** and **model_kwargs** are `None`,
`model.forward(*batch)` is invoked.

**output_logits_idx** (default is `0`): if model.forward() returns a
`tuple`, it's the position of logits in that `tuple`.

**output_loss_idx** (default is `1`): if `model.forward()` calculates a
loss in the train mode by itself, it's the position of the loss in the
returning `tuple`.

**min_epochs** (default is `0`): the number of epochs we continue training
even if the number of bad epochs surpassed the **bad_epochs** param.

**max_epochs** (default is `None`): the total number of epochs can't be
greater than this value.

**bad_epochs** (default is `5`): we stop training if the control_metric
doesn't increase on the validation set for a period of this number of
epochs (really, the algorithm is slightly more complex but the meaning is
like that).

**adam_lr** (default is `.0001`), **adam_betas** (default is
`(0.9, 0.999)`), **adam_eps** (default is `1e-08`), **adam_weight_decay**
(default is `0`), **adam_amsgrad** (default is `False`): params for *Adam*
optimizer.

**adamw_lr** (default is `5e-5`), **adamw_betas** (default is
`(0.9, 0.999)`), **adamw_eps** (default is `1e-8`), **adamw_weight_decay**
(default is `0.01`), **adamw_amsgrad** (default is `False`) params for
*AdamW* optimizer.

**sgd_lr** (default is `.001`), **sgd_momentum** (default is `.9`),
**sgd_weight_decay** (default is `0`), **sgd_dampening** (default is `0`),
**sgd_nesterov** (default is `False`): params for *SGD* optimizer.

**grad_norm_clip** (default is `None`): if defined, we clip gradient norm
of the model's parameters to that value.

**criterion** (default is `None`): the function to calculate the loss. If
`None`, we suppose that the model in the train mode calculates the loss by
itself.

**optimizer** (default is `'SGD'`): the function to update the model's
parameters. Allowed values are: `'Adam'`, `'AdamW'`, `'SGD'` or instance
of the `torch.optim.Optimizer` class.

**scheduler** (default is `None`): the function to update the learning
rate. If defined, it's invoked just as `scheduler.step()`.

**postprocess_method** (default is `'strip_mask'`): the function to
postprocess both predicted and gold labels after model validation (e.g. to
remove labels of masked data). Allowed values are: `'strip_mask'`,
`'strip_mask_bert'` or the callable object implementin the syntax: `preds,
golds = postprocess_method(<predicted labels>, <gold labels>, batch)`.

**control_metric** (of `str` type; default is `'accuracy'`): the metric to
control the model performance in the validation time. The vaues allowed
are: `'loss'`, `'accuracy'`, `'precision'`, `'recall'`, `'f1'`.

**save_ckpt_method** (default is `None`): the function to save the best
model. Called every time as the model performance get better. Invoked as
`save_ckpt_method(model, save_dir)`. If `None`, the standard method of the
`Trainer` class is used.

**output_indent** (default is `4`: just for formatting the output.

**log_file** (default is `sys.stdout`): where to print training progress
messages.

#### The trainer

```pyton
from junky.trainer import Trainer

trainer = Trainer(config, model, train_dataloader, test_dataloader=None,
                  force_cpu=False)
```

Args:

**config**: an instance of the `TrainerConfig` class or the dict that
contains initialization data for it.

**model**: the model to train.

**train_dataloader**, **test_dataloader**: instances of the
`torch.utils.data.DataLoader` classes delivered data for training and
validation steps.

**force_cpu**: if `False` (default), the **model** and batches will be
transfered to the `torch.cuda.current_device()`. So, don't forget to set
default device with torch.cuda.set_device(\<device>) before create
the instance of the `Trainer` class. If **force_cpu** is `True` the
**model** and batches are remained on the CPU during training.

### Examples:

1. The regular model:

```python
from junky.trainer TrainerConfig, Trainer
import torch


DEVICE = 'cuda:0'
MODEL_CKPT_PATH = 'model'

###########
# the part with datasets and model creation is omitted
###########

torch.cuda.set_device(DEVICE)

trainer_config = TrainerConfig(
    MODEL_CKPT_PATH, batch_lens_idx=-1, batch_labels_idx=2,
    model_args=[0, 1], model_kwargs={'labels': 2},
    output_logits_idx=0, output_loss_idx=1, grad_norm_clip=None,
    optimizer='Adam', scheduler=None, postprocess_method='strip_mask'
)
trainer = Trainer(trainer_config, model, train_dataloader,
                  test_dataloader=test_dataloader)
res = trainer.train()
```

1. The BERT fine-tuning:

```python
from junky.trainer TrainerConfig, Trainer
import torch
from transformers import BertForTokenClassification, BertTokenizer


DEVICE = 'cuda:0'
EPOCHS = 3
EMB_MODEL_NAME, EMB_CKPT_PATH = 'bert-base-multilingual-cased', 'emb_model'

###########
# the part with datasets creation is omitted
###########

torch.cuda.set_device(DEVICE)

emb_config = BertForTokenClassification.from_pretrained(
    EMB_MODEL_NAME, id2label=y2t, label2id=t2y,
    output_hidden_states=True, output_attentions=False
)
emb_model = BertForTokenClassification.from_pretrained(EMB_MODEL_NAME,
                                                       config=emb_config)

def create_optimizer(full_tuning=True):
    if full_tuning:
        param_optimizer = list(emb_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                              if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': .01},
            {'params': [p for n, p in param_optimizer
                              if any(nd in n for nd in no_decay)],
             'weight_decay_rate': .0}
        ]
    else:
        param_optimizer = list(emb_model.classifier.named_parameters())
        optimizer_grouped_parameters = \
            [{'params': [p for n, p in param_optimizer]}]

    return AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

emb_optimizer = create_optimizer()
emb_scheduler = get_linear_schedule_with_warmup(
    emb_optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * EPOCHS
)
trainer_config = TrainerConfig(
    EMB_CKPT_PATH, epochs=EPOCHS, batch_lens_idx=1, batch_labels_idx=2,
    model_args=[0, 1], model_kwargs={'labels': 2},
    output_logits_idx=0, output_loss_idx=1, grad_norm_clip=1.,
    optimizer=emb_optimizer, scheduler=emb_scheduler,
    postprocess_method = 'strip_mask_bert', save_ckpt_method=\
        lambda model, ckpt_path: model.save_pretrained(ckpt_path)
)
trainer = Trainer(trainer_config, emb_model, train_dataloader,
                  test_dataloader=test_dataloader)
res = trainer.train()
```
