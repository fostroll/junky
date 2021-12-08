import itertools
import json
import junky
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, \
                            precision_score, recall_score
import time
import torch
from tqdm import tqdm
import sys


class BaseConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k), f'ERROR: Unknown config attribute {k}'
            setattr(self, k, v)

    def as_dict(self):
        return {x: y for x, y in ((x, getattr(self, x)) for x in dir(self)
                                      if not x.startswith('__'))
                    if not callable(y)}

    def save(self, path):
        with open(path, 'wt', encoding='utf-8') as f:
            print(json.dumps(self.as_dict(), sort_keys=True, indent=4),
                  file=f)

    @classmethod
    def load(cls, path):
        with open(path, 'rt', encoding='utf-8') as f:
            config = json.loads(f.read())
        return cls(**config)


class TrainerConfig(BaseConfig):
    """
    The configurator for the `Trainer` class.

    Args:

    **save_dir** (`str`): the directory where to save the best model.

    **save_prefix** (`str`; default is ''): the prefix for the model file
    names.

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

    ***NB:*** If both **model_args** and **model_kwargs** are `None`,
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

    **epoch_steps** (default is `1`): the number of passes for each epoch. It
    has meaning when train dataloader is a function that reload dataset before
    yielding batches.

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

    **max_grad_norm** (default is `None`): if defined, we clip gradient norm
    of the model's parameters to that value.

    **criterion** (default is `None`): the function to calculate the loss. If
    `None`, we suppose that the model in the train mode calculates the loss by
    itself.

    **optimizer** (default is `'SGD'`): the function to update the model's
    parameters. Allowed values are: `'Adam'`, `'AdamW'`, `'SGD'` or instance
    of the `torch.optim.Optimizer` class.

    **scheduler** (default is `None`): the function to update the learning
    rate. If defined, it's invoked just as `scheduler.step()`.

    **postprocess_method** (default is `None`): the function to postprocess
    both predicted and gold labels after model validation (e.g. to remove
    labels of masked data). Allowed values are: `None` (no postprocessing),
    `'strip_mask'`, `'strip_mask_bert'` or the callable object implementing
    the syntax: `preds, golds = postprocess_method(<predicted labels>, <gold
    labels>, batch)`.

    **control_metric** (default is `'accuracy'`): the metric to control the
    model performance in the validation time. The vaues allowed are: `'loss'`,
    `'accuracy'`, `'precision'`, `'recall'`, `'f1'`.

    **save_ckpt_method** (`callable`; default is `None`): the function to save
    the best model. Called every time as the model performance get better.
    Invoked as `save_ckpt_method(model, save_dir)`. If `None`, the standard
    method of the `Trainer` class is used.

    **binary_threshold** (float; default is `None`): a value between `0` and
    `1` specifying the threshold of rounding for binary classification. Note
    that in this case, the final operation of the model in the `eval` mode
    must be `torch.sigmoid()`.

    **output_indent** (default is `4`): just for formatting the output.

    **loss_ma_batches** (int; default is `None`): the number of batches to
    average the loss while the progress indication. `None` means unlimited.

    **loss_ema_smoothing** (int; default is `2`): the "smoothing" parameter
    for the EMA calculation if **loss_ma_method** is `'EMA'`.

    **loss_ma_method** (default is `'EMA'`): the method to average the loss
    while progress indication. The values allowed are: `'EMA'`, `'SMA'`.

    **log_file** (default is `sys.stdout`): where to print training progress
    messages.
    """
    save_dir = None
    save_prefix = ''

    parallel = False  # not implemented yet

    batch_labels_idx = -1
    batch_lens_idx = 2
    model_args = None
    model_kwargs = None

    output_logits_idx = 0
    output_loss_idx = 1
    min_epochs = 0
    max_epochs = None
    bad_epochs = 5
    epoch_steps = 1

    adam_lr, adam_betas, adam_eps, adam_weight_decay, adam_amsgrad = \
        .0001, (0.9, 0.999), 1e-08, 0, False
    sgd_lr, sgd_momentum, sgd_weight_decay, sgd_dampening, sgd_nesterov = \
        .001, .9, 0, 0, False
    adamw_lr, adamw_betas, adamw_eps, adamw_weight_decay, adamw_amsgrad = \
        5e-5, (0.9, 0.999), 1e-8, 0.01, False

    max_grad_norm = None
    criterion = None
    optimizer = 'SGD'  # 'Adam', 'AdamW', 'SGD'
    scheduler = None
    postprocess_method = None
    control_metric = 'accuracy'
        # 'loss', 'accuracy', 'precision', 'recall', 'f1'
    save_ckpt_method = None

    binary_threshold = .5

    output_indent = 4
    loss_ma_batches = None  # `None` means inlimited
    loss_ema_smoothing = 2
    loss_ma_method = 'EMA'  # 'EMA', 'SMA'
    log_file = sys.stdout

    def __init__(self, save_dir, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir


class Trainer():
    """
    The model trainer.

    Args:

    **config**: an instance of the `TrainerConfig` class or the dict that
    contains initialization data for it.

    **model**: the model to train.

    **train_dataloader**, **test_dataloader**: instances of the
    `torch.utils.data.DataLoader` classes delivered data for training and
    validation steps. Also, it is possible to pass `callable` to either of
    this params, in which case it should return `torch.utils.data.DataLoader`
    instance. This `callable` will be called before starting each epoch step
    and can take three params: `split` that can be either `'train'` or
    '`test`', `epoch` that is the current epoch number (note that the number
    of the 1st epoch is `1`), and `step`. If `config.epoch_steps` is `1`,
    `step` is always `1`.

    **device**: if not `None`, the **model** will be transfered to the
    specified device.
    """
    def __init__(self, config, model, train_dataloader, test_dataloader=None,
                 device=False):
        self.config = TrainerConfig(**config) \
                          if isinstance(config, dict) else \
                      config
        assert test_dataloader is not None or \
               (train_dataloader is not None and config.max_epochs), \
            'ERROR: Either `test_dataloaders` of `train_dataloader` with ' \
            '`config.max_epochs` param must be defined.'
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if config.parallel:
            self.model = torch.nn.DataParallel(self.model)
#             self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        if device:
            self.model.cpu().to(device)
        self.device = next(self.model.parameters()).device

    def save_ckpt(self):
        config = self.config
        save_dir, save_prefix = config.save_dir, config.save_prefix

        # Take care of distributed/parallel training
        model = self.model.module if hasattr(self.model, 'module') else \
                self.model

        print('Saving checkpoint to {}'.format(save_dir),
              file=config.log_file)

        if config.save_ckpt_method:
            kwargs = {'save_prefix': save_prefix} if save_prefix else {}
            config.save_ckpt_method(model, save_dir, **kwargs)
        else:
            # Create output directory if needed
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(os.path.join(save_dir, save_prefix + 'config.json'),
                      'wt') as f:
                print(json.dumps(model.config, sort_keys=True, indent=4),
                      file=f)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, save_prefix + 'state_dict.pt'),
                       pickle_protocol=2)

        # Good practice: save your training arguments together
        # with the trained model:
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    def postprocess_strip_mask(self, preds, golds, batch):
        lens = batch[self.config.batch_lens_idx]
        preds = [x[:y] for x, y in zip(preds, lens)]
        golds = [x[:y] for x, y in zip(golds, lens)]
        return preds, golds

    def postprocess_strip_mask_bert(self, preds, golds, batch):
        lens = batch[self.config.batch_lens_idx]
        preds = [x[1:y + 1] for x, y in zip(preds, lens)]
        golds = [x[1:y + 1] for x, y in zip(golds, lens)]
        return preds, golds

    def train(self, best_score=True):
        """The method that run the training.

        Args:

        **best_score** (`float|bool`, default is `True`): the starting point
        to compare the calculating control metric with. If `True`, the initial
        performance of the model will be measured and used as the best score.
        Thus, we will not save the trained model if it's worse than untrained.
        `None` of `False` means, don't check it.
        """
        config, model = self.config, self.model
        min_epochs, max_epochs, bad_epochs  = \
            config.min_epochs, config.max_epochs, config.bad_epochs
        epoch_steps = config.epoch_steps

        batch_labels_idx = config.batch_labels_idx
        model_args, model_kwargs = \
            config.model_args or [], config.model_kwargs or {}
        batch_payload_len = len(model_args) + len(model_kwargs)
        output_loss_idx, output_logits_idx = \
            config.output_loss_idx, config.output_logits_idx

        max_grad_norm = config.max_grad_norm
        criterion = config.criterion
        optimizer = config.optimizer
        scheduler = config.scheduler
        postprocess_method = config.postprocess_method
        control_metric = config.control_metric

        binary_threshold = config.binary_threshold

        if isinstance(optimizer, str):
            # Take care of distributed/parallel training
            raw_model = self.model.module \
                            if hasattr(self.model, 'module') else \
                        self.model
            optimizer = optimizer.lower()
            if optimizer == 'adam':
                optimizer = torch.optim.Adam(
                    params=raw_model.parameters(),
                    lr=config.adam_lr,
                    betas=config.adam_betas,
                    eps=config.adam_eps,
                    weight_decay=config.adam_weight_decay,
                    amsgrad=config.adam_amsgrad
                )
            elif optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    params=raw_model.parameters(),
                    lr=config.sgd_lr,
                    momentum=config.sgd_momentum,
                    weight_decay=config.sgd_weight_decay,
                    dampening=config.sgd_dampening,
                    nesterov=config.sgd_nesterov
                )
            elif optimizer == 'adamw':
                optimizer = torch.optim.AdamW(
                    params=raw_model.parameters(),
                    lr=config.adamw_lr,
                    betas=config.adamw_betas,
                    eps=config.adamw_eps,
                    weight_decay=config.adamw_weight_decay,
                    amsgrad=config.adamw_amsgrad
                )
            else:
                raise ValueError(f'ERROR: unknown optimizer "{optimizer}".')

        if isinstance(postprocess_method, str):
            postprocess_method = postprocess_method.lower()
            if postprocess_method == 'strip_mask':
                postprocess_method = self.postprocess_strip_mask
            elif postprocess_method == 'strip_mask_bert':
                postprocess_method = self.postprocess_strip_mask_bert
            else:
                raise ValueError('ERROR: unknown postprocess method '
                                f'"{postprocess_method}".')

        print_indent = ' ' * config.output_indent
        log_file = config.log_file

        def run_epoch(split, epoch, step=0):
            assert split in ['train', 'test']
            is_train = split == 'train'
            model.train(is_train)
            dataloader = self.train_dataloader if is_train else \
                         self.test_dataloader
            if callable(dataloader):
                dataloader = dataloader(split, epoch, step)

            preds, golds, losses = [], [], []
            step_ = f'.{step}' if step else ''
            pbar = tqdm(dataloader, total=len(dataloader),
                        desc=f'Epoch {epoch}{step_}' if is_train else
                             f'Epoch {epoch} test',
                        mininterval=2, file=log_file)

            loss_sum = 0
            loss_ma_method, loss_ma_batches = \
                config.loss_ma_method, config.loss_ma_batches
            K = config.loss_ema_smoothing / (1 + loss_ma_batches) \
                    if loss_ma_batches else \
                None
            for batch_no, batch in enumerate(pbar, start=1):
                batch = [x.to(self.device, non_blocking=True)
                             if isinstance(x, torch.Tensor) else
                         x
                             for x in batch]

                # forward the model
                with torch.set_grad_enabled(is_train):
                    output = model(*batch) if not batch_payload_len else \
                             model(*[batch[x] for x in model_args],
                                   **{x: batch[y]
                                          for x, y in model_kwargs.items()})
                    # .flatten() is in multilabel case
                    logits, loss = \
                        output[output_logits_idx], output[output_loss_idx]
                    losses.append(loss.item())

                if is_train:
                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_grad_norm)
                    # update parameters
                    optimizer.step()
                    # update the learning rate
                    if scheduler:
                        scheduler.step()

                    # report progress
#                     pbar.set_postfix(train_loss=loss.item(), refresh=False)
                    #EMA = loss.item() * K + EMA * (1 - K)
                    if not loss_ma_batches or batch_no <= loss_ma_batches:
                        loss_sum += losses[-1]
                        MA = loss_sum / batch_no
                    elif loss_ma_method == 'SMA':
                        loss_sum += losses[-1] - losses[-1 - loss_ma_batches]
                        MA = loss_sum / loss_ma_batches
                    elif loss_ma_method == 'EMA':
                        MA += K * (losses[-1] - MA)
                    else:
                        raise ValueError('ERROR: Unknown loss_ma_method '
                                        f'"{loss_ma_method}".')
                    pbar.set_postfix(train_loss_MA=MA, refresh=False)

                else:
                    golds_ = batch[batch_labels_idx]
                    logits.detach_()
                    preds_ = (logits >= binary_threshold).int() \
                                 if len(golds_.shape) \
                                 == len(logits.shape) else \
                             logits.max(dim=-1)[1]

                    if postprocess_method:
                        preds_, golds_ = \
                            postprocess_method(preds_, golds_, batch)
                    preds.extend(preds_)
                    golds.extend(golds_)

            loss = float(np.mean(losses))
            if not is_train:
                preds = torch.hstack(preds).cpu().numpy()
                golds = torch.hstack(golds).cpu().numpy()
            return loss if is_train else (loss, preds, golds)

        best_epoch = None
        if best_score in [None, False]:
            best_score = float('-inf')
        prev_score = best_score
        score = None

        train_losses, test_losses = [], []
        best_test_preds, best_test_golds = [], []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        bad_epochs_ = 0
        start_time = time.time()

        with_train = self.train_dataloader is not None
        with_test = self.test_dataloader is not None
        best_loss = prev_loss = float('inf')
        test_loss = None
        test_first = best_score is True
        for epoch in range(not test_first, max_epochs + 1) \
                         if max_epochs else \
                     itertools.count(start=not test_first):
            print_str = f'{print_indent}Losses: '

            if with_train and not test_first:
                need_backup = True
                if epoch_steps > 1:
                    train_loss = 0
                    for step in range(1, epoch_steps + 1):
                        train_loss_ = run_epoch('train', epoch, step=step)
                        train_loss += train_loss_
                    train_loss /= epoch_steps
                else:
                    train_loss = run_epoch('train', epoch)
                train_losses.append(train_loss)
                print_str = f'train = {train_loss:.8f}'
            else:
                need_backup = False

            if with_test:
                test_loss, test_preds, test_golds = run_epoch('test', epoch)
                test_losses.append(test_loss)

                accuracy = accuracy_score(test_golds, test_preds)
                precision = \
                    precision_score(test_golds, test_preds, average='macro')
                recall = recall_score(test_golds, test_preds, average='macro')
                f1 = f1_score(test_golds, test_preds, average='macro')

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

                score = -test_loss if control_metric == 'loss' else \
                        accuracy if control_metric == 'accuracy' else \
                        precision if control_metric == 'precision' else \
                        recall if control_metric == 'recall' else \
                        f1 if control_metric == 'f1' else \
                        None

                if with_train and not test_first:
                    print_str += ', '
                print_str += \
                    'test = {:.8f}\n'.format(test_loss) \
                  + '{}Test: accuracy = {:.8f}\n'.format(print_indent,
                                                         accuracy) \
                  + '{}Test: precision = {:.8f}\n' \
                        .format(print_indent, precision) \
                  + '{}Test: recall = {:.8f}\n'.format(print_indent, recall) \
                  + '{}Test: f1_score = {:.8f}'.format(print_indent, f1)

                if not with_train:
                    break
                if test_first or score > best_score:
                    test_first = False
                    best_score = score
                    best_epoch = epoch
                    best_test_golds, best_test_preds = \
                        test_golds[:], test_preds[:]
                    bad_epochs_ = 0
                    print_str += \
                        '\nnew maximum score {:.8f}'.format(score)
                else:
                    need_backup = False
                    if score <= prev_score:
                        bad_epochs_ += 1
                    sgn = '{} {}'.format('==' if score == best_score else
                                         '<<',
                                         '<' if score < prev_score else
                                         '=' if score == prev_score else
                                         '>')
                    print_str += '\nBAD EPOCHS: {} ({})' \
                                     .format(bad_epochs_, sgn)
                    if bad_epochs_ >= bad_epochs \
                   and epoch >= min_epochs:
                        print_str += '\nMaximum bad epochs exceeded. ' \
                                     'Process has been stopped. ' \
                                   + ('No models could surpass '
                                     f'`best_score={best_score}` given'
                                          if best_epoch is None else
                                     f'Best score {best_score} '
                                     f'(on epoch {best_epoch})')
                        break

                print(print_str, file=log_file)
                log_file.flush()
                prev_score = score
                print_str = ''

            if need_backup:
                self.save_ckpt()

            if epoch == max_epochs:
                print_str = 'Maximum epochs exceeded. ' \
                            'Process has been stopped. ' \
                          + (f'No models could surpass '
                            f'`best_score={best_score}` given'
                                 if best_epoch is None else
                            f'Best score {best_score} '
                            f'(on epoch {best_epoch})')

        if print_str:
            print(print_str, file=log_file)
        print('Elapsed time: {}'
                  .format(junky.seconds_to_strtime(time.time() - start_time)),
              file=log_file)
        log_file.flush()

        return {'best_epoch': best_epoch,
                'best_score': best_score,
                'best_test_golds': best_test_golds,
                'best_test_preds': best_test_preds,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies,
                'precisions': precisions,
                'recalls': recalls,
                'f1s': f1s}
