import torch
import sys


class TrainerConfig():
    """
    The configurator for the `Trainer` class

    Args:

    **save_dir**: the directory where to save the best model.

    **\*args**: any of the params defined below.

    """
    parallel = False  # not implemented yet

    batch_labels_idx = -1  # labels position in the batch received from loader
    batch_lens_idx = 2     # lengthts of sentences in the batch. Used only in
        # the postprocessing, so it will be ignored if postprocess doesn't
        # apply (see the `Trainer` class description).
    model_args = None      # positional arguments of model.forward() method.
        # It must be indices of the batch positions
    model_kwargs = {}      # keyword arguments of model.forward() method in
        # train mode. It's the dict of kwarg names and corresponding batch
        # positions
    # Example: model_args=[0, 1], model_kwargs={'labels': 2} results in
    # invoking model(batch[0], batch[1], labels=batch[2]). Before this, all
    # the batch will be moved to the model's device.
    ############
    output_logits_idx = 0  # if model.forward() returns a tuple, it's the
        # position of logits in that tuple
    output_loss_idx = 1    # if model.forward() calculates a loss in train
        # mode by itself, it's the position of the loss in the returning tuple

    min_epochs = 0     # the number of epochs we continue training even if
        # the number of bad epochs surpassed the bad_epochs param
    max_epochs = None  # the total number of epochs can't be greater than this
        # value
    bad_epochs = 5     # we stop training if the control_metric doesn't
        # increase on the validation set for a period of this number of epochs
        # (really, the algorithm is slightly more complex but the meaning is
        # it)

    adam_lr, adam_betas, adam_eps, adam_weight_decay, adam_amsgrad = \
        .0001, (0.9, 0.999), 1e-08, 0, False
    sgd_lr, sgd_momentum, sgd_weight_decay, sgd_dampening, sgd_nesterov = \
        .001, .9, 0, 0, False
    adamw_lr, adamw_betas, adamw_eps, adamw_weight_decay, adamw_amsgrad = \
        5e-5, (0.9, 0.999), 1e-8, 0.01, False
    optimizer = 'SGD'  # allowed values are: 'Adam', 'AdamW', 'SGD'

    grad_norm_clip = None
    control_metric = 'accuracy'  # the metric to control the model performance
        # in the validation time. The vaues allowed are: 'loss', 'accuracy',
        # 'precision', 'recall', 'f1'
    log_file = sys.stdout
    output_indent = 4  # just to formatting the output

    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        for k, v in kwargs.items():
            assert hasattr(self, k), f'ERROR: Unknown config attribute {k}'
            setattr(self, k, v)


class Trainer():
    """
    The model trainer.

    Args:

    **config**: an instance of the `TrainerConfig` class.

    **model**: the model to train.

    **train_dataloader**, **test_dataloader**: instances of the
    `torch.utils.data.DataLoader` classes delivered data for training and
    validation steps.

    **criterion**: the function to calculate the loss. If omitted, we suppose
    that the `model` in train mode calculates the loss by itself.

    **optimizer**: the function to update the **model**'s parameters.
    Additionally, values of the `str` type are allowed as described in the
    `TrainerConfig` class definition. If `None`, the value from **config** is
    used.

    **scheduler**: the function to update the learning rate. If defined, it's
    invoked just as `scheduler.step()`.

    **save_ckpt_method**: the function to save the best **model**. Called
    every time as the **model** performance get better. Invoked as
    `save_ckpt_method(model, save_dir)` where the `save_dir` param must be
    defined in the **config**.

    **force_cpu**: if `False` (default), the **model** and batches will be
    transfered to the `torch.cuda.current_device()`. So, don't forget to set
    default device with torch.cuda.set_device(\<device>) before create
    the instance of the `Trainer` class. If **force_cpu** is `True` the
    **model** and batches are remained on the CPU during training.
    """
    def __init__(self, config, model, train_dataloader, test_dataloader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 postprocess_method=None, save_ckpt_method=None,
                 force_cpu=False):
        self.config = config
        assert test_dataloader or config.max_epochs, \
            'ERROR: Either `test_dataloaders` of `config.max_epochs` param' \
            'must be defined'
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer or config.optimizer
        self.scheduler = scheduler
        self.save_ckpt_method = save_ckpt_method
        self.postprocess_method = \
            self.postprocess_strip_mask \
                if postprocess_method == 'strip_mask' else \
            self.postprocess_strip_mask_bert \
                if postprocess_method == 'strip_mask_bert' else \
            postprocess_method

        self.device = 'cpu'
        if torch.cuda.is_available() and not force_cpu:
            self.device = torch.cuda.current_device()
        if config.parallel:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
#                 self.model = torch.nn.parallel.DistributedDataParallel(self.model).to(self.device)
        else:
            self.model.to(self.device)

    def save_ckpt(self):
        config = self.config
        save_dir = config.save_dir

        # Take care of distributed/parallel training
        model = self.model.module if hasattr(self.model, 'module') else \
                self.model

        # Create output directory if needed
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Saving checkpoint to {}'.format(save_dir),
              file=config.log_file)

        if self.save_ckpt_method:
            self.save_ckpt_method(model, save_dir)
        else:
            with open(os.path.join(save_dir, 'config.json'), 'wt') as f:
                print(json.dumps(model.config, sort_keys=True, indent=4), file=f)
            torch.save(model.state_dict(), os.path.join(save_dir, 'state_dict.pt'),
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

    def train(self, best_score=None):
        config, model = self.config, self.model
        batch_labels_idx = config.batch_labels_idx
        model_args, model_kwargs = config.model_args, config.model_kwargs
        batch_payload_len = 0 if model_args is None else \
                            (len(model_args) + len(model_kwargs))
        output_loss_idx, output_logits_idx = \
            config.output_loss_idx, config.output_logits_idx
        log_file = config.log_file

        optimizer = self.optimizer
        if isinstance(optimizer, str):
            # Take care of distributed/parallel training
            raw_model = self.model.module if hasattr(self.model, 'module') else \
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
                raise ValueError('ERROR: unknown optimizer '
                                f'{config.optimizer}')

        scheduler = self.scheduler
        control_metric = config.control_metric
        print_indent = ' ' * config.output_indent

        def run_epoch(split, epoch):
            assert split in ['train', 'test']
            is_train = split == 'train'
            model.train(is_train)
            dataloader = self.train_dataloader if is_train else \
                         self.test_dataloader

            preds, golds, losses = [], [], []
            pbar = tqdm(dataloader, total=len(dataloader),
                        desc='Epoch {}'.format(epoch),
                        mininterval=2, file=log_file) if is_train else \
                   dataloader

            EMA, K = 0, 2 / (10 - 1)
            for batch in pbar:
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
                    if config.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       config.grad_norm_clip)
                    # update parameters
                    optimizer.step()
                    # update the learning rate
                    if scheduler:
                        scheduler.step()

                    # report progress
#                     pbar.set_postfix(train_loss=loss.item(), refresh=False)
                    EMA = loss.item() * K + EMA * (1 - K)
                    pbar.set_postfix(train_loss_EMA=EMA, refresh=False)

                else:
                    preds_ = logits.detach().max(dim=-1)[1]
                    golds_ = batch[batch_labels_idx]
                    if self.postprocess_method:
                        preds_, golds_ = \
                            self.postprocess_method(preds_, golds_, batch)
                    preds.extend(preds_)
                    golds.extend(golds_)

            loss = float(np.mean(losses))
            if not is_train:
                preds = torch.hstack(preds).cpu().numpy()
                golds = torch.hstack(golds).cpu().numpy()
            return loss if is_train else (loss, preds, golds)

        best_epoch = None
        if best_score is None:
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

        best_loss = prev_loss = float('inf')
        test_loss = None
#         self.tokens = 0 # counter used for learning rate decay
        for epoch in range(1, config.max_epochs + 1) if config.epochs else \
                     itertools.count(start=1):
            print_str = 'Epoch {}: \n'.format(epoch)
            need_backup = True

            train_loss = run_epoch('train', epoch)
            train_losses.append(train_loss)
            print_str += '{}Losses: train = {:.8f}' \
                             .format(print_indent, train_loss)
            if self.test_dataloader is not None:
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

                print_str += \
                    ', test = {:.8f}\n'.format(test_loss) \
                  + '{}Test: accuracy = {:.8f}\n'.format(print_indent,
                                                         accuracy) \
                  + '{}Test: precision = {:.8f}\n' \
                        .format(print_indent, precision) \
                  + '{}Test: recall = {:.8f}\n'.format(print_indent, recall) \
                  + '{}Test: f1_score = {:.8f}'.format(print_indent, f1)

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_test_golds, best_test_preds = test_golds[:], test_preds[:]
                    bad_epochs_ = 0
                    print_str += '\nnew maximum score {:.8f}'.format(score)
                else:
                    need_backup = False
                    if score <= prev_score:
                        bad_epochs_ += 1
                    sgn = '{} {}'.format('==' if score == best_score else '<<',
                                         '<' if score < prev_score else
                                         '=' if score == prev_score else
                                         '>')
                    print_str += '\nBAD EPOCHS: {} ({})' \
                                     .format(bad_epochs_, sgn)
                    if bad_epochs_ >= config.bad_epochs \
                   and epoch >= config.min_epochs:
                        print_str += \
                            '\nMaximum bad epochs exceeded. ' \
                            'Process has been stopped. ' \
                          + ('No models could surpass `best_score={}` given'
                                 if best_epoch is None else
                             'Best score {} (on epoch {})').format(best_score,
                                                                   best_epoch)
                        break

                print(print_str, file=log_file)
                log_file.flush()
                prev_score = score
                print_str = ''

            if need_backup:
                self.save_ckpt()

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
