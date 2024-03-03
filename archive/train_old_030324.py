import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def unpack_batch(batch, input_key = "input_ids", label_key = "labels"):
    if isinstance(batch, tuple):
        return batch[:2]
    elif isinstance(batch, dict):
        inputs = batch[input_key]
        labels = batch[label_key] if label_key in batch else None
        return inputs, labels
    else:
        return batch, None

## Forward functions

def default_forward_batch(model, batch):
    x, _ = unpack_batch(batch)
    x = x.to(model.device)
    logits = model(x)
    return logits

def multiple_choice_forward_batch(model, batch):
    input_ids, _ = unpack_batch(batch)
    input_ids = input_ids.to(model.device)
    logits = []
    for i in range(input_ids.shape[1]):
        outputs = model(input_ids=input_ids[:, i, :].squeeze(1))
        logits.append(outputs)
    logits = torch.stack(logits, dim=1)
    #print(logits.shape)
    return logits 


## Loss functions
# let's try an alternate version using the more common loss object
loss_object = nn.CrossEntropyLoss()
def masked_cross_entropy_loss(batch, outputs, pad_token_id = -100):
    x, labels = unpack_batch(batch)
    if labels is None:
        labels = x[..., 1:]
        logits = outputs[..., :-1, :]
    else:
        logits = outputs

    labels = labels.to(logits.device)    
    #loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=pad_token_id) # this is a more sensible way of doing it but for some reason it's slow as heck.
    loss_object.ignore_index = pad_token_id
    loss = loss_object(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss

def multiple_choice_loss(batch, outputs, pad_token_id = -100):
    # The only difference here is is that "labels" are the answers, not the next tokens, so we need to pull the next tokens from the input.
    x, _ = unpack_batch(batch)
    labels = x[..., 1:]
    logits = outputs[..., :-1, :]
    labels = labels.to(logits.device)    
    loss_object.ignore_index = pad_token_id
    loss = loss_object(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss


## Learning rate schedules
from torch.optim.lr_scheduler import LambdaLR

## These are awkward; they are basically going to be functions that return functions that wrap functions, because we need to delay choosing the optimizer and number of steps
def constant_schedule(optimizer, total_steps):
    return LambdaLR(optimizer, lambda step: 1.0)

def get_linear_schedule(end_factor = 0.1, warmup_steps = 2000, end_step = None):
    def _f(optimizer, total_steps):
        nonlocal end_step
        if end_step is None:
            end_step = total_steps
        def _g(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return end_factor + (1.0 - end_factor) * (1.0 - (step - warmup_steps) / (end_step - warmup_steps))
        return LambdaLR(optimizer, _g)
    return _f

## Metric functions

# checks whether all unmasked tokens are correct, e.g. for LAMBADA.
def masked_answer_accuracy(batch, outputs, pad_token_id = -100):
    logits = outputs.detach()
    _, labels = unpack_batch(batch)
    labels = labels.detach().to(logits.device)
    preds = logits.argmax(dim=-1).to(logits.device)
    preds = torch.where(labels != pad_token_id, preds, torch.tensor(pad_token_id))
    acc = (preds == labels).float()
    acc = acc.prod(dim=-1)
    acc = acc.mean()
    return acc.item()

# https://github.com/EleutherAI/lm-evaluation-harness/issues/539
# compares the logprob choices, e.g. for HellaSwag
import torch.nn.functional as F
def multiple_choice_logprob_accuracy(batch, outputs, pad_token_id = -100):
    input_ids, labels = unpack_batch(batch)
    input_ids = input_ids.detach().to(outputs.device)
    lm_logits = outputs.detach()[:, :, :-1, :]
    lm_labels = input_ids[:, :, 1:]
    mask = (lm_labels != pad_token_id)
    log_probs = F.log_softmax(lm_logits, dim=-1)   
    log_probs = torch.gather(log_probs, -1, lm_labels.unsqueeze(-1)).squeeze(-1) # verified by hand that this works
    masked = log_probs * mask
    sum_log_probs = masked.sum(dim=-1)
    # I tried exponentiating and also normalizing by length, but both made the results worse...this I think is what people recommend.
    preds = sum_log_probs.argmax(dim=-1)
    labels = labels.detach().to(preds.device)
    acc = (preds == labels).float().mean()
    return acc.item()


def padded_token_counter(batch, outputs, pad_token_id = -100):
    x, _ = unpack_batch(batch).to(outputs.device)
    x = x.detach() # I don't think this is actually necessary
    mask = (x != pad_token_id)
    count = mask.sum().item()
    return count


## Logging functions
def print_log_row(row): # simple default
    print(row)


import os
import csv
def make_csv_logger(fname, also_print = True, resume_existing = False, overwrite_existing = False):
    # make sure the file name looks right
    if not fname.endswith(".csv"):
        fname = fname + ".csv"
    # make sure the file doesn't already exist
    if os.path.exists(fname):
        if resume_existing:
            print(f"Resuming logging to {fname}.")
        elif overwrite_existing:
            print(f"Overwriting {fname}.")
            os.remove(fname)
        else:
            raise ValueError(f"File {fname} already exists.  Set resume_existing or overwrite_existing to True to proceed.")
    # define a logging function
    def _f(row):
        if also_print:
            print(row)
        # create the file and give it the right header if it doesn't exist
        if not os.path.exists(fname):
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row.keys())
        # append the row to the file
        with open(fname, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row.values())
    # return the function
    return _f


## Logger class
class MetricsLogger():
    def __init__(self, metrics, report_log_row = print_log_row, pad_token_id = -100):
        self.metrics = metrics
        self.batch_metrics = {k: [] for k in metrics}
        self.batch_metrics["loss"] = []
        self.log = []
        self.report_log_row = report_log_row
        self.logging_timestamp = None
        self.epoch = 0
        self.pad_token_id = pad_token_id

    def accumulate_batch_metrics(self, batch, outputs, loss_item): # everything should be back on CPU At this point.
        self.batch_metrics["loss"].append(loss_item)
        # We could count tokens here, but only if we know the padding token.
        for k, v in self.metrics.items():
            self.batch_metrics[k].append(v(batch, outputs, pad_token_id = self.pad_token_id)) # these should have already been converted to scalars.


    def log_metrics(self, mode, step): # everything should be back on CPU at this point.
        row = {}
        row["mode"] = mode
        row["epoch"] = self.epoch
        row["step"] = step
        current_timestamp = time.time()
        row["seconds"] = current_timestamp - self.logging_timestamp
        if len(self.log) == 0:
            row["total_s"] = row["seconds"]
        else:
            row["total_s"] = self.log[-1]["total_s"] + row["seconds"]
        row["loss"] = sum(self.batch_metrics["loss"]) / len(self.batch_metrics["loss"])
        row["ppl"] = torch.exp(torch.tensor(row["loss"])).item()
        for k, v in self.metrics.items():
            row[k] = sum(self.batch_metrics[k]) / len(self.batch_metrics[k])
        # report and add the row to the log
        self.report_log_row(row)
        self.log.append(row)
        for k in self.batch_metrics:
            self.batch_metrics[k] = []

## Evaluator class
class Evaluator():
    def __init__(
        self,
        model,
        eval_loader,
        device = torch.device("cpu"),
        forward_batch = default_forward_batch,
        batch_loss = masked_cross_entropy_loss,
        metrics = {},
        report_log_row = print_log_row,
        metrics_logger = None, # if None, will be created automatically
        tokenizer = None,
        pad_token_id = -100
    ):
        self.device = device
        self.model = model.to(device) # troubleshooting performance
        self.eval_loader = eval_loader
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        self.tokenizer = tokenizer
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = pad_token_id
        if self.pad_token_id == -100:
            print("Warning: Evaluator using default pad token ID of -100.  This is probably not what you want.")
        self.logger = metrics_logger if metrics_logger is not None else MetricsLogger(metrics, report_log_row = report_log_row, pad_token_id = self.pad_token_id)
        

    def evaluate(self): # this is such a short function, I don't really see where it could have gone wrong...
        self.model.eval()
        with torch.no_grad():
            self.logger.logging_timestamp = time.time()
            for eval_step, eval_batch in enumerate(self.eval_loader, start=1):
                output = self.forward_batch(self.model, eval_batch)
                loss = self.batch_loss(eval_batch, output, pad_token_id = self.pad_token_id)
                self.logger.accumulate_batch_metrics(eval_batch, output, loss.item())
        # I don't think there's ever a reason to log the evaluation metrics at times other than the end of the evlauation
        self.logger.log_metrics("eval", eval_step)
        self.logger.logging_timestamp = time.time()

def _get_batch_size(batch):
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    elif isinstance(batch, tuple) or isinstance(batch, list):
        for item in batch:
            size = _get_batch_size(item)
            if size is not None:
                return size
    elif isinstance(batch, dict):
        for item in batch.values():
            size = _get_batch_size(item)
            if size is not None:
                return size
    else:
        return None

# import named tuples
from collections import namedtuple
# split a batch into smaller batches.  be prepared for tensors, tuples, or dictionaries, but assume you will reach tensors if you recurse far enough.
import math
def _split_batch(batch, batch_size, new_size): 
    n_splits = math.ceil(_get_batch_size(batch)/ new_size) # Ah...so, this is a bit of a problem.  because some batches could end up being smaller.
    # also, it's hard to grab the size directly because 
    if isinstance(batch, torch.Tensor):
            return torch.split(batch, new_size, dim = 0)
    elif isinstance(batch, tuple) and hasattr(batch, "_fields"):
        # turn it into a list of named tuples of tensors
        split = [_split_batch(item, batch_size, new_size) for item in batch]
        return [namedtuple(batch.__class__.__name__, [f"{k}_{i}" for k in batch._fields]) for i in range(n_splits)]
    elif isinstance(batch, tuple):
        # turn it into a list of tuples of tensors
        split = [_split_batch(item, batch_size, new_size) for item in batch]
        return [tuple([s[i] for s in split]) for i in range(n_splits)]
    elif isinstance(batch, list):
        split = [_split_batch(item, batch_size, new_size) for item in batch]
        return [s[i] for s in split for i in range(n_splits)]
    elif isinstance(batch, dict):
        # turn it into a list of dictionaries of tensors
        split = {k: _split_batch(v, batch_size, new_size) for k, v in batch.items()}
        batch = [{k: v[i] for k, v in split.items()} for i in range(n_splits)]
        return batch
    else:
        # otherwise we just broadcast it
        return [batch for _ in range(n_splits)]

## Trainer class
import gc
import pprint

from torch.nn.utils import clip_grad_norm_

from dataclasses import dataclass # keep it lightweight for now
@dataclass
class TrainerState():
    epoch: int
    step: int
    global_step: int
    should_training_stop: bool

class Trainer():
    def __init__(self,
        model,
        train_loader,
        device = torch.device("cpu"),
        eval_loader = None,
        evaluator = None, # if an evaluation loader was provided, this will be created automatically
        log_every = None, # defaults to once per epoch
        eval_every = None, # defaults to once per epoch
        forward_batch = default_forward_batch,
        batch_loss = masked_cross_entropy_loss,
        report_log_row = print_log_row,
        metrics = {},
        metrics_logger = None, # if report_log_row is not None, this will be created automatically
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        gradient_clipping = 1.0, # set to None to disable
        weight_decay = 0.01,
        schedule = constant_schedule,
        gradient_accumulation_batch_size = None,
        tokenizer = None,
        epochs = 1,
        pad_token_id = -100, # if None and tokenizer is provided, will be inferred from the tokenizer
        callbacks = [],
        allow_tf32 = True,
        set_grad_to_none = True,
        autocast_dtype = torch.bfloat16, # torch.float32 for None, basically.  I'm not 100% sure everything is implemented perfectly but it seems to work.
    ):
        self.callbacks = callbacks
        if isinstance(device, str):
            device = torch.device(device) # I think model.to() already accepts this
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.log_every = log_every
        self.eval_every = eval_every
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        self.tokenizer = tokenizer
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = pad_token_id
        if self.pad_token_id == -100:
            print("Warning: Trainer using default pad token ID of -100.  This is probably not what you want.")
        # metrics and logging
        self.report_log_row = report_log_row
        if self.report_log_row is not None and metrics_logger is None:
            self.logger = MetricsLogger(
                metrics,
                report_log_row = report_log_row,
                pad_token_id = self.pad_token_id
            )
        else:
            self.logger = metrics_logger
        # build an evaluator if necessary
        if evaluator is None and eval_loader is not None:
            self.evaluator = Evaluator(
                model,
                eval_loader,
                device = device,
                forward_batch = forward_batch,
                batch_loss = batch_loss,
                report_log_row = report_log_row,
                metrics_logger = self.logger,
                tokenizer = self.tokenizer,
                pad_token_id = self.pad_token_id
            )
        else:
            self.evaluator = evaluator
        # data types and optimizations
        self.allow_tf32 = allow_tf32
        self.set_grad_to_none = set_grad_to_none # This seems to make no difference but I think it's strictly better
        self.autocast_dtype = autocast_dtype
        self.scaler = torch.cuda.amp.GradScaler(enabled = autocast_dtype != torch.float32) # I *think* it's okay to use a scaler with bf16
        # hyperparameters and training details
        self.optimizer = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)
        self.epochs = epochs
        self.scheduler = schedule(self.optimizer, len(train_loader) * epochs)
        self.gradient_clipping = gradient_clipping
        # schedule could be a lambda, 
        self.gradient_accumulation_batch_size = gradient_accumulation_batch_size if gradient_accumulation_batch_size is not None else train_loader.batch_size
        self.state = TrainerState(
            epoch = 0,
            step = 0,
            global_step = 0,
            should_training_stop = False
        )
        # this will eventually probably contain a lot of things but I'm going to start slow

    def train(self, epochs = None):
        epochs = epochs if epochs is not None else self.epochs
        self._save_allowing_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        for callback in self.callbacks:
            callback.on_train_begin(self)
        try:
            if self.logger is not None:
                start_epoch = self.logger.epoch
                self.logger.logging_timestamp = time.time()
            else:
                start_epoch = 0
            print(f"Training for {epochs} epochs starting from epoch {start_epoch + 1}; {len(self.train_loader)} steps per epoch.")
            for epoch in range(1 + start_epoch, epochs + 1 + start_epoch):
                self.state.epoch = epoch
                print(f"Beginning epoch {epoch}")
                if self.logger is not None:
                    self.logger.epoch = epoch
                ## Training loop
                for train_step, train_batch in enumerate(self.train_loader, start=1):
                    self.state.global_step += 1
                    self.state.step = train_step
                    self.model.train()
                    for callback in self.callbacks:
                        callback.on_step_begin(self)
                    # I've never seen anyone do gradient accumulation this way, with an inner loop, but it seems like it should work.
                    # Ah, I may have figured out why people don't use this.  So the problem is, if you pass a dictionary or tensors, it's not going to work.  It's going to be a problem
                    #split_batch = torch.split(train_batch, self.gradient_accumulation_batch_size, dim = 0)
                    split_batch = _split_batch(train_batch, self.train_loader.batch_size, self.gradient_accumulation_batch_size)
                    for split in split_batch:
                        #with torch.autocast(device_type = self.device.type, enabled = self.autocast_dtype):
                        with torch.autocast(enabled = self.autocast_dtype != torch.float32, device_type = self.device.type, dtype = self.autocast_dtype):
                            output = self.forward_batch(self.model, split)
                            loss = self.batch_loss(split, output, pad_token_id = self.pad_token_id)
                            loss_item = loss.item() # this seems like the safest way to do it
                            if self.logger is not None: # I don't think the gradient scale affects this
                                self.logger.accumulate_batch_metrics(split, output, loss_item) # this kind of seems like it should cause device issues but apparently it doesn't?
                        # backward pass takes place in full precision
                        #loss.backward()
                        self.scaler.scale(loss).backward()

                    # do the optimizer step after the accumulation
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clipping is not None:
                        clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    #self.optimizer.step()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none = self.set_grad_to_none)
                    # check whether to do logging and evaluation...should these be folded into trainer state?
                    self.state.should_log = False
                    self.state.should_evaluate = False # doing this will make it so you can't specify logging or evaluation at the beginning of the step.
                    
                    do_evaluation, do_logging = False, False
                    if train_step == len(self.train_loader):
                        do_logging = True
                        do_evaluation = True
                    # if we are about to evaluate, log as well
                    elif self.eval_every is not None and train_step % self.eval_every == 0:
                        do_logging = True
                        do_evaluation = True
                    # otherwise check logging frequency
                    elif self.log_every is not None and train_step % self.log_every == 0:
                        do_logging = True
                    # logging
                    if do_logging:
                        self.logger.log_metrics("train", self.state.step)
                        self.logger.logging_timestamp = time.time()

                    for callback in self.callbacks:
                        callback.on_step_end(self)
                    ## Evaluation loop
                    if do_evaluation and self.eval_loader is not None:
                        self.evaluator.evaluate()

                    # end of batch
                    if self.state.should_training_stop:
                        break
                # end of epoch

        finally:
            print("running cleanup routines")
            for callback in self.callbacks:
                callback.on_train_exit(self)
            self.clean_up_gpu()
            torch.backends.cuda.matmul.allow_tf32 = self._save_allowing_tf32


    def reset_scheduler(self, epochs = None): # arguably should this reset training entirely?  including callbacks and initialzation?
        self.epochs = epochs if epochs is not None else self.epochs
        self.scheduler = schedule(self.optimizer, len(self.train_loader) * self.epochs)

    def save(self, path):
       # the path will be a directory.  First, make sure that it exists.
        if not os.path.exists(path):
            os.mkdir(path)
        # save the state dict
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        import json
        # save the logger
        if self.logger is not None:
            with open(os.path.join(path, "log.json"), "w") as f:
                json.dump(self.logger.log, f) # might be nicer as a CSV but this is fine for now
        # repr all the properties
        props = [p for p in dir(self) if not p.startswith("_") and not callable(getattr(self, p))]
        props.insert(0, "model") # the model is callable but we want to save it anyway
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({p: repr(getattr(self, p)) for p in props}, f)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))

    @staticmethod
    def read_metadata(path):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        # take a look and then return it
        pprint.pprint(metadata)
        return metadata

    
    @staticmethod
    def clean_up_gpu(): # you can still run into problems with Jupyter notebooks if you don't restart the kernel
        gc.collect()
        torch.cuda.empty_cache()



## Trainer Callbacks
# In theory some of the logging and printing functions could be rolled into callbacks.

class TrainerCallback():
    def __init__(self):
        pass

    def on_train_begin(self, trainer):
        pass

    #def on_train_end(self, trainer): # I've promised myself I won't implement things prematurely
    #    pass

    def on_train_exit(self, trainer):
        pass

    def on_step_begin(self, trainer):
        pass


import random
class SimpleTestCallback(TrainerCallback):
    def on_train_begin(self, trainer):
        print("Test callback saw training begin.")
    def on_train_exit(self, trainer):
        print("Test callback saw training exit.")
    def on_step_begin(self, trainer):
        if random.random() < 0.01:
            print(f"Trainer noted beginning step {trainer.state.step} (global step {trainer.state.global_step}) of epoch {trainer.state.epoch}.")

from functools import partial
class ResidualGatingWarmupCallback(TrainerCallback):
    def __init__(self, warmup_steps = 2000, start_gate = 0.0, end_gate = 1.0):
        self.warmup_steps = warmup_steps
        self.start_gate = start_gate
        self.end_gate = end_gate
        self.current_gate = start_gate
        self.hooks = []

    #def _hook(self):
        #def _h(module, input, output):
            #if self.current_gate < 1.0:
                #output = output * self.current_gate
            #return output


    def _hook(self, module, input, output):
        if self.current_gate < 1.0:
            return output * self.current_gate
        else:
            return output

    def _remove_hooks(self):
        print("Removing hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def on_train_begin(self, trainer):
        for i, layer in enumerate(trainer.model.decoder.layers):
            print(f"Attaching hooks to layer {i}.")
            #hook = layer.seq_block.register_forward_hook(self._hook())
            hook = layer.seq_block.register_forward_hook(partial(self._hook))
            self.hooks.append(hook)
            #hook = layer.ff_block.register_forward_hook(self._hook())
            hook = layer.ff_block.register_forward_hook(partial(self._hook))
            self.hooks.append(hook)

    def on_step_begin(self, trainer):
        if trainer.state.global_step < self.warmup_steps:
            self.current_gate = self.start_gate + (self.end_gate - self.start_gate) * (trainer.state.global_step / self.warmup_steps)
    
        elif trainer.state.global_step == self.warmup_steps:
            self.current_gate = self.end_gate
            self._remove_hooks()
        else:
            self.current_gate = self.end_gate

    def on_train_exit(self, trainer):
        self._remove_hooks()

 
            
### Is this a good place for this?
from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer
ByteLevelTokenizer = ByT5Tokenizer