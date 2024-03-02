# So one big thing that would be a hassle is that the Trainer and Evaluator need to share state.
# They could both hold a reference to it. 
# They could pass it to each other.

# So if I take the purely functional route of passing the state around explicitly... 
    # I could make it a dictionary, in which case custom functions could choose to stick weird things in there, which is both good and bad in obvious ways. 
    # I could make it a class with defined fields, and when some new obscure thing is needed, I could genuinely add it to the definition.   Okay yeah, I guess as long as I manipulate batch and output it'll always work. 


# So let's talk about my current needs.  Right now, I can imagine functions that do slightly different things based on what epoch we are in.  But we might also want dataloaders that do that. 
# I can imagine metrics that want to track components of losses that are not the main loss.  Those could in theory be part of "outputs" I suppose?  Actually that works really wel I think. 
# I think curriculum learning has to be implemented with custom samplers.  So I should throw together just a length warmup to see. 

# Now, let's see if I can figure out the pad_token thing.  So you can definitely...okay, got it.

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

class TrainerCallback():
    # So at some point I had the idea of passing a training/evaluation state around so that the trainer evaluator, and metrics logger could all see it. 
    # I don't remember what ultimately came of that.  It looks like we're currently not sending step number, for example, to a state. 
    # Now, the truth is, you're rarely if every going to want to have an evaluation callback.  But if you did, maybe you would pass an evaluation state instead of a training state?
    # https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.TrainerCallback for inspiration.
    # The callbacks take what seem to me to be an inordinate number of arguments. But heaven forbid they see the actual Trainer itself.
    # There's kind of an oddball object passed called TrainerControl that has these bools starting with "should_", that govern what will happen next; most notably the training might end.
    # The docs say the that of the arguments, only the TrainerControl "can" be changed by the callback.  It's not clear whether that means it's a bad idea, or you literally can't because you receive copies, or what. 


    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_step_begin(self, trainer):
        pass

    def on_train_exit(self, trainer):
        pass

class ResidualGatingWarmupCallback(TrainerCallback):
    def __init__(self, warmup_steps, start_gate = 0.0, end_gate = 0.0):
        self.warmup_steps = warmup_steps
        self.start_gate = start_gate
        self.end_gate = end_gate
        self.current_gate = start_gate
        self.hooks = []

    def _hook(self):
        def _h(module, input, output):
            if self.current_gate < 1.0:
                output = output * self.current_gate
            return output

    def on_train_begin(self, trainer):
        for i, layer in enumerate(trainer.model.decoder.layers):
            layer.seq_block(register_forward_hook(self._hook))
            layer.ff_block.register_forward_hook(self._hook)
            self.hooks.append(hook)

    def on_step_begin(self, trainer):
        if trainer.step < self.warmup_steps:
            self.current_gate = self.start_gate + (self.end_gate - self.start_gate) * (trainer.step / self.warmup_steps)
            ### Oh...it's actually kind of bad to do steps this way, because steps could be more than the number of steps in an epoch.  We might need to track a variable called total_step or global_step or something.
        else:
            self.current_gate = 1.0 # in theory you could remove the hooks here; I don't know if they impact performance when they're not being used.

    def on_train_end(self, trainer):
        pass

    def on_train_exit(self, trainer): # So, this is a bit tricky.  We might want to save some kind of "on_train_interrupted" thing...
        for hook in self.hooks:
            hook.remove()
            
            
    
import gc
import pprint

from torch.nn.utils import clip_grad_norm_
import contextlib


#Okay, so if I want to implement this residual gating warmup thing, I need to get serious about callbacks and hooks.
# We also probably need to wrap training in a context manager, so we can clean things up when we're done, but what is the context manager here?  Does contextlib have some tools for this?
# Oh, actually I think what you do if you don't want the whole "with" context thing is you just use try/finally.

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
        pad_token_id = -100 # if None and tokenizer is provided, will be inferred from the tokenizer
    ):
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
        # hyperparameters and training details
        self.optimizer = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)
        self.epochs = epochs
        self.scheduler = schedule(self.optimizer, len(train_loader) * epochs)
        self.gradient_clipping = gradient_clipping
        # schedule could be a lambda, 
        self.gradient_accumulation_batch_size = gradient_accumulation_batch_size if gradient_accumulation_batch_size is not None else train_loader.batch_size
        self.state = {}

    def train(self, epochs = None):
        epochs = epochs if epochs is not None else self.epochs ## !!! does the state need to know this stuff?  Also, should this update self.epochs?
        try: # wrap the entire training process in a try/finally loop so we can clean up the GPU and hooks if necessary
            if self.logger is not None:
                start_epoch = self.logger.epoch ##!!! Should this be folded into the state?
                self.logger.logging_timestamp = time.time() ##!!! Should this be folded into the state?
            else:
                start_epoch = 0
            print(f"Training for {epochs} epochs starting from epoch {start_epoch + 1}; {len(self.train_loader)} steps per epoch.")
            for epoch in range(1 + start_epoch, epochs + 1 + start_epoch):
                self.state.epoch = epoch # !!! added
                print(f"Beginning epoch {epoch}")
                if self.logger is not None: 
                    self.logger.epoch = epoch
                ## Training loop
                for train_step, train_batch in enumerate(self.train_loader, start=1):
                    self.state.step = train_step
                    #self.state.batch = train_batch # !!! This would rarely be useful for callbacks, but at one point I was considering changing all the training functions to use the state
                    #!!! I think for now maybe we start by only using the state for what we need to use it for.
                    self.model.train()
                    # I've never seen anyone do gradient accumulation this way, with an inner loop, but it seems like it should work.
                    # Ah, I may have figured out why people don't use this.  So the problem is, if you pass a dictionary or tensors, it's not going to work.  It's going to be a problem
                    #split_batch = torch.split(train_batch, self.gradient_accumulation_batch_size, dim = 0)
                    split_batch = _split_batch(train_batch, self.train_loader.batch_size, self.gradient_accumulation_batch_size)
                    for split in split_batch: 
                        output = self.forward_batch(self.model, split)
                        loss = self.batch_loss(split, output, pad_token_id = self.pad_token_id)
                        if self.logger is not None:
                            self.logger.accumulate_batch_metrics(split, output, loss.item()) # this kind of seems like it should cause device issues but apparently it doesn't?
                        loss.backward()
                    
                    if self.gradient_clipping is not None:
                        # we might need to "unscale" here if I ever implement mixed precision
                        clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    # do the optimizer step after the accumulation
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    # check whether to do logging and evaluation
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
                        self.logger.log_metrics("train", train_step)
                        self.logger.logging_timestamp = time.time()
                    ## Evaluation loop
                    if do_evaluation and self.eval_loader is not None:
                        self.evaluator.evaluate()
        finally:
            # probably run some callbacks here
            self.clean_up_gpu()


    def reset_scheduler(self, epochs = None):
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

from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer
ByteLevelTokenizer = ByT5Tokenizer