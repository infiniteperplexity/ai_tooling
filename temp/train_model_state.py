# So one big thing that would be a hassle is that the Trainer and Evaluator need to share state.
# They could both hold a reference to it. 
# They could pass it to each other.

# So if I take the purely functional route of passing the state around explicitly... 
    # I could make it a dictionary, in which case custom functions could choose to stick weird things in there, which is both good and bad in obvious ways. 
    # I could make it a class with defined fields, and when some new obscure thing is needed, I could genuinely add it to the definition. 

# So let's talk about my current needs.  Right now, I can imagine functions that do slightly different things based on what epoch we are in.  But we might also want dataloaders that do that. 
# I can imagine metrics that want to track components of losses that are not the main loss.  Those could in theory be part of "outputs" I suppose?  Actually that works really wel I think. 
# 

import torch
import torch.nn as nn
import time

## utility functions

def unpack_inputs(batch, key="input_ids"):
    if isinstance(batch, tuple):
        return batch[0]
    elif isinstance(batch, dict):
        return batch[key]
    else:
        return batch

def unpack_targets(batch, key="labels"):
    if isinstance(batch, tuple):
        return batch[1]
    elif isinstance(batch, dict):
        return batch[key]
    else:
        return batch


## Forward functions

def default_forward_batch(model, batch):
    x = unpack_inputs(batch).to(model.device)
    logits = model(x)
    return logits

def multiple_choice_forward_batch(model, batch):
    choice_ids = batch["choice_ids"].to(model.device)
    logits = []
    for i in range(choice_ids.shape[1]):
        outputs = model(input_ids=choice_ids[:, i, :].squeeze(1))
        logits.append(outputs)
    logits = torch.stack(logits, dim=1)
    return logits 


## Loss functions

_padless_loss = nn.CrossEntropyLoss()
def padless_causal_loss(outputs, batch):
    x = unpack_inputs(batch).to(outputs.device)
    labels = x[:, 1:]
    logits = outputs[:, :-1]
    loss = _padless_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss

## pass the pad_token_id for the tokenizer you're using
def define_pad_causal_loss(pad_token_id, key = "input_ids"):
    _loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def _f(outputs, batch):
        x = unpack_inputs(batch, key=key).to(outputs.device)
        labels = x[:, 1:]
        logits = outputs[:, :-1]
        loss = _loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss
    return _f

def define_pad_masked_loss(pad_token_id, key = "labels"):
    _loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def _f(outputs, batch):
        logits = outputs
        x = unpack_inputs(batch).to(outputs.device)
        labels = unpack_targets(batch, key = key).to(outputs.device)
        loss = _loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss
    return _f

def define_multiple_choice_loss(pad_token_id, key="choice_ids"): # kind of useless but include it for consistency's sake
    _loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def _f(outputs, batch):
        choice_ids = batch[key].to(outputs.device)
        lm_logits = outputs[:, :, :-1, :]
        lm_labels = choice_ids[:, :, 1:]
        loss = _loss(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_labels.reshape(-1))
        return loss
    return _f

## Metric functions

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# checks whether all unmasked tokens are correct, e.g. for LAMBADA.
def define_masked_answer_accuracy(pad_token_id, target_key = "target_ids"):
    def _f(outputs, batch):
        logits = outputs
        labels = unpack_targets(batch, key = target_key).to(logits.device)
        preds = logits.argmax(dim=-1).to(logits.device)
        preds = torch.where(labels != pad_token_id, preds, torch.tensor(pad_token_id))
        acc = (preds == labels).float()
        acc = acc.prod(dim=-1)
        acc = acc.mean()
        return acc.item()
    return _f

# https://github.com/EleutherAI/lm-evaluation-harness/issues/539
# compares the logprob choices, e.g. for HellaSwag
import torch.nn.functional as F
def define_multiple_choice_logprob_accuracy(pad_token_id, input_key="choice_ids", target_key = "labels"):
    def _f(outputs, batch):
        choice_ids = batch[input_key].to(outputs.device)
        lm_logits = outputs[:, :, :-1, :]
        lm_labels = choice_ids[:, :, 1:]
        mask = (lm_labels != pad_token_id)
        log_probs = F.log_softmax(lm_logits, dim=-1)   
        log_probs = torch.gather(log_probs, -1, lm_labels.unsqueeze(-1)).squeeze(-1) # verified by hand that this works
        masked = log_probs * mask
        sum_log_probs = masked.sum(dim=-1)
        # I tried exponentiating and also normalizing by length, but both made the results worse...this I think is what people recommend.
        preds = sum_log_probs.argmax(dim=-1)
        labels = batch[target_key].to(outputs.device)
        acc = (preds == labels).float().mean()
        return acc.item()
    return _f


## So, one tricky thing about these...we would ideally like them to sum rather than average, although it's easy enough to calculate the sum given the average and the steps.
def unpadded_token_counter(outputs, batch):
    x = unpack_inputs(batch).to(outputs.device)
    count = torch.ones_like(x).sum().item()
    return count

def define_padded_token_counter(pad_token_id, key = "input_ids"):
    def _f(outputs, batch):
        x = unpack_inputs(batch, key = key).to(outputs.device)
        mask = (x != pad_token_id)
        count = mask.sum().item()
        return count
    return _f




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
    def __init__(self, metrics, report_log_row = print_log_row):
        self.metrics = metrics
        self.batch_metrics = {k: [] for k in metrics}
        self.batch_metrics["loss"] = []
        self.log = []
        self.report_log_row = report_log_row
        self.logging_timestamp = None
        self.epoch = 0

    def accumulate_batch_metrics(self, batch, outputs, loss_item): # everything should be back on CPU At this point.
        self.batch_metrics["loss"].append(loss_item)
        # We could count tokens here, but only if we know the padding token.
        for k, v in self.metrics.items():
            self.batch_metrics[k].append(v(batch, outputs)) # these should have already been converted to scalars.


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
        batch_loss = padless_causal_loss,
        metrics = {},
        report_log_row = print_log_row,
        metrics_logger = None # if None, will be created automatically
    ):
        self.device = device
        self.model = model.to(device) # troubleshooting performance
        self.eval_loader = eval_loader
        self.logger = metrics_logger if metrics_logger is not None else MetricsLogger(metrics, report_log_row = report_log_row)
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        if batch_loss == padless_causal_loss:
            print("Warning: No pad token was defined for the loss function.  Loss will be computed over all tokens, including padding tokens.")
        

    def evaluate(self): # this is such a short function, I don't really see where it could have gone wrong...
        self.model.eval()
        with torch.no_grad():
            self.logger.logging_timestamp = time.time()
            for eval_step, eval_batch in enumerate(self.eval_loader, start=1):
                output = self.forward_batch(self.model, eval_batch)
                loss = self.batch_loss(output, eval_batch)
                self.logger.accumulate_batch_metrics(output, eval_batch, loss.item())
        # I don't think there's ever a reason to log the evaluation metrics at times other than the end of the evlauation
        self.logger.log_metrics("eval", eval_step)
        self.logger.logging_timestamp = time.time()


## Trainer class
import gc
import pprint
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
        batch_loss = padless_causal_loss,
        report_log_row = print_log_row,
        metrics = {},
        metrics_logger = None, # if report_log_row is not None, this will be created automatically
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        lr_scheduler = None,
        gradient_accumulation_batch_size = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.log_every = log_every
        self.eval_every = eval_every
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        if batch_loss == padless_causal_loss:
            print("Warning: No pad token was defined for the loss function.  Loss will be computed over all tokens, including padding tokens.")
        # metrics and logging
        self.report_log_row = report_log_row
        if self.report_log_row is not None and metrics_logger is None:
            self.logger = MetricsLogger(metrics, report_log_row = report_log_row)
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
                metrics_logger = self.logger
            )
        else:
            self.evaluator = evaluator
        # hyperparameters and training details
        self.optimizer = optimizer(model.parameters(), lr = lr)
        self.lr_scheduler = lr_scheduler
        self.gradient_accumulation_batch_size = gradient_accumulation_batch_size if gradient_accumulation_batch_size is not None else train_loader.batch_size
        

    def train(self, epochs = 1):
        if self.logger is not None:
            start_epoch = self.logger.epoch
            self.logger.logging_timestamp = time.time()
        else:
            start_epoch = 0
        print(f"Training for {epochs} epochs starting from epoch {start_epoch + 1}; {len(self.train_loader)} steps per epoch.")
        for epoch in range(1 + start_epoch, epochs + 1 + start_epoch):
            print(f"Beginning epoch {epoch}")
            if self.logger is not None:
                self.logger.epoch = epoch
            ## Training loop
            for train_step, train_batch in enumerate(self.train_loader, start=1):
                self.model.train()
                # I've never seen anyone do gradient accumulation this way, with an inner loop, but it seems like it should work.
                split_batch = torch.split(train_batch, self.gradient_accumulation_batch_size, dim = 0)
                for split in split_batch:
                    split = split.to(self.device)
                    output = self.forward_batch(self.model, split)
                    loss = self.batch_loss(output, split)
                    if self.logger is not None:
                        self.logger.accumulate_batch_metrics(split.detach(), output.detach(), loss.item()) # this kind of seems like it should cause device issues but apparently it doesn't?
                    loss.backward()
                # do the optimizer step after the accumulation
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
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

