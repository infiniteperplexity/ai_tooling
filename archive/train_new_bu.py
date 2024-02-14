## Loss functions

def default_forward_batch(model, batch):
    x = batch
    x = x.to(model.device)
    logits = model(x)
    return logits


import torch.nn as nn
ce = nn.CrossEntropyLoss()
def padless_causal_loss(outputs, batch):
    x = batch.to(outputs.device)
    labels = x[:, 1:]
    logits = outputs[:, :-1]
    loss = ce(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss

## pass the pad_token_id for the tokenizer you're using
def define_pad_causal_loss(pad_token_id):
    pce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def _f(outputs, batch):
        x = batch.to(outputs.device)
        labels = x[:, 1:]
        logits = outputs[:, :-1]
        loss = ce(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss
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


## Trainer class
import torch
import time
class Trainer():
    default_forward_batch = default_forward_batch
    default_loss = padless_causal_loss
    define_pad_causal_loss = define_pad_causal_loss
    print_log_row = print_log_row
    make_csv_logger = make_csv_logger

    def __init__(self,
        model, # required
        train_loader, # required
        eval_loader = None, # not strictly required
        device = torch.device("cpu"),
        forward_batch = default_forward_batch,
        batch_loss = padless_causal_loss,
        report_log_row = print_log_row,
        log_every = None, # defaults to once per epoch
        eval_every = None, # defaults to once per epoch,
        optimizer = torch.optim.AdamW,
        lr = 0.001,
        lr_scheduler = None,
        gradient_accumulation_batch_size = None,
        epochs = 4,
        metrics = {},
    ):
        # basics of the trainer
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.log_every = log_every
        self.eval_every = eval_every
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        self.report_log_row = report_log_row
        if batch_loss == padless_causal_loss:
            print("Warning: No pad token was defined for the loss function.  Loss will be computed over all tokens, including padding tokens.")
        # hyperparameters and training details
        self.optimizer = optimizer(model.parameters(), lr = lr)
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.gradient_accumulation_batch_size = gradient_accumulation_batch_size if gradient_accumulation_batch_size is not None else train_loader.batch_size
        # metrics and logging
        self.metrics = metrics
        self.batch_metrics = {k: [] for k in metrics}
        self.batch_metrics["loss"] = []
        self.log = []
        for k, v in metrics.items(): 
            self.log["train"][k] = []
            self.log["eval"][k] = []

    def train(self):
        logging_timestamp = time.time()
        for epoch in range(1, self.epochs+1):
            print(f"Beginning epoch {epoch}")
            ## Training loop
            for train_step, train_batch in enumerate(self.train_loader, start=1):
                self.model.train()
                # I've never seen anyone do gradient accumulation this way, with an inner loop, but it seems like it should work.
                split_batch = torch.split(train_batch, self.gradient_accumulation_batch_size, dim = 0)
                for split in split_batch:
                    split = split.to(self.device)
                    output = self.forward_batch(self.model, split)
                    loss = self.batch_loss(output, split)
                    self.accumulate_batch_metrics(split, output, loss)
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
                    self.log_metrics("train", epoch, train_step, logging_timestamp)
                    logging_timestamp = time.time()
                ## Evaluation loop
                if do_evaluation and self.eval_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for eval_step, eval_batch in enumerate(self.eval_loader, start=1):
                            eval_batch = eval_batch.to(self.device)
                            output = self.forward_batch(self.model, eval_batch)
                            loss = self.batch_loss(output, eval_batch)
                            self.accumulate_batch_metrics(eval_batch, output, loss)
                            # Log at the end of the evalution, or at specified intervals
                            if eval_step == len(self.eval_loader) or (self.log_every is not None and eval_step % self.log_every == 0):
                                self.log_metrics("eval", epoch, eval_step, logging_timestamp)
                                logging_timestamp = time.time()

    def accumulate_batch_metrics(self, batch, outputs, loss): # everything should be back on CPU At this point.
        self.batch_metrics["loss"].append(loss.item())
        for k, v in self.metrics.items():
            self.batch_metrics[k].append(v(batch, outputs))

    def log_metrics(self, mode, epoch, step, timestamp): # everything should be back on CPU At this point.
        row = {}
        row["mode"] = mode
        row["epoch"] = epoch
        row["step"] = step
        current_timestamp = time.time()
        row["seconds"] = current_timestamp - timestamp
        row["loss"] = sum(self.batch_metrics["loss"]) / len(self.batch_metrics["loss"])
        row["ppl"] = torch.exp(torch.tensor(row["loss"])).item()
        for k, v in self.metrics.items():
            row[k] = sum(self.batch_metrics[k]) / len(self.batch_metrics[k])
        # report and add the row to the log
        self.report_log_row(row)
        self.log.append(row)
        for k in self.batch_metrics:
            self.batch_metrics[k] = []

    def evaluate(self, eval_loader = None, report_log_row = print_log_row):
        if eval_loader is None:
            eval_loader = self.eval_loader
        # Run one full evaluation loop
        self.model.eval()
        with torch.no_grad():
            for eval_step, eval_batch in enumerate(self.eval_loader, start=1):
                eval_batch = eval_batch.to(self.device)
                output = self.forward_batch(self.model, eval_batch)
                loss = self.batch_loss(output, eval_batch)
                self.accumulate_batch_metrics(eval_batch, output, loss)
                # Log only after the full evaluation
            self.log_metrics("eval", 0, eval_step, 0)
        # return log
        return self.log

    import gc
    @staticmethod
    def clean_up_gpu(): # you can still run into problems with Jupyter notebooks if you don't restart the kernel
        gc.collect()
        torch.cuda.empty_cache()
