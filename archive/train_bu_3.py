import json
import csv
import os
import torch

# Should this be in here?
class DirCSVLogger():
    def __init__(self, dirname):
        self.dirname = dirname
        self.csv_file = os.path.join(dirname, "log.csv")
        self.metadata_file = os.path.join(dirname, "metadata.json")

    def write_metadata(self, metadata):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def write_row(self, row, mode = "a"):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        with open(self.csv_file, mode) as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_rows(self):
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)
            return list(reader)

    def get_metadata(self):
        with open(self.metadata_file, "r") as f:
            return json.load(f)



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


def define_pad_causal_loss(pad_token_id):
    pce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def _f(outputs, batch):
        x = batch.to(outputs.device)
        labels = x[:, 1:]
        logits = outputs[:, :-1]
        loss = ce(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss
    return _f

def print_log_row(row):
    print(row)

### Okay, this seems like it's actually running, although I can't tell if the loss is going down or not. 
# I'm going to need to add gradient accumulation sooner rather than later, we run out of memory on long sequences even with small models and moderate batch sizes.
import time
class Trainer():
    default_forward_batch = default_forward_batch
    default_loss = padless_causal_loss
    define_pad_causal_loss = define_pad_causal_loss
    print_log_row = print_log_row

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
        gradient_accumulation_steps = 1,
        epochs = 4,
        metrics = {},
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.log_every = log_every if log_every is not None else len(train_loader) // train_loader.batch_size # I'm not sure that works quite right but we shall see
        self.eval_every = eval_every if eval_every is not None else len(train_loader) // train_loader.batch_size
        self.forward_batch = forward_batch
        self.batch_loss = batch_loss
        self.report_log_row = report_log_row
        if batch_loss == padless_causal_loss:
            print("Warning: No pad token was defined for the loss function.  Loss will be computed over all tokens, including padding tokens.")

        self.optimizer = optimizer(model.parameters(), lr = lr)
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        ## Metrics and logging
        self.metrics = metrics
        self.batch_metrics = {k: [] for k in metrics}
        self.batch_metrics["loss"] = []
        self.log = []
        for k, v in metrics.items(): 
            self.log["train"][k] = []
            self.log["eval"][k] = []

    def train(self):
        logging_timestamp = time.time()
        for epoch in range(self.epochs):
            print(f"Beginning epoch {epoch}")
            ## Training loop
            for train_step, train_batch in enumerate(self.train_loader):
                self.model.train()
                train_batch = train_batch.to(self.device)
                output = self.forward_batch(self.model, train_batch)
                loss = self.batch_loss(output, train_batch)
                self.accumulate_batch_metrics(train_batch, output, loss)
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                # Log at the end of the epoch, or at specified intervals, or if it's time for evaluation
                if train_step > 0 and (train_step == len(self.train_loader) - 1 or (self.log_every is not None and train_step % self.log_every == 0) or (self.eval_every is not None and train_step % self.eval_every == 0)):
                    self.log_metrics("train", epoch, train_step, logging_timestamp)
                    logging_timestamp = time.time()
                ## Evaluation loop
                if train_step > 0 and train_step % self.eval_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for eval_step, eval_batch in enumerate(self.eval_loader):
                            eval_batch = eval_batch.to(self.device)
                            output = self.forward_batch(self.model, eval_batch)
                            loss = self.batch_loss(output, eval_batch)
                            self.accumulate_batch_metrics(eval_batch, output, loss)
                            # Log at the end of the evalution, or at specified intervals
                            if eval_step > 0 and (eval_step == len(self.eval_loader) - 1 or (self.log_every is not None and eval_step % self.log_every == 0)):
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
        for k, v in self.metrics.items():
            row[k] = sum(self.batch_metrics[k]) / len(self.batch_metrics[k])
        # report and add the row to the log
        self.report_log_row(row)
        self.log.append(row)
        for k in self.batch_metrics:
            self.batch_metrics[k] = []
