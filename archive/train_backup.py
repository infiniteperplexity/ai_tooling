import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import time

class MetricsLogger(): # metrics should be a dictionary of Metrics
    def __init__(self, metrics):
        self.metrics = metrics
        self.epoch_log = {k:[] for k in metrics.keys()}
        self.training_log = {k:[]  for k in metrics.keys()}

    def log_batch(self, batch, output, loss):
        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            self.epoch_log[metric_name].append(metric(batch, output, loss))


    def log_epoch(self):
        for metric_name in self.metrics.keys():
            self.training_log[metric_name].append(np.mean(self.epoch_log[metric_name]))
            self.epoch_log[metric_name] = []

        latest = {k: v[-1] for k, v in self.training_log.items()}
        return latest

def loss_metric(batch, output, loss):
    return loss.detach().item()

def accuracy_metric(batch, output, loss):
    _, y = batch
    y_pred = torch.argmax(output, dim=2)
    return (y_pred == y).float().mean().item()
    
def sequence_accuracy_metric(batch, output, loss):
    _, y = batch
    y_pred = torch.argmax(output, dim=2)
    return (y_pred == y).all(dim=1).float().mean().item()


def default_forward_batch(model, batch, device):
    x, _ = batch
    x = x.to(device)
    logits = model(x)
    return logits

loss_fn = nn.CrossEntropyLoss()
def default_loss(outputs, batch, device):
    _, y = batch
    logits = outputs
    ## Arguably we shoudl be getting rid of padding...but is that important for this use case?
    y = y.to(device)
    loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
    return loss

def default_step_counter(step):
    if step % 100 == 0:
        print(f"Step {step}")
        

def default_epoch_logger(epoch, epoch_train_time, train_log, test_log):
    print(f"Epoch {epoch} took {epoch_train_time:.2f} seconds")
    print(f"Test: {test_log}")

import json
def file_epoch_logger(file, desc):
    def logger(epoch, epoch_train_time, train_log, test_log):
        print(f"Epoch {epoch} took {epoch_train_time:.2f} seconds")
        print(f"Test: {test_log}")
        # check if the file exists
        fname = file
        # if it doesn't end with .json, add it
        if not fname.endswith(".json"):
            fname = fname + ".json"
        try:
            with open(fname, "r") as f:
                jsn = json.load(f)

        except FileNotFoundError:
            print(f"Creating new file {fname}")
            jsn = {
                "description": desc,
                "data": []
            }

        row = {
            "epoch": epoch,
            "epoch_train_time": epoch_train_time
        }
        for key in train_log.keys():
            row[f"train_{key}"] = train_log[key]
        for key in test_log.keys():
            row[f"test_{key}"] = test_log[key]

        jsn["data"].append(row)
        with open(fname, "w") as f:
            f.write(json.dumps(jsn, indent=2))

    return logger


def train(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 30,
    batch_forward = default_forward_batch,
    batch_loss = default_loss,
    device: torch.device = torch.device("cpu"),
    train_metrics = None,
    test_metrics = None,
    train_counter = None,
    test_counter = None,
    epoch_logger = default_epoch_logger
) -> None:
    default_keras_lr = 0.001
    lr = default_keras_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        print(f"Beginning epoch {epoch}")
        epoch_train_start = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            if train_counter is not None:
                train_counter(step)
            output = batch_forward(model, batch, device)
            loss = batch_loss(output, batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if train_metrics is not None:
                train_metrics.log_batch(batch, output.detach().cpu(), loss.detach().cpu())

        epoch_train_time = time.time() - epoch_train_start

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                if test_counter is not None:
                    test_counter(step)
                output = batch_forward(model, batch, device)
                loss = batch_loss(output, batch, device)
                if test_metrics is not None:
                    test_metrics.log_batch(batch, output.cpu(), loss.cpu())

        if train_metrics is not None:
            train_log = train_metrics.log_epoch() # don't print it for now
        if test_metrics is not None:
            test_log = test_metrics.log_epoch()
        
        if epoch_logger is not None:
            train_log = train_log if train_metrics is not None else None
            test_log = test_log if test_metrics is not None else None
            epoch_logger(
                epoch, 
                epoch_train_time,
                train_log,
                test_log,
            )

    return train_metrics, test_metrics