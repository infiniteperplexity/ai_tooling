# one useful thing about a suite would be if it cached the datasets, and the models.  So yeah. 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from train import (
    Evaluator,
    print_log_row,
    multiple_choice_forward_batch,
    define_pad_causal_loss, define_pad_masked_loss, define_multiple_choice_loss,
    define_masked_answer_accuracy, define_multiple_choice_loglikelihood_accuracy
)


## Abstract task class
class SuiteTask():
    def __init__(self, model, tokenizer, device = "cuda", batch_size = 32, process_batch_size = 1000, report_log_row = print_log_row, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.batch_size = batch_size
        self.process_batch_size = process_batch_size
        self.loader = None
        self.evaluator = None
        self.report_log_row = report_log_row

    def create_loader(self):
        return None

    def create_evaluator(self):
        return None

    def evaluate(self):
        return self.evaluator.evaluate()

## Several specific tasks

class LambadaTask(SuiteTask):
    def __init__(self, model, tokenizer, max_len = 512, use_openai_version = True, **kwargs):
        super().__init__(model, tokenizer, **kwargs) # let's see if this works, without listing out all the specific arguments
        if use_openai_version:
            self.checkpoint = 'EleutherAI/lambada_openai'
        else:
            self.checkpoint = 'lambada'
        self.loader = self.create_loader(max_len = max_len, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)
        

    def create_loader(self, max_len = 512, batch_size = 32, process_batch_size = 1000):
        def _process_batch(batch):
            result = {"input_ids": [], "target_ids": []}
            for text in batch["text"]:
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                _, last = text.rsplit(" ", 1)
                last = " " + last
                target_ids = self.tokenizer(last, return_tensors="pt").input_ids
                target_ids = F.pad(target_ids, (input_ids.shape[-1]-target_ids.shape[-1], max_len-input_ids.shape[-1]), value = self.tokenizer.pad_token_id)
                input_ids = F.pad(input_ids, (0,max_len-input_ids.shape[-1]), value = self.tokenizer.pad_token_id)
                input_ids = input_ids[:, :-1]
                target_ids = target_ids[:, 1:]
                result["input_ids"].append(input_ids)
                result["target_ids"].append(target_ids)
            result["input_ids"] = torch.cat(result["input_ids"])
            result["target_ids"] = torch.cat(result["target_ids"])
            return result

        self.raw = datasets.load_dataset(self.checkpoint, split = "test")
        self.processed = self.raw.map(_process_batch, batched=True, batch_size = self.process_batch_size).select_columns(["input_ids", "target_ids"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = self.batch_size, shuffle = False)
        return dl

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            batch_loss = define_pad_causal_loss(self.tokenizer.pad_token_id),
            metrics = {"accuracy": define_masked_answer_accuracy(self.tokenizer.pad_token_id)}
        )
        return evaluator



class HellaSwagTask(SuiteTask):
    def __init__(self, model, tokenizer, max_len = 256, n_choices = 4, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.loader = self.create_loader(max_len = max_len, n_choices = n_choices, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)
    
    def create_loader(self, max_len = 256, n_choices = 4, batch_size = 32, process_batch_size = 1000):
        def tokenize_choices(batch):
            choice_tokens = {"choice_ids": []}
            for ctx, endings in zip(batch["ctx"], batch["endings"]):
                choice_ids = []
                for i in range(n_choices):
                    text = ctx + " " + endings[i]
                    tokenized = self.tokenizer(text, max_length=max_len, padding="max_length", truncation=True)
                    choice_ids.append(tokenized["input_ids"])
                choice_tokens["choice_ids"].append(choice_ids)
            label = [int(item) for item in batch["label"]]
            choice_tokens["labels"] = label
            return choice_tokens

        self.raw = datasets.load_dataset("Rowan/hellaswag", split="validation")
        self.processed = self.raw.map(tokenize_choices, batched=True, batch_size = process_batch_size).select_columns(["choice_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = batch_size, shuffle = False)
        return dl

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            forward_batch = multiple_choice_forward_batch,
            batch_loss = define_multiple_choice_loss(self.tokenizer.pad_token_id, key="choice_ids"),
            metrics = {"accuracy": define_multiple_choice_loglikelihood_accuracy(self.tokenizer.pad_token_id, target_key = "labels")}
        )
        return evaluator


## Wrapper class for HuggingFace models

class HuggingFaceWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            return outputs[0]
        elif isinstance(outputs, dict):
            return outputs["logits"]
        else:
            return outputs.logits

    @property
    def device(self):
        return self.model.device