# one useful thing about a suite would be if it cached the datasets, and the models.  So yeah. 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from train import (
    Evaluator,
    print_log_row,
    multiple_choice_forward_batch, multiple_choice_loss,
    masked_answer_accuracy, multiple_choice_logprob_accuracy
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
            result = {"input_ids": [], "labels": []}
            for text in batch["text"]:
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                _, last = text.rsplit(" ", 1)
                last = " " + last
                labels = self.tokenizer(last, return_tensors="pt").input_ids
                labels = F.pad(labels, (input_ids.shape[-1]-labels.shape[-1], max_len-input_ids.shape[-1]), value = self.tokenizer.pad_token_id)
                input_ids = F.pad(input_ids, (0,max_len-input_ids.shape[-1]), value = self.tokenizer.pad_token_id)
                input_ids = input_ids[:, :-1]
                labels = labels[:, 1:]
                result["input_ids"].append(input_ids)
                result["labels"].append(labels)
            result["input_ids"] = torch.cat(result["input_ids"])
            result["labels"] = torch.cat(result["labels"])
            return result

        self.raw = datasets.load_dataset(self.checkpoint, split = "test")
        self.processed = self.raw.map(_process_batch, batched=True, batch_size = self.process_batch_size).select_columns(["input_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = self.batch_size, shuffle = False)
        return dl

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            metrics = {"accuracy": masked_answer_accuracy},
            tokenizer = self.tokenizer
        )
        return evaluator


class SwagTask(SuiteTask):
    def __init__(self, model, tokenizer, max_len = 128, n_choices = 4, **kwargs):
        super().__init__(model, tokenizer, **kwargs) #ugh, I think you actually do have to pass through all the arguments.
        self.loader = self.create_loader(max_len = max_len, n_choices = n_choices, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)

    def create_loader(self, max_len = 128, n_choices = 4, batch_size = 32, process_batch_size = 1000):
        def tokenize_choices(batch):
            choice_tokens = {"input_ids": []}
            endings = [batch[f"ending{i}"] for i in range(n_choices)]
            for ctx, *ends in zip(batch["startphrase"], *endings):
                input_ids = []
                for i in range(n_choices):
                    text = ctx + " " + ends[i]
                    tokenized = self.tokenizer(text, max_length=max_len, padding="max_length", truncation=True)
                    input_ids.append(tokenized["input_ids"])
                choice_tokens["input_ids"].append(input_ids)
            label = [int(item) for item in batch["label"]]
            choice_tokens["labels"] = label
            return choice_tokens

        self.raw = datasets.load_dataset("swag", split="validation")
        self.processed = self.raw.map(tokenize_choices, batched=True, batch_size = process_batch_size).select_columns(["input_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = batch_size, shuffle = False)
        return dl

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            forward_batch = multiple_choice_forward_batch,
            batch_loss = multiple_choice_loss,
            metrics = {"accuracy": multiple_choice_logprob_accuracy},
            tokenizer = self.tokenizer
        )
        return evaluator


class HellaSwagTask(SuiteTask):
    def __init__(self, model, tokenizer, max_len = 256, n_choices = 4, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.loader = self.create_loader(max_len = max_len, n_choices = n_choices, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)
    
    def create_loader(self, max_len = 256, n_choices = 4, batch_size = 32, process_batch_size = 1000):
        def tokenize_choices(batch):
            choice_tokens = {"input_ids": []}
            for ctx, endings in zip(batch["ctx"], batch["endings"]):
                input_ids = []
                for i in range(n_choices):
                    text = ctx + " " + endings[i]
                    tokenized = self.tokenizer(text, max_length=max_len, padding="max_length", truncation=True)
                    input_ids.append(tokenized["input_ids"])
                choice_tokens["input_ids"].append(input_ids)
            label = [int(item) for item in batch["label"]]
            choice_tokens["labels"] = label
            return choice_tokens

        self.raw = datasets.load_dataset("Rowan/hellaswag", split="validation")
        self.processed = self.raw.map(tokenize_choices, batched=True, batch_size = process_batch_size).select_columns(["input_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = batch_size, shuffle = False)
        return dl

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            forward_batch = multiple_choice_forward_batch,
            batch_loss = multiple_choice_loss,
            metrics = {"accuracy": multiple_choice_logprob_accuracy},
            tokenizer = self.tokenizer
        )
        return evaluator


class ChildrensBookTask(SuiteTask):
    def __init__(self, model, tokenizer, configuration = 'CN', line_break = " ", mask = "XXXXX", max_len = 1024, n_choices = 10, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.loader = self.create_loader(configuration = configuration, max_len = max_len, n_choices = n_choices, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)
    
    def create_loader(self, configuration = 'CN', line_break = " ", mask = "XXXXX", max_len = 1024, n_choices = 10, batch_size = 1, process_batch_size = 1000):
        def tokenize_choices(batch):
            choice_tokens = {"input_ids": []}
            sentences = [line_break.join(sentences + [question]) for sentences, question in zip(batch["sentences"], batch["question"])]
            for sentence, options in zip(sentences, batch["options"]):
                input_ids = []
                for i in range(n_choices):
                    text = sentence.replace(mask, options[i])
                    tokenized = self.tokenizer(text, max_length=max_len, padding="max_length")
                    input_ids.append(tokenized["input_ids"][-max_len:])
                choice_tokens["input_ids"].append(input_ids)
            label = [options.index(answer) for options, answer in zip(batch["options"], batch["answer"])]
            choice_tokens["labels"] = label
            return choice_tokens

        self.raw = datasets.load_dataset("cbt", configuration, split="test")
        self.processed = self.raw.map(tokenize_choices, batched=True, batch_size = process_batch_size).select_columns(["input_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = batch_size, shuffle = False)
        return dl

    @staticmethod
    def get_configuration_names():
        return datasets.get_dataset_config_names("cbt")

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            forward_batch = multiple_choice_forward_batch,
            batch_loss = multiple_choice_loss,
            metrics = {"accuracy": multiple_choice_logprob_accuracy},
            tokenizer = self.tokenizer
        )
        return evaluator



class BlimpTask(SuiteTask):
    def __init__(self, model, tokenizer, configuration = 'adjunct_island', max_len = 64, n_choices = 2, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.loader = self.create_loader(configuration = configuration, max_len = max_len, n_choices = n_choices, batch_size = self.batch_size, process_batch_size = self.process_batch_size)
        self.evaluator = self.create_evaluator(device = self.device, report_log_row = self.report_log_row)
    
    def create_loader(self, configuration = 'adjunct_island', max_len = 64, n_choices = 2, batch_size = 32, process_batch_size = 1000):
        def tokenize_choices(batch):
            choice_tokens = {"input_ids": []}
            for good, bad in zip(batch["sentence_good"], batch["sentence_bad"]):
                good_ids = self.tokenizer(good, max_length=max_len, padding="max_length")["input_ids"]
                bad_ids = self.tokenizer(bad, max_length=max_len, padding="max_length")["input_ids"]
                choice_tokens["input_ids"].append([good_ids, bad_ids])
            label = [0 for _ in batch["sentence_good"]] # as long as this isn't used for training, it should be fine to have the correct answer always be the same
            choice_tokens["labels"] = label
            return choice_tokens

        self.raw = datasets.load_dataset("nyu-mll/blimp", configuration, split="train")
        self.processed = self.raw.map(tokenize_choices, batched=True, batch_size = process_batch_size).select_columns(["input_ids", "labels"])
        self.processed.set_format(type="torch")
        dl = DataLoader(self.processed, batch_size = batch_size, shuffle = False)
        return dl

    @staticmethod
    def get_congifuration_names():
        return datasets.get_dataset_config_names("nyu-mll/blimp")

    def create_evaluator(self, device = "cuda", report_log_row = print_log_row):
        evaluator = Evaluator(
            self.model,
            self.loader,
            device = device,
            forward_batch = multiple_choice_forward_batch,
            batch_loss = multiple_choice_loss,
            metrics = {"accuracy": multiple_choice_logprob_accuracy},
            tokenizer = self.tokenizer
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

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)