from transformers import AutoTokenizer
import torch
class WrappedHuggingFaceTokenizer():
    """
    Wraps a HuggingFace tokenizer with a simpler interface for use in PyTorch models.
    """
    MISTRAL = "mistralai/Mistral-7B-v0.1"
    GPT2 = "gpt2"
    def __init__(self, checkpoint = "gpt2", ignore_index = -100):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side = "right")
        self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        self.ignore_index = ignore_index

    def __call__(self, texts):
        tokenized = self.hf_tokenizer(texts, padding = "longest", return_tensors = "pt").input_ids
        mask = tokenized == self.hf_tokenizer.pad_token_id
        tokenized[mask] = self.ignore_index
        return tokenized

    def decode(self, tokenized):
        return self.hf_tokenizer.decode(tokenized)


from torch.utils.data import Dataset
class DataSynthesizer(Dataset):
    """ 
    Wrap a function for generating synthetic data as a PyTorch Dataset.
    """
    def __init__(
        self,
        fun,
        length: int = 1000,
    ):
        self.length = length
        self.function = fun

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return self.function()