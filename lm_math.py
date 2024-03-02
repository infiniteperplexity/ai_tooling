import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class DataSynthesizer(Dataset):
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

import random
def generate_addition(digits = 3):
    a = random.randint(0, 10**digits-1)
    b = random.randint(0, 10**digits-1)
    c = a + b
    equation = str(a) + "+" + str(b)
    answer = str(c)
    return equation, answer

def generate_subtraction(digits = 3):
    a = random.randint(0, 10**digits-1)
    b = random.randint(0, 10**digits-1)
    c = a - b
    equation = str(a) + "-" + str(b)
    answer = str(c)
    return equation, answer



def encode_autoregressive_equation(fun, max_seq_len):
    equation, answer = fun()
    input = coder.start_char + equation + "=" + answer
    output = equation + "=" + answer + coder.end_char
    
    input = input.ljust(MAX_LEN, coder.pad_char)
    output = output.ljust(MAX_LEN, coder.pad_char)
    input = coder.encode(input)
    output = coder.encode(output)
    return input, output