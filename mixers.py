import functools
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


### Causal Masking ###

## Basically not implemented as a class yet...we need to at least implement this in several classes so we understand how to generalize.
class CausalMasks:
    _cache = {}

### Dummies ###

def no_activation(x):
    return x

class NoNorm(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.model_size = model_size

    def forward(self, x):
        return x


class NoMixer(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.model_size = model_size

    def forward(self, x):
        return x


class ReZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        return x * self.resweight


### Norms ###

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        In practice this seems to be slower than LayerNorm, which completely defeats the point.
        A hand-rolled LayerNorm is even slower than this, which suggests that it's the C implementation that makes their version go faster.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


    


### Vectorizers ###

class EmbeddingVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = None, tied_weights = True, dropout = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.dropout = nn.Dropout(dropout)
        self.tied_weights = tied_weights

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

    def get_tieable_weights(self):
        return self.embedding.weight


class EmbeddingAndPositionalVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512, tied_weights = True, dropout = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.positional_embedding = nn.Embedding(max_seq_len, model_size)
        self.dropout = nn.Dropout(dropout)
        self.tied_weights = tied_weights

    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device = x.device)
        pos_emb = self.positional_embedding(positions)
        x = self.embedding(x) + pos_emb
        x = self.dropout(x)
        return x

    def get_tieable_weights(self):
        return self.embedding.weight


class UpscalingEmbeddingsVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512, scaling = 8, tied_weights = True, dropout = 0.0, norm = nn.LayerNorm):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size // scaling)
        self.positional_embedding = nn.Embedding(max_seq_len, model_size // scaling)
        # As far as I can tell, ALBERT norms and dropouts the embeddings before upscaling.
        self.norm = norm(model_size // scaling)
        self.dropout = nn.Dropout(0.0)
        self.emb_proj = nn.Linear(model_size // scaling, model_size, bias = False)
        self.tied_weights = tied_weights

    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device = x.device)
        emb = self.embedding(x)
        pos_emb = self.positional_embedding(positions)
        x = emb + pos_emb
        x = self.norm(x)
        x = self.dropout(x)
        x = self.emb_proj(x)
        return x

    def get_tieable_weights(self):
        return self.embedding.weight

class OneHotVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = None, dropout = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        # the model size has to equal the vocab size plus the max sequence length

    def forward(self, x):
        x = torch.nn.functional.one_hot(x, self.vocab_size).float()
        x = self.dropout(x)
        return x
    
class OneHotAndPositionalVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512, dropout = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, s = x.shape
        one_hot = F.one_hot(x, self.vocab_size).float()
        positions = torch.arange(s, device = x.device).unsqueeze(0).expand(b, -1)
        pos_emb = F.one_hot(positions, self.max_seq_len).float()
        x = torch.cat([one_hot, pos_emb], dim = -1)
        x = self.dropout(x)
        return x


### Sequence Mixers ###

## Rotary Positional Embeddings
# I have confirmed that the outputs are the correct shape, and that in at least one test case, the outputs are the same as the outputs from the Hugging Face implementation.
class RoPE:
    _cache = {}
    @classmethod
    def _populate_cache(cls, dim, seq_len, device, cache_key):
        dim, period = cache_key
        inv_freq = 1.0 / (period ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        t = torch.arange(seq_len, device = device, dtype = torch.int64).type_as(inv_freq)
        with torch.autocast(device_type = device.type, enabled=False):
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(torch.get_default_dtype())
            sin = emb.sin().to(torch.get_default_dtype())
        cls._cache[cache_key] = (seq_len, cos, sin)
        
    @classmethod
    def _get_cached_sin_con(cls, dim, seq_len, device, period = 10_000):
        cache_key = (dim, period)
        if cache_key not in cls._cache or seq_len > cls._cache[cache_key][0]:
            cls._populate_cache(dim, seq_len, device, cache_key)
        _, cos, sin = cls._cache[cache_key]
        return cos[:seq_len].to(device), sin[:seq_len].to(device)

    @classmethod
    def embed(cls, x, period = 10_000, head_size = None):
        seq_len = x.size(-2)
        device = x.device
        dim = head_size if head_size is not None else x.size(-1)
        cos, sin = cls._get_cached_sin_con(dim, seq_len, device, period = period)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        embedded = (x * cos) + (rotated * sin)
        return embedded

### Attention using the Hugging Face implementation of GPT2, intended for use as a reference and validation tool.
# I have confirmed that this works identifcally to my own implementation, at least as an isolated layer.
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2Config
class GPT2AttentionMixer(nn.Module):
    def __init__(self, model_size, num_heads = 1, dropout = 0.0):
        super().__init__()
        self.model_size = model_size
        self.num_heads = num_heads
        self.attn = GPT2Attention(GPT2Config(n_head = num_heads, n_embd = model_size, attn_pdrop = dropout, resid_pdrop = 0.0))

    def forward(self, x):
        x = self.attn(x)[0]
        return x

class AttentionMixer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout = 0.0, apply_rope = False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.apply_rope = apply_rope
        self.k_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.q_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.v_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias = False)
        for weights in [self.k_proj, self.q_proj, self.v_proj, self.out_proj]: # why did I have to do this here again?
            nn.init.normal_(weights.weight, std = 0.02)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        keys = self.k_proj(x)
        queries = self.q_proj(x)
        values = self.v_proj(x)
        # split into heads
        batch_size, seq_len, model_dim = x.shape
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE
        if self.apply_rope:
            queries = RoPE.embed(queries, head_size = self.head_dim)
            keys = RoPE.embed(keys, head_size = self.head_dim)
        # compute attention
        attn = (queries @ keys.transpose(-1, -2)) / self.head_dim**0.5
        # apply causal mask
        # !!! This is the one part that works a bit differently from the GPT2 implementation; they use where, I use addition.  It should not, and does not seem to make a meaningful difference.
        mask = self._get_causal_mask(attn)
        attn = attn + mask
        # apply softmax
        #attn = attn.softmax(dim = -1)
        attn = self.softmax(attn)
        # apply dropout
        attn = self.dropout(attn)
        # apply attention
        out = attn @ values
        # merge heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, model_dim)
        # apply output projection
        out = self.out_proj(out)
        return out

    # cache causal masks on the class, so the layers can share them...I validated that these do not register as parameters.
    _cached_causal_masks = {}
    @classmethod
    def _get_causal_mask(cls, attns):
        seq_len = attns.shape[-1] # no support currently for separate key and query lengths
        neg_inf = torch.finfo(attns.dtype).min
        cache_key = (seq_len, neg_inf, attns.device.type)
        if cache_key not in cls._cached_causal_masks:
            neg_infs = neg_inf * torch.ones(seq_len, seq_len, dtype = attns.dtype)
            causal_mask = torch.triu(neg_infs, diagonal = 1).unsqueeze(0).unsqueeze(0).to(attns.device)
            cls._cached_causal_masks[cache_key] = causal_mask
        #return cls._cached_causal_masks[seq_len].to(attns.device)
        return cls._cached_causal_masks[cache_key]

    @classmethod
    def clear_causal_mask_cache(cls):
        cls._cached_causal_masks = {}

import pynvml
def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import math
from fast_transformers.causal_product import causal_dot_product
#from causal_attention import causal_dot_product as causal_dot_product_cuda
class LinearAttentionMixer(nn.Module):
    def __init__(self, model_dim, num_heads, apply_rope = False, feature_map = None, eps = 1e-12): # should probably parameterize which version of the function to use?
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.apply_rope = apply_rope
        self.k_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.q_proj = nn.Linear(model_dim, model_dim, bias = False)
        #self.feature_map = feature_map if feature_map is not None else self.identity_map
        self.feature_map = feature_map if feature_map is not None else self.pos_elu
        self.v_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias = False)
        for weights in [self.k_proj, self.q_proj, self.v_proj, self.out_proj]: # copying this from the normal attention mixer, but I don't remember why it's a thing.
            nn.init.normal_(weights.weight, std = 0.02)
        self.eps = eps

    @classmethod
    def identity_map(cls, x, head_dim):
        return x

    _cached_sqrts = {}
    @classmethod
    def _cached_sqrt(cls, x):
        if x not in cls._cached_sqrts:
            cls._cached_sqrts[x] = math.sqrt(x)
        return cls._cached_sqrts[x]


    @classmethod
    def taylor_expansion(cls, x, head_dim):
        """ assume inputs (b, h, l, d) """ 
        device = x.device
        _1 = torch.ones(x[:,:,:,:1].shape, device = device)
        _x = x / cls._cached_sqrt(cls._cached_sqrt(head_dim))
        _x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / cls._cached_sqrt(2*head_dim)
        return torch.cat([_1, _x, _x2], dim=-1)


    @classmethod
    def pos_elu(cls, x, head_dim):
        # this is suppsed to get multiplied by a temperature that defaults to 1 but I'm not going to do that unless it looks promising
        return F.elu(x) + 1

    @classmethod
    def relu(cls, x, head_dim):
        return F.relu(x).clamp(min=1e-12)

    def forward(self, x):
        keys = self.k_proj(x)
        queries = self.q_proj(x)
        values = self.v_proj(x)
        # split into heads
        batch_size, seq_len, model_dim = x.shape
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE...I assume this is allowed for linear attention, and I assume it comes before the feature map.
        if self.apply_rope:
            queries = RoPE.embed(queries, head_size = self.head_dim)
            keys = RoPE.embed(keys, head_size = self.head_dim)
        # apply feature maps
        queries = self.feature_map(queries, self.head_dim)
        keys = self.feature_map(keys, self.head_dim)
        # compute linear attention
        # hold on...they seem to do the head-splitting after the feature map.
        #if keys.device.type == "cuda":
            #causal_dot_product = causal_dot_product_cuda
        #else:
            #causal_dot_product = causal_dot_product_cpu
            
        if causal_dot_product is not None: # trying to use the logic from the fast_transformers library
            Z = 1/(torch.einsum("nlhi,nlhi->nlh", queries, keys.cumsum(1)) + self.eps)
            attn = causal_dot_product(
                queries.permute(0,2,1,3).contiguous().float(),
                keys.permute(0,2,1,3).contiguous().float(),
                values.permute(0,2,1,3).contiguous().float(),
            ).permute(0,2,1,3).contiguous()
            attn.to(x.dtype)
            attn = attn * Z[:, :, :, None].contiguous()
        else:
            queries, keys, values =  queries.unsqueeze(-2), keys.unsqueeze(-2), values.unsqueeze(-1)
            attn = ((queries * (keys * values).cumsum(dim=2)).sum(dim=-1) / ((queries * keys).cumsum(dim=2)).sum(dim=-1) + self.eps) # the non-causal version of this just uses sums instead of cumsums
            attn = attn.transpose(1, 2).contiguous()
        # merge heads
        attn = attn.view(batch_size, seq_len, model_dim)
        # apply output projection
        out = self.out_proj(attn)
        return out
    
# in testing
from mamba_ssm import Mamba
class MambaMixer(nn.Module):
    def __init__(self, model_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba = Mamba(
            d_model = model_dim,
            d_state = d_state,
            d_conv = d_conv,
            expand = expand
        )

    def forward(self, x):
        x = self.mamba(x)
        return x


# Do we want this?  torch.backends.cudnn.benchmark = True?
class SeqConv(nn.Module):
    def __init__(self, model_dim, kernel_size): #, num_heads = 1): ## Probably best not to mess with num_heads; I'm not sure I understood it correctly.
        super().__init__()
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(model_dim, model_dim, kernel_size, padding = kernel_size - 1, groups = model_dim, bias = False)
        #self.num_heads = num_heads
        #self.conv = nn.Conv1d(model_dim, model_dim, kernel_size, padding = kernel_size - 1, groups = model_dim // num_heads, bias = False)

    def forward(self, x):
        b, l, d = x.shape
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :l]
        x = x.transpose(1, 2)
        return x


class SeqConvMixer(nn.Module):
    def __init__(self, model_size, kernel_size = 3, activation = F.gelu):#, num_heads = 1):
        super().__init__()
        self.model_size = model_size
        self.kernel_size = kernel_size
        #self.num_heads = num_heads 
        self.seq_conv = SeqConv(model_size, kernel_size)
        self.activation = activation

    def forward(self, x):
        x = self.seq_conv(x)
        x = self.activation(x)
        return x


class GatedConvSeqMixer(nn.Module):
    def __init__(self, model_size, conv_kernel_size = 3, gate_kernel_size = 3, activation = F.gelu):#, num_heads = 1
        super().__init__()
        self.model_size = model_size
        self.conv_kernel_size = conv_kernel_size
        self.gate_kernel_size = gate_kernel_size
        self.conv = SeqConv(model_size, conv_kernel_size)
        self.gate = SeqConv(model_size, gate_kernel_size)
        self.activation = activation

    def forward(self, x):
        up = self.conv(x)
        gate = self.gate(x)
        gate = self.activation(gate)
        x = up * gate
        return x

### State Mixers ###

class MLPMixer(nn.Module):
    def __init__(self, model_size, expansion = 4, bias = False, activation = F.gelu):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        hidden_dim = int(model_size * expansion)
        self.fc1 = nn.Linear(model_size, hidden_dim, bias = bias)
        self.activation = activation() if isinstance(activation, type) else activation
        self.fc2 = nn.Linear(hidden_dim, model_size, bias = bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x



class MixedHeadMixer(nn.Module):
    def __init__(self, model_size, heads = []):
        super().__init__()
        self.model_size = model_size
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        # bs, sl, d = x.shape
        # d meeds to be split amongs the various heads. 
        split = torch.split(x, [h.model_size for h in self.heads], dim = -1)
        split_out = [h(s) for h, s in zip(self.heads, split)]
        x = torch.cat(split_out, dim = -1)
        return x


class GatedStateMixer(nn.Module): # I'm pretty sure that things like SwiGLU already do this.
    def __init__(self, model_size, expansion = 4, bias = False, gate_activation = F.silu, up_activation = no_activation):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        hidden_dim = int(model_size * expansion)
        self.upscale = nn.Linear(model_size, hidden_dim, bias = bias)
        self.up_activation = up_activation
        self.gate = nn.Linear(model_size, hidden_dim)
        self.gate_activation = gate_activation
        self.downscale = nn.Linear(hidden_dim, model_size, bias = bias)

    def forward(self, x):
        up = self.upscale(x)
        up = self.up_activation(up)
        gate = self.gate(x)
        gate = self.gate_activation(gate)
        x = up * gate
        x = self.downscale(x)
        return x

### Generalized Mixers ###

## Completely untested
class ChimeraMixer(nn.Module):
    def __init__(
        self,
        model_size,
        heads = (),
    ):
        super().__init__()
        self.model_size = model_size
        self.heads = nn.ModuleList(heads)
        self.out_proj = nn.Linear(model_size, model_size, bias = False)


    def forward(self, x):
        split = torch.split(x, [h.model_size for h in self.heads], dim = -1)
        split_out = [h(s) for h, s in zip(self.heads, split)] # I think this actually worked out right just as 
        x = torch.cat(split_out, dim = -1)
        x = self.out_proj(x) # I think you need the attention layer to have no output projection then?
        return x



### Task Heads ###

class CausalLanguageModelHead(nn.Module): # So this is actually kind of stupid...it's now just a classifier.
    def __init__(self, model_size, tokenizer = None, vocab_size = None, pad_token_id = None, loss_fn = nn.CrossEntropyLoss(), tied_weights = True):
        super().__init__()
        if tokenizer is None:
            assert vocab_size is not None, "If no tokenizer is provided, you must provide a vocab size."
            if pad_token_id is None:
                pad_token_id = 0
        else:
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.fc = nn.Linear(model_size, vocab_size, bias = False)
        #self.dropout = nn.Dropout(dropout) # this isn't a thing in GPT-2 as far as I can tell.
        self.loss_fn = loss_fn
        self.tied_weights = tied_weights

    def forward(self, x):
        x = self.fc(x)
        #x = self.dropout(x)
        return x

    def tie_weights(self, wt):
        assert wt.shape == self.fc.weight.shape, "The weight matrices must have the same shape to be tied."
        self.fc.weight = wt


class DownscalingLanguageModelHead(CausalLanguageModelHead):
    def __init__(self, model_size, tokenizer = None, vocab_size = None, pad_token_id = None, dropout = 0.0, loss_fn = nn.CrossEntropyLoss(), tied_weights = True, scaling = 8, norm = nn.LayerNorm):
        super().__init__(model_size, vocab_size = vocab_size, pad_token_id = pad_token_id, loss_fn = loss_fn, tied_weights = tied_weights, activation = nn.GELU)
        self.cls_proj = nn.Linear(model_size, model_size // scaling, bias = False)
        self.activation = activation() if isinstance(activation, type) else activation
        self.norm = norm(model_size // scaling)
        self.fc = nn.Linear(model_size // scaling, vocab_size, bias = False)

    def forward(self, x):
        x = self.cls_proj(x)
        x = self.activation(x) # as far as I can tell, ALBERT does not use dropout here but it does use an activation and a layer norm
        x = self.norm(x)
        x = self.fc(x)
        return x

### Initialization Strategies ###
# I think I've talked myself into thinking this belongs here rather than with the Trainer.
def default_weights(mdl):
    pass

# The initialization for GPT-2 is so similar to this, and no one else seems to have used it, that I think I'll ignore it.
def init_weights_normal(mdl, mean = 0.0, std = 0.02, tokenizer = None, pad_token_id = None):
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
    for m in mdl.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Embedding):
            m.weight.data.normal_(mean, std)
            if pad_token_id is not None:
                m.weight.data[pad_token_id].zero_()

### Architectural Blocks ###

class LayerStack(nn.ModuleList):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class LayerBlock(nn.ModuleDict):
    def forward(self, x):
        for k, v in self.items():
            x = v(x)
        return x

class PreNormResidualBlock(nn.ModuleDict):
    layer_order = ("norm", "mixer", "dropout")
    def __init__(self, dct):
        super().__init__()
        for key in self.layer_order:
            if key in dct:
                self[key] = dct[key]
        
    def forward(self, x):
        residual = x
        for key in self.layer_order:
            x = self[key](x)
        x = x + residual
        return x

class PostNormResidualBlock(nn.ModuleDict):
    layer_order = ("mixer", "dropout", "norm")
    def __init__(self, dct):
        super().__init__()
        for key in self.layer_order:
            if key in dct:
                self[key] = dct[key]
                
    def forward(self, x):
        residual = x
        for key in self.layer_order:
            if key == "norm":
                x = x + residual
            x = self[key](x)
        return x


class ReZeroResidualBlock(nn.ModuleDict):
    layer_order = ("mixer", "norm", "dropout")
    def __init__(self, dct):
        super().__init__()
        for key in self.layer_order:
            if key == "norm": # ignore whatever was passed and use ReZero
                self[key] = ReZero()
            elif key in dct:
                self[key] = dct[key]

    def forward(self, x):
        residual = x
        for key in self.layer_order:
            x = self[key](x)
        x = x + residual
        return x


class ReZeroBlockWithNorm(nn.ModuleDict):
    layer_order = ("norm", "mixer", "rezero", "dropout")
    def __init__(self, dct):
        super().__init__()
        for key in self.layer_order:
            if key == "rezero": # ignore whatever was passed and use ReZero
                self[key] = ReZero()
            elif key in dct:
                self[key] = dct[key]

    def forward(self, x):
        residual = x
        for key in self.layer_order:
            x = self[key](x)
        x = x + residual
        return x



class DecoderBackbone(nn.Module):
    def __init__(self, num_layers, model_size, seq_mixer, ff_mixer, norm, dropout, residual_block):
        super().__init__()
        seq_mixer, seq_kwargs = self._unpack(seq_mixer)
        ff_mixer, ff_kwargs = self._unpack(ff_mixer)
        norm, norm_kwargs = self._unpack(norm)
        self.model_size = model_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            seq_block = residual_block({
                "mixer": seq_mixer(self.model_size, **seq_kwargs),
                "dropout": nn.Dropout(dropout),
                "norm": norm(self.model_size, **norm_kwargs)
            })
            ff_block = residual_block({
                "mixer": ff_mixer(self.model_size, **ff_kwargs),
                "dropout": nn.Dropout(dropout),
                "norm": norm(self.model_size, **norm_kwargs)
            })
            layer = LayerBlock({
                "seq_block": seq_block,
                "ff_block": ff_block,
            })
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _unpack(self, x):
        return x if isinstance(x, tuple) else (x, {})

class CyclingDecoderBackbone(nn.Module):
    def __init__(self, num_layers, model_size, seq_mixer, ff_mixer, norm, dropout, residual_block):
        super().__init__()
        seq_mixer, seq_kwargs = self._unpack(seq_mixer)
        ff_mixer, ff_kwargs = self._unpack(ff_mixer)
        norm, norm_kwargs = self._unpack(norm)
        self.model_size = model_size
        self.num_layers = num_layers
        self.seq_block = residual_block({
            "mixer": seq_mixer(self.model_size, **seq_kwargs),
            "dropout": nn.Dropout(dropout),
            "norm": norm(self.model_size, **norm_kwargs)
        })
        self.ff_block = residual_block({
            "mixer": ff_mixer(self.model_size, **ff_kwargs),
            "dropout": nn.Dropout(dropout),
            "norm": norm(self.model_size, **norm_kwargs)
        })
        

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.seq_block(x)
            x = self.ff_block(x)
        return x

    def extra_repr(self):
        return f"(Number of repeating layers: {self.num_layers})"

    def _unpack(self, x):
        return x if isinstance(x, tuple) else (x, {})

# I'm trying to figure out how to implement the alternating decoder backbone, because there are at least two recent models that use such a setup, and also I have a specific thing to baseline. 
# I think it makes sense to simply assume a list of unpackable arguements.
# The problem...oh...actually...
class AlternatingDecoderBackbone(nn.Module):
    def __init__(self, num_layers, model_size, seq_mixer, ff_mixer, norm, dropout, residual_block):
        super().__init__()
        # repeat these sequences up until the number of layers is reached or exceeded.
        if not isinstance(seq_mixer, list):
            seq_mixer = [seq_mixer]
        if not isinstance(ff_mixer, list):
            ff_mixer = [ff_mixer]
        seq_mixers_kwargs = []
        while len(seq_mixers_kwargs) < num_layers:
            seq_mixers_kwargs += [self._unpack(x) for x in seq_mixer]
        ff_mixers_kwargs = []
        while len(ff_mixers_kwargs) < num_layers:
            ff_mixers_kwargs += [self._unpack(x) for x in ff_mixer]
        norm, norm_kwargs = self._unpack(norm)
        self.model_size = model_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            seq_mixer_, seq_mixer_kwargs = seq_mixers_kwargs[i]
            ff_mixer_, ff_mixer_kwargs = ff_mixers_kwargs[i]
            seq_block = residual_block({
                "mixer": seq_mixer_(self.model_size, **seq_mixer_kwargs),
                "dropout": nn.Dropout(dropout),
                "norm": norm(self.model_size, **norm_kwargs)
            })
            ff_block = residual_block({
                "mixer": ff_mixer_(self.model_size, **ff_mixer_kwargs),
                "dropout": nn.Dropout(dropout),
                "norm": norm(self.model_size, **norm_kwargs)
            })
            layer = LayerBlock({
                "seq_block": seq_block,
                "ff_block": ff_block,
            })
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _unpack(self, x):
        return x if isinstance(x, tuple) else (x, {})


### Base class, more-or-less tested
from functools import partial
class MixerModel(nn.Module):
    def __init__(
        self,
        tokenizer = None,
        vocab_size = None,
        pad_token_id = None,
        model_size = None,
        num_layers = 1,
        max_seq_len = None,
        vectorizer = EmbeddingAndPositionalVectorizer,
        seq_mixer =(AttentionMixer, {"num_heads": 1}),
        ff_mixer = (MLPMixer, {"expansion": 4}),
        norm = nn.LayerNorm,
        head = CausalLanguageModelHead,
        init_strategy = (init_weights_normal, {"mean": 0.0, "std": 0.02}),
        dropout = 0.0,
        residual_block= PreNormResidualBlock,
        use_initial_norm = False,
        use_final_norm = True,
        decoder_backbone = DecoderBackbone
    ):
        super().__init__()
        ## Basic specs
        self.model_size = model_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        ## Handle stuff related to tokenization
        if tokenizer is None:
            assert vocab_size is not None, "If no tokenizer is provided, you must provide a vocab size."
            if pad_token_id is None:
                print("No pad_token_id was provided.  Assuming 0.")
                pad_token_id = 0
        else:
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.vocab_size = tokenizer.vocab_size
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        ## Vectorizer
        vectorizer, vectorizer_kwargs = self._unpack(vectorizer)
        norm_init, norm_kwargs = self._unpack(norm)
        self.vectorizer = vectorizer(vocab_size, model_size, dropout = dropout, max_seq_len = max_seq_len, **vectorizer_kwargs)
        ## Model backbone
        self.initial_norm = None if not use_initial_norm else norm_init(model_size, **norm_kwargs)        
        self.residual_block = residual_block
        self.decoder = decoder_backbone(num_layers, model_size, seq_mixer, ff_mixer, norm, dropout, residual_block)
        self.final_norm = None if not use_final_norm else norm_init(model_size, **norm_kwargs)
        ## Task head
        head, head_kwargs = self._unpack(head)
        self.head = head(model_size, tokenizer = self.tokenizer, vocab_size = self.vocab_size, pad_token_id = self.pad_token_id, **head_kwargs)
        ## Initialize weights
        init_strategy, init_kwargs = self._unpack(init_strategy)
        self.init_strategy = partial(init_strategy, **init_kwargs)
        self.initialize()
        ## Tie weights
        if self.head.tied_weights:
            print("note: tying weights")
            self.head.tie_weights(self.vectorizer.get_tieable_weights())

    def _unpack(self, x):
        return x if isinstance(x, tuple) else (x, {})

    def initialize(self, init_strategy = None, **kwargs):
        if init_strategy is not None:
            self.init_strategy = partial(init_strategy, **kwargs)
        self.init_strategy(self)

    def forward(self, x):
        x = self.vectorizer(x)
        x = x if self.initial_norm is None else self.initial_norm(x)
        x = self.decoder(x)
        x = x if self.final_norm is None else self.final_norm(x)
        x = self.head(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self, include_embeddings = True):
        params = [param for param in self.parameters()  if include_embeddings or not param.shape[0] == self.vocab_size]
        return sum(param.numel() for param in params)