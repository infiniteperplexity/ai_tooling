import functools
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

### Dummies ###

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


### Norms ###

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
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

# written from scratch
class AttentionMixer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout = 0.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.k_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.q_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.v_proj = nn.Linear(model_dim, model_dim, bias = False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias = False)
        for weights in [self.k_proj, self.q_proj, self.v_proj, self.out_proj]:
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
    #def __init__(self, model_size, expansion = 4, bias = False, activation = F.gelu):
    def __init__(self, model_size, expansion = 4, bias = False, activation = nn.GELU):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        self.fc1 = nn.Linear(model_size, model_size * expansion, bias = bias)
        if isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation

        self.fc2 = nn.Linear(model_size * expansion, model_size, bias = bias)

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
        # d needs to be split amongs the various heads. 
        split = torch.split(x, [h.model_size for h in self.heads], dim = -1)
        split_out = [h(s) for h, s in zip(self.heads, split)]
        x = torch.cat(split_out, dim = -1)
        return x


class GatedStateMixer(nn.Module): # I'm pretty sure that things like SwiGLU already do this.
    def __init__(self, model_size, expansion = 4, bias = False, activation = F.gelu):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        self.fc1 = nn.Linear(model_size, model_size * expansion, bias = bias)
        self.gate = nn.Linear(model_size, model_size * expansion)
        self.fc2 = nn.Linear(model_size * expansion, model_size, bias = bias)
        self.activation = activation

    def forward(self, x):
        up = self.fc1(x)
        gate = self.gate(x)
        gate = self.activation(gate)
        x = up * gate
        x = self.fc2(x)
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
    def __init__(self, model_size, tokenizer = None, vocab_size = None, pad_token_id = None, dropout = 0.0, loss_fn = nn.CrossEntropyLoss(), tied_weights = True):
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
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.tied_weights = tied_weights

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x

    def tie_weights(self, wt):
        assert wt.shape == self.fc.weight.shape, "The weight matrices must have the same shape to be tied."
        self.fc.weight = wt


class DownscalingLanguageModelHead(CausalLanguageModelHead):
    def __init__(self, model_size, tokenizer = None, vocab_size = None, pad_token_id = None, dropout = 0.0, loss_fn = nn.CrossEntropyLoss(), tied_weights = True, scaling = 8):
        super().__init__(model_size, vocab_size = vocab_size, pad_token_id = pad_token_id, dropout = dropout, loss_fn = loss_fn, tied_weights = tied_weights)
        self.cls_proj = nn.Linear(model_size, model_size // scaling, bias = False)
        self.fc = nn.Linear(model_size // scaling, vocab_size, bias = False)

    def forward(self, x):
        x = self.cls_proj(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x

### Initialization Strategies ###

def init_model_normal(mdl, mean = 0.0, std = 0.02, tokenizer = None, pad_token_id = None):
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


### Base class, more-or-less tested
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
        norm = (nn.LayerNorm, {"elementwise_affine": False}),
        head = CausalLanguageModelHead,
        init_strategy = (init_model_normal, {"mean": 0.0, "std": 0.02}),
        dropout = 0.0,
        residual_block= PreNormResidualBlock,
        use_initial_norm = False,
        use_final_norm = True,
        decoder_backbone = DecoderBackbone
    ):
        super().__init__()
        if tokenizer is None:
            assert vocab_size is not None, "If no tokenizer is provided, you must provide a vocab size."
            if pad_token_id is None:
                print("No pad_token_id was provided.  Assuming 0.")
                pad_token_id = 0
        else:
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id

        vectorizer, vectorizer_kwargs = self._unpack(vectorizer)
        norm_init, norm_kwargs = self._unpack(norm)
        head, head_kwargs = self._unpack(head)
        init, init_kwargs = self._unpack(init_strategy)
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.vocab_size = tokenizer.vocab_size
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
        self.model_size = model_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.init_strategy = init(self, tokenizer = self.tokenizer, pad_token_id = self.pad_token_id, **init_kwargs) # I honestly wonder if this should go in the Trainer instead?
        self.vectorizer = vectorizer(vocab_size, model_size, dropout = dropout, max_seq_len = max_seq_len, **vectorizer_kwargs)
        self.initial_norm = None if not use_initial_norm else norm_init(model_size, **norm_kwargs)        
        self.residual_block = residual_block
        self.decoder = decoder_backbone(num_layers, model_size, seq_mixer, ff_mixer, norm, dropout, residual_block)
        self.final_norm = None if not use_final_norm else norm_init(model_size, **norm_kwargs)
        self.head = head(model_size, tokenizer = self.tokenizer, vocab_size = self.vocab_size, pad_token_id = self.pad_token_id, dropout = dropout, **head_kwargs) # Do I need to do something to make sure the arguments don't conflict?
        if self.head.tied_weights:
            print("note: tying weights")
            self.head.tie_weights(self.vectorizer.get_tieable_weights())

    def _unpack(self, x):
        return x if isinstance(x, tuple) else (x, {})

    def forward(self, x):
        x = self.vectorizer(x)
        x = self.embed_dropout(x)
        x = x if self.initial_norm is None else self.initial_norm(x)
        x = self.decoder(x)
        x = x if self.final_norm is None else self.final_norm(x)
        x = self.head(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
