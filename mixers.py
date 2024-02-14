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


### Vectorizers ###

class EmbeddingVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = None, tied_weights = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.tied_weights = tied_weights

    def forward(self, x):
        x = self.embedding(x)
        return x

    def get_tieable_weights(self):
        return self.embedding.weight


class EmbeddingAndPositionalVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.positional_embedding = nn.Embedding(max_seq_len, model_size)

    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device = x.device)
        pos_emb = self.positional_embedding(positions)
        x = self.embedding(x) + pos_emb
        return x

    def get_tieable_weights(self):
        return self.embedding.weight


class UpscalingEmbeddingsVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512, scaling = 8): # I'm not sure if you can do tied weights with this?
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_size // scaling)
        self.positional_embedding = nn.Embedding(max_seq_len, model_size // scaling) # It's counterintuitive to me that we scale down the positional embeddings but that's what ALBERT does
        self.emb_proj = nn.Linear(model_size // scaling, model_size)

    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device = x.device)
        emb = self.embedding(x)
        pos_emb = self.positional_embedding(positions)
        x = emb + pos_emb
        x = self.emb_proj(x)
        return x

    def get_tieable_weights(self):
        return self.embedding.weight


class OneHotVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = None):
        super().__init__()
        self.vocab_size = vocab_size
        # the model size has to equal the vocab size plus the max sequence length

    def forward(self, x):
        return torch.nn.functional.one_hot(x, self.vocab_size).float()

    
class OneHotAndPositionalVectorizer(nn.Module):
    def __init__(self, vocab_size, model_size, max_seq_len = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, x):
        b, s = x.shape
        one_hot = F.one_hot(x, self.vocab_size).float()
        positions = torch.arange(s, device = x.device).unsqueeze(0).expand(b, -1)
        pos_emb = F.one_hot(positions, self.max_seq_len).float()
        x = torch.cat([one_hot, pos_emb], dim = -1)
        return x


### Sequence Mixers ###

class AttentionMixer(nn.Module):
    def __init__(self, model_size, num_heads = 1, dropout = 0.0):
        super().__init__()
        self.model_size = model_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(model_size, num_heads = num_heads, dropout = dropout, bias = False, batch_first = True)
        self.requires_mask = True

    def forward(self, x):
        b, s, l = x.shape
        x = self.attn(x, x, x, is_causal = True, attn_mask = self.get_causal_mask(x, s))[0]
        return x

    causal_masks = {}
    def get_causal_mask(cls, x, s):
        if s not in cls.causal_masks or cls.causal_masks[s].device != x.device:
            cls.causal_masks[s] = nn.Transformer.generate_square_subsequent_mask(s).to(x.device)
        return cls.causal_masks[s]


class SeqConv(nn.Module):
    def __init__(self, model_dim, kernel_size):
        super().__init__()
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(model_dim, model_dim, kernel_size, padding = kernel_size - 1, groups = model_dim)
    
    def forward(self, x):
        b, l, d = x.shape
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :l]
        x = x.transpose(1, 2)
        return x


class SeqConvMixer(nn.Module):
    def __init__(self, model_size, kernel_size = 3):
        super().__init__()
        self.model_size = model_size
        self.kernel_size = kernel_size
        self.seq_conv = SeqConv(model_size, kernel_size)

    def forward(self, x):
        x = self.seq_conv(x)
        x = torch.relu(x)
        return x


class GatedConvSeqMixer(nn.Module):
    def __init__(self, model_size, conv_kernel_size = 3, gate_kernel_size = 3):
        super().__init__()
        self.model_size = model_size
        self.conv_kernel_size = conv_kernel_size
        self.gate_kernel_size = gate_kernel_size
        self.conv = SeqConv(model_size, conv_kernel_size)
        self.gate = SeqConv(model_size, gate_kernel_size)

    def forward(self, x):
        up = self.conv(x)
        gate = self.gate(x)
        gate = torch.relu(gate)
        x = up * gate
        return x


### State Mixers ###

class MLPMixer(nn.Module):
    def __init__(self, model_size, expansion = 4):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        self.fc1 = nn.Linear(model_size, model_size * expansion)
        self.fc2 = nn.Linear(model_size * expansion, model_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class GatedStateMixer(nn.Module): # I'm pretty sure that things like SwiGLU already do this.
    def __init__(self, model_size, expansion = 4):
        super().__init__()
        self.model_size = model_size
        self.expansion = expansion
        self.fc1 = nn.Linear(model_size, model_size * expansion)
        self.gate = nn.Linear(model_size, model_size * expansion)
        self.fc2 = nn.Linear(model_size * expansion, model_size)

    def forward(self, x):
        up = self.fc1(x)
        gate = self.gate(x)
        gate = torch.relu(gate)
        x = up * gate
        x = self.fc2(x)
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
        self.fc = nn.Linear(model_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.tied_weights = tied_weights

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x

    def tie_weights(self, wt):
        self.fc.weight = wt


class DownscalingLanguageModelHead(CausalLanguageModelHead):
    def __init__(self, model_size, tokenizer = None, vocab_size = None, pad_token_id = None, dropout = 0.0, loss_fn = nn.CrossEntropyLoss(), tied_weights = True, scaling = 8):
        super().__init__(model_size, vocab_size = vocab_size, pad_token_id = pad_token_id, dropout = dropout, loss_fn = loss_fn, tied_weights = tied_weights)
        self.cls_proj = nn.Linear(model_size, model_size // scaling)
        self.fc = nn.Linear(model_size // scaling, vocab_size)

    def forward(self, x):
        x = self.cls_proj(x)
        x = self.fc(x)
        return x


### Model Backbone ###

# ! Unsolved problem: What if we want to pass along hidden states, attentions, et cetera?  Is that rare enough that forward hooks would be the way to go?
class MixerModel(nn.Module):
    def __init__(
        self,
        tokenizer = None,
        vocab_size = None,
        pad_token_id = None,
        model_size = None,
        num_layers = 1,
        #num_heads = 1, think about whether to implement this
        max_seq_len = None,
        vectorizer = EmbeddingAndPositionalVectorizer,
        seq_mixer =(AttentionMixer, {"num_heads": 1}),
        ff_mixer = (MLPMixer, {"expansion": 4}),
        norm = (nn.LayerNorm, {"elementwise_affine": False}),
        head = CausalLanguageModelHead,
        dropout = 0.0,
        use_residuals = True,
        block_order = "rsnrfn" # not yet implemented
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

        unpack = lambda x: x if isinstance(x, tuple) else (x, {})
        vectorizer, vectorizer_kwargs = unpack(vectorizer)
        norm, norm_kwargs = unpack(norm)
        seq_mixer, seq_kwargs = unpack(seq_mixer)
        ff_mixer, ff_kwargs = unpack(ff_mixer)
        head, head_kwargs = unpack(head)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.model_size = model_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.vectorizer = vectorizer(vocab_size, model_size, max_seq_len = max_seq_len, **vectorizer_kwargs)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = norm(model_size, **norm_kwargs)
        self.decoder = nn.ModuleList()
        self.use_residuals = use_residuals
        self.block_order = block_order
        for i in range(num_layers):
            seqm = seq_mixer(model_size, **seq_kwargs)
            drop1 = nn.Dropout(dropout)
            norm1 = norm(model_size, **norm_kwargs)
            ffm = ff_mixer(model_size, **ff_kwargs)
            drop2 = nn.Dropout(dropout)
            norm2 = norm(model_size, **norm_kwargs)
            layer = nn.ModuleDict({
                "seq_mixer": seqm,
                "drop1": drop1,
                "norm1": norm1,
                "ff_mixer": ffm,
                "drop2": drop2,
                "norm2": norm2
            })
            self.decoder.append(layer)
        
        self.head = head(model_size, tokenizer = tokenizer, vocab_size = vocab_size, pad_token_id = pad_token_id, dropout = dropout, **head_kwargs) # Do I need to do something to make sure the arguments don't conflict?
        if self.head.tied_weights:
            self.head.tie_weights(self.vectorizer.get_tieable_weights())
        # check if the vectorizer and the fc layer should be tied
        if hasattr(self.vectorizer, "tied_weights") and self.vectorizer.tied_weights:
            self.vectorizer.tie_weights(self.classifier)

    def forward(self, x):
        x = self.vectorizer(x)
        x = self.embed_norm(x) # there's also the issue of whether this comes before or after the mixer in various models
        x = self.embed_dropout(x)
        for layer in self.decoder:
            residual = x
            x = layer["seq_mixer"](x)
            x = x + residual if self.use_residuals else x
            x = layer["drop1"](x)
            x = layer["norm1"](x)
            residual = x
            x = layer["ff_mixer"](x)
            x = x + residual if self.use_residuals else x
            x = layer["drop2"](x)
            x = layer["norm2"](x)
        x = self.head(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())