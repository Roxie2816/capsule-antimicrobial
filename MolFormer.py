import argparse
import glob
import torch
import shutil
import rdkit
from torch import nn
# import args
import os
import numpy as np
import random
import getpass
import regex as re
import torch
import numpy as np
import random
import collections
import math

class Encoder():

    def __init__(self, max_length=500, add_bos=True, add_eos=True, feature_size=32):
        self.vocab_encoder = torch.load('./model/pubchem_canon_zinc_final_vocab_sorted.pth')

        self.max_length = max_length
        self.min_length = 1
        self.mod_length = 42
        self.mlm_probability = .15
        self.avg_length = 66
        self.tail = 122
        self.b0_cache=collections.deque()
        self.b1_cache=collections.deque()
        self.b2_cache=collections.deque()
        self.b3_cache=collections.deque()
        self.bucket0=collections.deque()
        self.bucket1=collections.deque()
        self.bucket2=collections.deque()
        self.bucket3=collections.deque()
        if feature_size == 32:
            self.b0_max=1100
            self.b1_max=700
            self.b2_max=150
            self.b3_max=50
        else:
            self.b0_max=1382
            self.b1_max=871
            self.b2_max=516
            self.b3_max=311
        values = list(self.vocab_encoder.values())
        num_top = 0
        middle_top = 0
        bottom = 0
        for  count in values:
            if count > 100000:
                num_top += 1
            if count > 50:
                middle_top += 1
        middle_top = middle_top - num_top
        self.cutoffs = [num_top+4, middle_top]
        self.char2id = {"<bos>":0, "<eos>":1, "<pad>":2, "<mask>":3}
        self.id2char = {0:"<bos>", 1:"<eos>", 2:"<pad>", 3:"<mask>"}
        self.pad  = self.char2id['<pad>']
        self.mask = self.char2id['<mask>']
        self.eos  = self.char2id['<eos>']
        self.bos  = self.char2id['<bos>']
        pos = 0
        for key, value in self.vocab_encoder.items():
        #for pos, key in enumerate(self.vocab_encoder.keys()):
            self.char2id[key] = pos+4
            self.id2char[pos+4] = key
            pos += 1
        self.char2id["<unk>"] = pos + 4
        self.id2char[pos+4] = "<unk>"
        self.pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.add_bos = add_bos
        self.add_eos = add_eos

    def encode(self, char):
        #if len(char) > self.max_length:
        #    char = char[:self.max_length]
        if self.add_bos == True:
            char = ['<bos>'] + char
        if self.add_eos == True:
            char = char + ['<eos>']

        return torch.tensor([self.char2id[word] for word in char])

    def encoder(self, tokens):
        #return *map(lambda x: self.encode(x), tokens)
        return [self.encode(mol) for mol in tokens]

    def process_text(self, text):
        #print(text)
        #random length sequences seems to help training
        mod_length = self.mod_length #+ random.randint(-1, 3)
        avg_length = self.avg_length #+ random.randint(-3, 5)
        for mol in text:
            #fill up buckets and caches
            if '\n' in mol['text']:
                print('carriage return in mol')
            raw_regex = self.regex.findall(mol['text'].strip('\n'))
            length = len(raw_regex)
            if length > self.min_length and length < mod_length:
                if len(self.bucket0) < self.b0_max:
                    self.bucket0.append(raw_regex)
                else:
                    self.b0_cache.append(raw_regex)
            elif length >= mod_length and length < avg_length:
                if len(self.bucket1) < self.b1_max:
                    self.bucket1.append(raw_regex)
                else:
                    self.b1_cache.append(raw_regex)
            elif length >= avg_length and length < self.tail:
                self.b2_cache.append(raw_regex)
                #if len(bucket2) < self.b2_max:
                #    bucket2.append(raw_regex)
                #else:
                #    self.b2_cache.append(raw_regex)
            elif length >= self.tail and length < self.max_length:
                self.b3_cache.append(raw_regex)
                #if len(bucket3) < self.b3_max:
                #    bucket3.append(raw_regex)
                #else:
                #    self.b3_cache.append(raw_regex)

        #print('before Cache size  {} {} {} {}'.format(len(self.b0_cache), len(self.b1_cache), len(self.b2_cache), len(self.b3_cache)))
        #pour cache elements into any open bucket
        if len(self.bucket0) < self.b0_max and len(self.b0_cache) > 0:
            cache_size = len(self.b0_cache)
            max_margin = self.b0_max-len(self.bucket0)
            range0 = min(cache_size, max_margin)
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))] + [self.b0_cache.pop() for i in range(range0)]
            #self.b0_cache =  collections.deque(self.b0_cache[:self.b0_max-len(bucket0)])
            #print('0 type {}'.format(type(self.b0_cache)))
        else:
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))]
        if len(self.bucket1) < self.b1_max and len(self.b1_cache) > 0:
            cache_size = len(self.b1_cache)
            max_margin = self.b1_max-len(self.bucket1)
            range1 = min(cache_size, max_margin)
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))] + [self.b1_cache.pop() for i in range(range1)]
        else:
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))]

        if len(self.b2_cache) > self.b2_max:
            cache_size = len(self.b2_cache)
            max_margin = self.b2_max
            range2 = min(cache_size, max_margin)
            outbucket2 =  [self.b2_cache.pop() for i in range(range2)]
        else:
            outbucket2=[]
        if len(self.b3_cache) > self.b3_max:
            cache_size = len(self.b3_cache)
            max_margin = self.b3_max
            range3 = min(cache_size, max_margin)
            outbucket3 =  [self.b3_cache.pop() for i in range(range3)]
        else:
            outbucket3 = []
        return outbucket0, outbucket1, outbucket2, outbucket3

    def mask_tokens( self, inputs, special_tokens_mask= None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.size(), self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            #special_tokens_mask = special_tokens_mask.bool()

        #print(special_tokens_mask.size())
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id.keys()), labels.size(), dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    def pack_tensors(self, tokens):
        array = self.encoder(tokens)
        array =  torch.nn.utils.rnn.pad_sequence(array, batch_first=True, padding_value=self.pad)
        #lengths = (array!=self.pad).sum(dim=-1)
        #Bert tokenization
        special_token_mask = [list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff)) for stuff in array.tolist()]
        masked_array, masked_labels = self.mask_tokens(array, special_token_mask)
        return masked_array, masked_labels#, lengths
    def process(self, text):
        arrays = []
        #lengths = []
        targets = []
        for tokens in self.process_text(text):
            if len(tokens) > 0:
                array, target = self.pack_tensors(tokens)
                arrays.append(array)
                targets.append(target)
        return arrays, targets

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# from fast_transformers.builders import TransformerEncoderBuilder
# from fast_transformers.masking import LengthMask as LM
from fast_transformers.feature_maps import Favor,GeneralizedRandomFeatures
import torch.nn.functional as F
from functools import partial
# from apex import optimizers
# from transformers.optimization import Lamb
# 来源：HuggingFace transformers 3.5.1

from torch.nn import LayerNorm

from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder
from fast_transformers.builders.attention_builders import AttentionBuilder

"""The rotate attention layer performs all the query key value projections and
output projections leaving the implementation of the attention to the inner
attention module.
"""

from torch.nn import Linear, Module

from fast_transformers.attention import AttentionLayer
from fast_transformers.events import EventDispatcher, QKVEvent
import torch

class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = 0 
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            #if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None,:, None, :]
            self.sin_cached = emb.sin()[None,:, None, :]
            #else:
            #    cos_return = self.cos_cached[..., :seq_len]
            #    sin_return = self.sin_cached[..., :seq_len]
            #    return cos_return, sin_return
                
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotateAttentionLayer(AttentionLayer):
    """Rotate attention layer inherits from fast_transformer attention layer. 
        The only thing added is an Embedding encoding, for more information
        on the attention layer see the fast_transformers code
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(RotateAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using Rotation Embedding')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected. 
        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer

class Lamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                weight_norm = p.data.pow(2).sum().sqrt()
                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss
    
class LightningModule(pl.LightningModule):

    def __init__(self, config, vocab):
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        #location of cache File
        # Special symbols

        self.debug = config.debug
        self.text_encoder = Encoder(config.max_len)
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        # input embedding stem
        builder = RotateEncoderBuilder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            #attention_type='full',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        if config.restart_path == "":
            seed_everything(config.seed)




    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.99)
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        # optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        optimizer = Lamb(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idxl =     batch[0]
        targetsl = batch[1]
        #lengthsl = batch[2]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            #masking of the length of the inputs its handled in the Masked language part of the code
            #do not attempt to handle it in the forward of the transformer
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs/len(idxl)
            if chunk < len(idxl)-1:
                loss_tmp.backward()
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log('train_loss', loss, on_step=True)
        return {'loss':loss}#, 'log':tensorboard_log}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        loss = {'loss': avg_loss.item()}
        self.log('validation_loss', loss['loss'])
    def validation_step(self, batch, batch_idx):
        idxl =     batch[0]
        targetsl = batch[1]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs/len(idxl)
            if chunk < len(idxl)-1:
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log('train_loss', loss, on_step=True)
        return {'loss':loss}

from transformers import BertTokenizer
import regex as re

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string

import torch
from fast_transformers.masking import LengthMask as LM
from rdkit import Chem
from pathlib import Path
from argparse import Namespace
import yaml
from typing import List, Optional
import numpy as np
import pandas as pd
# def molformer(smiles_list: List[str]) -> np.ndarray:
def molformer(smiles_list: List[str]) -> pd.DataFrame:


    # 1. 加载配置和模型
    def _load_model():
        """加载预训练模型和tokenizer"""
        try:
            with open('./model/hparams.yaml', 'r') as f:
                config = Namespace(**yaml.safe_load(f))
            
            tokenizer = MolTranBertTokenizer('./model/bert_vocab.txt')
            
            ckpt = Path('./model/N-Step-Checkpoint_3_30000.ckpt')
            
            # from train_pubchem_light import LightningModule
            model = LightningModule.load_from_checkpoint(
                checkpoint_path=ckpt,
                config=config,
                vocab=tokenizer.vocab
            )
            model.eval()
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    # 2. 批处理生成器
    def _batch_split(data: List[str], batch_size: int = 64):
        """将数据分割成批次"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # 3. 规范化SMILES
    def _canonicalize(smiles: str) -> Optional[str]:
        """规范化SMILES字符串"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else None
        except:
            return None

    # 4. 嵌入生成函数
    def _embed(model, tokenizer, smiles_batch: List[str]) -> torch.Tensor:
        """生成单个批次的嵌入"""
        batch_enc = tokenizer.batch_encode_plus(
            smiles_batch, 
            padding=True, 
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            token_embeddings = model.blocks(
                model.tok_emb(batch_enc['input_ids']),
                length_mask=LM(batch_enc['attention_mask'].sum(-1))
            )
        
        # 平均池化
        mask = batch_enc['attention_mask'].unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()

    # 主处理流程
    try:
        # 规范化SMILES并过滤无效项
        valid_smiles = []
        for s in smiles_list:
            canon = _canonicalize(s)
            if canon:
                valid_smiles.append(canon)
        
        if not valid_smiles:
            raise ValueError("没有有效的SMILES输入")
        
        # 加载模型
        model, tokenizer = _load_model()
        
        # 分批处理生成嵌入
        embeddings = []
        for batch in _batch_split(valid_smiles):
            embeddings.append(_embed(model, tokenizer, batch))
        embeddings_array = torch.cat(embeddings).numpy()
        
        # 创建DataFrame并添加列名
        columns = [f"molformer_{i}" for i in range(embeddings_array.shape[1])]
        df = pd.DataFrame(embeddings_array, columns=columns)
        # 合并结果并转换为numpy
        # return torch.cat(embeddings).numpy()
        return df
    
    except Exception as e:
        raise RuntimeError(f"嵌入生成失败: {str(e)}")
    
if __name__ == '__main__':
    # Example SMILES strings
    smiles_list = ['CCO', 'C1CCCCC1', 'CC(=O)O']  # Note: fixed the input to be a list of strings
    
    try:
        # Get embeddings
        embeddings = molformer(smiles_list)
        
        # Optional: Also save as text file for human readability
        np.savetxt('molformer_embeddings.csv', embeddings)
        print(f"Text version saved to molformer_embeddings.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")