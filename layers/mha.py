''' 
MultiHeadAttention layers
'''
import torch
import torch.nn as nn
from torch.nn import init
from . import time_modulator as tmod
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, input_qdim, input_kdim, input_vdim, embed_dim, num_heads, output_dim = 16,
                 is_q_proj = True, is_k_proj = True, is_v_proj = True, is_output_proj = True,
                 qk_same_length = False, which_linear = nn.Linear, dropout = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads." 
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        
        self.input_qdim = input_qdim
        self.input_kdim = input_kdim
        self.input_vdim = input_vdim
        self.output_dim = output_dim
        
        self.is_q_proj      = is_q_proj
        self.is_k_proj      = is_k_proj
        self.is_v_proj      = is_v_proj
        self.is_output_proj = is_output_proj
        self.proj_together  = is_q_proj  and is_k_proj and is_v_proj and \
                              input_qdim == input_kdim == input_vdim and \
                              qk_same_length

        if self.proj_together:
            self.qkv_proj = which_linear(input_qdim, 3*embed_dim)
        else:
            if is_q_proj:
                self.q_proj   = which_linear(input_qdim, embed_dim)
            if is_k_proj:
                self.k_proj   = which_linear(input_kdim, embed_dim)
            if is_v_proj:
                self.v_proj   = which_linear(input_vdim, embed_dim)
        if is_output_proj:
            self.o_proj   = which_linear(embed_dim, self.output_dim)

        self.dropout_layer  = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)

    def scaled_dot_product(self, q, k, v, mask=None, causal_mask = False):
        d_k = q.size()[-1] 
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).permute(0, 1, 3, 2)           
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        if causal_mask:
            i, j        = attn_logits.shape[-2:]
            mask_aux    = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            attn_logits = attn_logits.masked_fill(mask_aux, -9e15)

        attn_logits = attn_logits - attn_logits.amax(dim = -1, keepdim = True).detach()
        attention = F.softmax(attn_logits, dim=-1)
        attention = self.dropout_layer(attention)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, value, key = None, query = None, mask=None, return_attention=False,
                                                    reshape = True, causal_mask = False):
        bs, sl, v_dim = value.size() #batch_size, source length, vdim
        tl = sl if query is None else query.size(1)
        
        if self.proj_together:
            tl  = sl
            qkv = self.qkv_proj(value)
            # Separate Q, K, V from linear output
            qkv = qkv.reshape(bs, sl, self.num_heads, 3*self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q  = self.q_proj(query).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) \
                     if self.is_q_proj else query.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            k  = self.k_proj(key).reshape(bs, sl, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  \
                 if self.is_k_proj else key.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            v  = self.v_proj(value).reshape(bs, sl, self.num_heads, self.head_dim).permute(0, 2, 1, 3) \
                 if self.is_v_proj else value.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

            # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask = mask, causal_mask = causal_mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]

        if self.is_output_proj:
            o  = self.o_proj(values.reshape(bs, tl, self.embed_dim)) 
        else:
            o   = values.reshape(bs, tl, -1) \
                 if reshape else values

        if return_attention:
            return o, attention.permute(0, 2, 1, 3)
        else:
            return o, None
    
class MultiheadAttentionHandler(nn.Module):
    def __init__(self, input_dim, head_dim, num_heads, which_linear = nn.Linear, dropout = 0.0, **kwargs):
        super().__init__()
        
        self.input_dim  = input_dim
        self.embed_dim  = head_dim * num_heads
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self.mod_dim    = self.head_dim  

        ### Projection to latent variable ####
        self.mha_lc = MultiheadAttention(input_dim, input_dim, input_dim,
                self.embed_dim, num_heads, output_dim = input_dim,
                is_q_proj = True, is_k_proj = True,
                is_v_proj = True, is_output_proj = True,
                which_linear = which_linear, dropout = dropout)

    def get_input_dim(self):
        return self.embed_dim
    def get_last_dim(self):
        return self.embed_dim

    def forward(self, emb_x, mask=None, return_attention=False, causal_mask = False):
        emb, attention = self.mha_lc(emb_x, key = emb_x,
                                    query = emb_x, mask = mask,
                                    return_attention = return_attention,
                                    causal_mask = causal_mask)
        return emb

# Arguments for parser
def add_sample_parser(parser):
  parser.add_argument(
    '--head_dim', type=int, default = 128,
    help='Number of head dimensions'
         '(default: %(default)s)')
  parser.add_argument(
    '--num_heads', type=int, default = 4,
    help='Number of heads'
         '(default: %(default)s)')
  return parser

def add_name_config(config):
  name = []
  if config['which_encoder'] == 'mha' or config['which_decoder'] == 'mha':
    name += [
      'MHA',
      'HD%d' % config['head_dim'],
      'NHead%d' % config['num_heads']]
  return name