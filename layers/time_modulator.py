''' 
Time modulator
'''
import torch
import torch.nn as nn
import math


class TimeModulator(nn.Module):
    def __init__(self, M, embed_dim, T_max):
        super().__init__()
        self.alpha_sin = nn.Parameter(torch.randn(M, embed_dim))
        self.alpha_cos = nn.Parameter(torch.randn(M, embed_dim))
        self.beta_sin  = nn.Parameter(torch.randn(M, embed_dim))
        self.beta_cos  = nn.Parameter(torch.randn(M, embed_dim))
        self.T_max     = T_max
        self.M         = M
        self.register_buffer('ar', torch.arange(M).unsqueeze(0).unsqueeze(0))
        self.embed_dim = embed_dim

    def get_sin(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.sin((2 * math.pi * self.ar * t.repeat(1, 1, self.M))/self.T_max )
    def get_cos(self, t):
        # t: Batch size x time x dim, dim = 1:
        return torch.cos((2 * math.pi * self.ar * t.repeat(1, 1, self.M))/self.T_max )
    def get_sin_cos(self, t):
        return self.get_sin(t), self.get_cos(t)
    
    def forward(self, x, sin_emb, cos_emb):
        alpha =  torch.matmul(sin_emb, self.alpha_sin) + torch.matmul(cos_emb, self.alpha_cos)
        beta  =  torch.matmul(sin_emb, self.beta_sin)  + torch.matmul(cos_emb, self.beta_cos)
        return x * alpha + beta


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, T_max):
        super().__init__()

        self.embed_dim = embed_dim
        initial_div_term = torch.exp(torch.arange(0.0, embed_dim, 2).float() * -(math.log(T_max) / embed_dim))
        self.register_buffer('w', initial_div_term)

    def get_sin(self, t, w):
        # t: Batch size x time x dim, dim = 1:\
        return torch.sin(t[:, :, None] * w)

    def get_cos(self, t, w):
        # t: Batch size x time x dim, dim = 1:
        return torch.cos(t[:, :, None] * w)

    def get_sin_cos(self, t):
        t = t.squeeze(-1)
        w = self.w.unsqueeze(0).unsqueeze(1) 
        return self.get_sin(t, w), self.get_cos(t, w)

    def forward(self, x, sin_emb, cos_emb):
        """
        t: Tensor de series de tiempo con dimensiones [batch_size, seq_len, 1]
        """       
        batch_size, seq_len, _ = x.shape
        pe = torch.empty(batch_size, seq_len, self.embed_dim, device=x.device)
        pe[:, :, 0::2] = sin_emb
        pe[:, :, 1::2] = cos_emb

        return x + pe
    

class TimeModulatorHandler(nn.Module):
    def __init__(self, input_dim, embed_dim, M, T_max, dataset_channel,
                       which_linear = nn.Linear, use_common_positional_encoding = False, **kwargs):
        super().__init__()
        self.input_dim                      = input_dim
        self.to_emb                         = which_linear(input_dim, embed_dim)
        self.dataset_channel                = dataset_channel
        self.use_common_positional_encoding = use_common_positional_encoding
        self.time_mod = []
        for i in range(dataset_channel):
          if self.use_common_positional_encoding:
            self.time_mod += [PositionalEncoding(embed_dim, T_max)]
          else:
            self.time_mod += [TimeModulator(M, embed_dim, T_max)]
        self.time_mod    = nn.ModuleList(self.time_mod)


class EncTimeModulatorHandler(TimeModulatorHandler):
    def __init__(self, cat_noise_to_E = False, **kwargs):
        super().__init__(**kwargs)
        self.cat_noise_to_E = cat_noise_to_E

    def forward(self, x, t, mask, var = None):
      # [Batch x seq len x features]
      all_mod_emb_x = []
      if self.cat_noise_to_E and var is not None:
        all_x = [torch.stack([x[:, :, i], var[:, :, i] ], -1) for i in range(self.dataset_channel)]
      else:
        all_x  = [x[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
      all_time = [t[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
      all_mask = [mask[:, :, i].unsqueeze(2) for i in range(self.dataset_channel)]
      for i in range(self.dataset_channel):
          ### Modulate input ###
          x, time, mask              = all_x[i], all_time[i], all_mask[i]
          emb_x                      = self.to_emb(x)
          time_emb_sin, time_emb_cos = self.time_mod[i].get_sin_cos(time)
          aux_emb_x                  = self.time_mod[i](emb_x, time_emb_sin, time_emb_cos)
          all_mod_emb_x             += [aux_emb_x]

      mod_emb_x, time, mask = torch.cat(all_mod_emb_x, 1), torch.cat(all_time, 1),\
                                           torch.cat(all_mask, 1)
      a_time = (time * mask + (1 - mask) * 9999999).argsort(1)
      return mod_emb_x.gather(1, a_time.repeat(1, 1, mod_emb_x.shape[-1])), mask.gather(1, a_time)

# Arguments for parser
def add_sample_parser(parser):
  ### Attention stuff ###
  parser.add_argument(
    '--M', type = int, default = 16,
    help='Number of component of fourier modulator (time)?(default: %(default)s)')
  parser.add_argument(
    '--Feat_M', type = int, default = 16,
    help='Number of component of fourier modulator (feat)?(default: %(default)s)')
  return parser

def add_name_config(config):
  name = []
  if config['which_encoder'] == 'mha' or config['which_decoder'] == 'mha':
    name += [
      'Tmod',
      'M%d' % config['M'] ]
  return name