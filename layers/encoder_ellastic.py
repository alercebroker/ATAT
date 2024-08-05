''' 
Encoder model
'''
import functools
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from . import time_modulator as tmod
from . import init_model as itmodel
from . import mha
from . import optimizers

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
 
class TokenClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.norm         = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.output_layer(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, embed_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ClassifierFeedForward(nn.Module):
    def __init__(self, input_dim, embed_dim, n_classes, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes)
        )
    def forward(self, x):
        return self.net(self.norm(x))

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg_general(**kwargs)
        self.create_layers(**kwargs)

    def cfg_general(self, head_dim, num_heads, attn_layers = 1, dropout = 0.0, **kwargs):
        self.input_dim   = head_dim * num_heads
        self.attn_layers = attn_layers
        self.dropout     = dropout

    def create_layers(self,**kwargs):
        self.layers = nn.ModuleList([])
        for _ in range(self.attn_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(self.input_dim, mha.MultiheadAttentionHandler(**{**kwargs, 'input_dim': self.input_dim})),
                PreNorm(self.input_dim, FeedForward(self.input_dim, 2* self.input_dim, dropout = self.dropout))
            ]))

    def get_input_dim(self):
        return self.layers[0][0].fn.get_input_dim()

    def forward(self, x, mask, causal_mask = False):
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(**{'x': x, 'mask': mask, 'causal_mask': causal_mask}) + x
            x = ff(x) + x
        return x

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.cfg_general(**kwargs)
        self.cfg_layers(**kwargs)
        self.cfg_bn(**kwargs)
        self.cfg_optimizers(**kwargs)
        self.create_layers(**kwargs)
        self.cfg_init(**kwargs)

    def cfg_general(self, dim_z = 2, dataset_channel=3,
                          which_encoder = 'vanilla', cat_noise_to_E = False,
                          which_train_fn = 'VAE', n_classes = 10, emb_norm_cte = 0.0,
                          dropout_first_mha = 0.0, dropout_second_mha = 0.0,
                          drop_mask_second_mha = False, **kwargs):
        # Data/Latent dimension   
        self.dataset_channel      = dataset_channel
        self.dim_z                = dim_z
        self.which_encoder        = which_encoder
        self.cat_noise_to_E       = cat_noise_to_E
        self.which_train_fn       = which_train_fn
        self.input_dim            = 1 if not self.cat_noise_to_E else 2
        self.n_classes            = n_classes
        self.emb_norm_cte         = emb_norm_cte
        self.dropout_first_mha    = dropout_first_mha
        self.dropout_second_mha   = dropout_second_mha
        self.drop_mask_second_mha = drop_mask_second_mha

    def reset_some_params(self, reset_tab_transformer = False, reset_lc_transformer = False, F_max =[]):
        self.time_modulator = tmod.EncTimeModulatorHandler(**{**nkwargs, 'embed_dim': self.input_dim_mha})
        self.dim_tab        = len(F_max)
        if (self.using_metadata or self.using_features) and not self.combine_lc_tab:
            if not self.not_tabular_transformer:
                self.tab_W_feat      = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))
                self.tab_b_feat      = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))

    def cfg_init(self, E_init='ortho', skip_init=False, **kwargs):
        self.init = E_init
        # Initialize weights
        if not skip_init:
            itmodel.init_weights(self)

    def cfg_layers(self, E_nl = 'relu', num_linear = 0, **kwargs):
        self.nl  = E_nl
        self.num_linear = num_linear

    def cfg_bn(self, BN_eps=1e-5, norm_style='in', **kwargs):
        self.BN_eps, self.norm_style = BN_eps, norm_style

    def cfg_optimizers(self, optimizer_type='adam', E_lr=5e-5, E_B1=0.0, E_B2=0.999, 
                                                    adam_eps=1e-8, weight_decay=5e-4, **kwargs):
        self.lr, self.B1, self.B2, self.adam_eps, self.weight_decay = E_lr, E_B1, E_B2, adam_eps, weight_decay
        self.optimizer_type = optimizer_type
        
    def create_layers(self, using_metadata = False, using_features = False, emb_to_classifier = 'token', F_max = [],
                            tab_detach = False, tab_num_heads = 4, tab_head_dim = 32,
                            tab_output_dim = 0, combine_lc_tab = False,
                            use_detection_token = False, not_tabular_transformer = False, **kwargs):

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right nows
        self.activation              = activation_dict[self.nl]
        self.which_linear            = nn.Linear
        self.which_embedding         = nn.Embedding
        self.which_bn                = nn.BatchNorm2d
        self.using_metadata          = using_metadata
        self.using_features          = using_features
        self.emb_to_classifier       = emb_to_classifier
        self.not_tabular_transformer = not_tabular_transformer

        nkwargs = kwargs.copy()
        nkwargs.update({'input_dim': self.input_dim, 'which_linear': self.which_linear,
                            'which_bn': self.which_bn, 'which_embedding': self.which_embedding})
        # Using MultiheadAttention
        self.transformer    = Transformer(**{**nkwargs, 'dropout': self.dropout_first_mha})
        self.input_dim_mha  = self.transformer.get_input_dim()
        self.time_modulator = tmod.EncTimeModulatorHandler(**{**nkwargs, 'embed_dim': self.input_dim_mha})

        self.tab_detach          = tab_detach
        self.tab_num_heads       = tab_num_heads
        self.tab_head_dim        = tab_head_dim
        self.tab_output_dim      = tab_output_dim
        self.use_detection_token = use_detection_token

        self.combine_lc_tab      = combine_lc_tab

        self.F_len    = len(F_max)
        self.dim_tab  = len(F_max)

        if self.combine_lc_tab:
            self.tab_input_dim_mha = self.input_dim_mha
            self.tab_W_feat   = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))
            self.tab_b_feat   = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))

        if (self.using_metadata or self.using_features) and not self.combine_lc_tab:
            self.tab_transformer    = Transformer(**{**nkwargs, 'head_dim' : self.tab_head_dim,
                                                                'num_heads': self.tab_num_heads,
                                                                'dropout'  : self.dropout_first_mha})
            self.tab_input_dim_mha  = self.tab_transformer.get_input_dim()
            if not self.not_tabular_transformer:
                self.input_dim_mix   = self.tab_input_dim_mha
                self.tab_token       = nn.Parameter(torch.randn(1, 1, self.tab_input_dim_mha))
                self.tab_W_feat      = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))
                self.tab_b_feat      = nn.Parameter(torch.randn(1, self.dim_tab, self.tab_input_dim_mha))
            else:
                self.input_dim_mix   = len(F_max)              
            self.tab_classifier  = TokenClassifier(self.input_dim_mix, self.n_classes)
            self.mix_classifier  = ClassifierFeedForward(self.input_dim_mix + self.input_dim_mha,
                                                         self.input_dim_mix + self.input_dim_mha,
                                                         self.n_classes, self.dropout_second_mha)
        self.token         = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))
        self.lc_classifier = TokenClassifier(self.input_dim_mha, self.n_classes)
        self.log_softmax   = torch.nn.LogSoftmax()
        if self.use_detection_token:
            self.detection_token     = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))
            self.non_detection_token = nn.Parameter(torch.randn(1, 1, self.input_dim_mha))

    def obtain_emb_to_classify(self, emb_x, mask, time = None, **kwargs):
        if self.emb_to_classifier == 'avg':
            return (emb_x * mask).sum(1)/mask.sum(1)
        elif self.emb_to_classifier == 'token':
            return emb_x[:, 0, :]

    def obtain_argsort(self, time, mask):
        return (time * mask + (1 - mask) * 9999999).argsort(1)

    def obtain_last_emb(self, emb_x, mask, time):
        bs           = emb_x.shape[0]
        time_r       = time.permute(0,2,1).reshape(bs, -1)
        mask_r       = mask.permute(0,2,1).reshape(bs, -1)
        a_time       = self.obtain_argsort(time_r, mask_r)
        time_sorted  = time_r.gather(1, a_time)   
        mask_sorted  = mask_r.gather(1, a_time)   
        idx          = (time_sorted * mask_sorted).argmax(1)
        return emb_x[torch.arange(bs), idx, :]

    def obtain_all_lc_emb(self, data, data_var = None, time = None, mask = None,
                                     tabular_feat = None, mask_detection = False, **kwargs):
        
        emb_x, mask    = self.time_modulator(data, time, mask, var = data_var)
        if self.emb_to_classifier == 'token':
            token_repeated = self.token.repeat(emb_x.shape[0], 1, 1)
            mask_token     = torch.ones(emb_x.shape[0], 1, 1).float().to(emb_x.device) 
            mask           = torch.cat([mask_token, mask], axis = 1)
            emb_x          = torch.cat([token_repeated, emb_x], axis = 1)

        if self.combine_lc_tab:
            tab_emb     = self.tab_W_feat * tabular_feat + self.tab_b_feat 
            tab_mask    = torch.ones(tabular_feat.shape).float().to(emb_x.device) 
            emb_x       = torch.cat([tab_emb, emb_x], axis = 1)
            mask        = torch.cat([tab_mask, mask], axis = 1)

        emb_x  = self.transformer(emb_x, mask)
        return emb_x

    def obtain_lc_emb(self, obtain_all_seq_not_token = False, **kwargs):
        emb_x = self.obtain_all_lc_emb(**kwargs)
        emb_to_classify = self.obtain_emb_to_classify(emb_x, **kwargs)
        if not obtain_all_seq_not_token:
            return emb_to_classify
        else:
            return emb_x, emb_to_classify

    def obtain_all_tab_emb(self, tabular_feat = None, **kwargs):
        tab_emb               = self.tab_W_feat * tabular_feat + self.tab_b_feat
        tab_token_repeated    = self.tab_token.repeat(tab_emb.shape[0], 1, 1)
        tab_emb               = torch.cat([tab_token_repeated, tab_emb], axis = 1)
        tab_emb               = self.tab_transformer(tab_emb, None)
        return tab_emb

    def obtain_tab_emb(self, obtain_all_seq_not_token = False, **kwargs):
        output = self.obtain_all_tab_emb(**kwargs)
        if not obtain_all_seq_not_token:
            return output[:, 0,  :]
        else:
            return output, output[:, 0,  :]
    
    def obtain_raw_feat(self, tabular_feat, **kwargs):
        return tabular_feat.squeeze()

    def predict_lc(self, **kwargs):
        z_rep = self.obtain_lc_emb(**kwargs)
        return {'MLP': self.log_softmax(self.lc_classifier(z_rep))}

    def predict_tab(self, **kwargs):
        emb_x = self.obtain_tab_emb(**kwargs)
        return {'MLPTab': self.log_softmax(self.tab_classifier(emb_x))}

    def predict_mix(self, **kwargs):
        emb_y = self(**kwargs)
        return {'MLPMix': self.log_softmax(emb_y['MLPMix'])}

    def predict_all(self, **kwargs):
        emb_y = self(**kwargs) 
        return {key: self.log_softmax(emb_y[key]) for key in emb_y.keys()} 

    def combine_lc_tab_emb(self, emb_lc, emb_tab, **kwargs):
        return torch.cat([emb_tab, emb_lc], axis = 1)

    def forward(self, global_step = 0, **kwargs):
        output = {}
        # Obtain lc embedding
        emb_lc = self.obtain_lc_emb(**kwargs)
        output.update({'MLP': self.lc_classifier(emb_lc)})
        if (self.using_metadata or self.using_features) and not self.combine_lc_tab:
            # Obtain tabular embedding
            emb_tab = self.obtain_tab_emb(**kwargs) if not self.not_tabular_transformer else self.obtain_raw_feat(**kwargs)
            output.update({'MLPTab': self.tab_classifier(emb_tab)})
            # Combine both embedding and we classified them with a MLP
            emb_mix    = self.combine_lc_tab_emb(emb_lc, emb_tab, **kwargs)
            mix_output = self.mix_classifier(emb_mix)
            #output.update({'MLPMix': mix_output if global_step > 20000 else mix_output.detach()})
            output.update({'MLPMix': mix_output})
        if not ('MLPMix' in output.keys()):
            output['MLPMix']  = output['MLP']
        return output

# Arguments for parser
def add_sample_parser(parser):
  parser.add_argument(
    '--attn_layers', type=int, default = 1,
    help='Number of attentions layers'
         '(default: %(default)s)')
  parser.add_argument(
    '--emb_to_classifier', type=str, default = 'avg',   
    help='what embedding to use'
         '(default: %(default)s)')
  parser.add_argument(
    '--using_tabular_feat', action='store_true', default=False,
    help='using tabular features?'
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