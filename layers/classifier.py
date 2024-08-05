'''
prior layers
'''
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import functools
import math
from . import optimizers
from . import general_layers as glayers
from . import init_model as itmodel

import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__()
        # Initialization style
        self.cfg_general(**kwargs)
        self.cfg_layers(**kwargs)
        self.cfg_bn(**kwargs)
        self.cfg_optimizers(**kwargs)
        self.create_layers(**kwargs)
        self.cfg_init(**kwargs)

    def cfg_general(self, dim_z=128, n_classes=1000, dim_f = 0,  is_sharpen = 0.0,
                                            use_extra_feat = False, which_train_fn = 'AE', **kwargs):
        # Data/Latent dimension   
        self.dim_z           = dim_z
        self.n_classes       = n_classes
        self.is_sharpen      = is_sharpen
        self.use_extra_feat  = use_extra_feat
        self.dim_f           = dim_f
        self.dim_ef          = 2 if which_train_fn == 'Energy' else 0

    def cfg_init(self, E_init='ortho', skip_init=False, **kwargs):
        self.init = E_init
        # Initialize weights
        if not skip_init:
            itmodel.init_weights(self)

    def cfg_layers(self, E_param='SN', E_nl='relu',  num_neural_classifier = 0,  
                          num_E_SVs=1, num_E_SV_itrs=1, SN_eps=1e-12, E_kernel_size = 3, **kwargs):
        self.E_param, self.nl, self.kernel_size  = E_param, E_nl, E_kernel_size
        self.num_E_SVs, self.num_E_SV_itrs, self.SN_eps = num_E_SVs, num_E_SV_itrs, SN_eps
        self.num_neural_classifier = num_neural_classifier

    def cfg_bn(self, BN_eps=1e-5, norm_style='in', **kwargs):
        self.BN_eps, self.norm_style = BN_eps, norm_style

    def cfg_optimizers(self, optimizer_type='adam', E_lr=2e-4, E_B1=0.0, E_B2=0.999, 
                                                    adam_eps=1e-8, weight_decay=5e-4, **kwargs):
        self.optimizer_type = optimizer_type
        self.lr, self.B1, self.B2, self.adam_eps, self.weight_decay = E_lr, E_B1, E_B2, adam_eps, weight_decay

    def create_layers(self, **kwargs):
      self.activation = glayers.activation_dict[self.nl]
      if self.E_param == 'SN':
          self.which_linear    = functools.partial(glayers.SNLinear, num_svs = self.num_E_SVs,
                                                    num_itrs = self.num_E_SV_itrs, eps=self.SN_eps)
      else:
          self.which_linear    = nn.Linear
      self.which_bn       = functools.partial(glayers.bn_1d, eps = self.BN_eps)
      if self.num_neural_classifier:
        self.classifier_ext  = []
        self.classifier_out  = []
        for i in range(self.num_neural_classifier - 1):
          self.classifier_ext += [self.which_linear(self.dim_z + self.dim_f + self.dim_ef,
                                                          self.dim_z + self.dim_f + self.dim_ef)]
          self.classifier_ext += [self.which_bn(self.dim_z + self.dim_f + self.dim_ef)]
          self.classifier_ext += [self.activation]
        self.classifier_out += [self.which_linear(self.dim_z + self.dim_f + self.dim_ef, self.n_classes)]
        self.classifier_out += [glayers.Sharpen(self.is_sharpen)]
        self.classifier_out += [torch.nn.LogSoftmax()]

        self.classifier_ext = nn.ModuleList(self.classifier_ext)
        self.classifier_out = nn.ModuleList(self.classifier_out)

    def log_latent_classifier(self, z, feat = None, obtain_embedding = False):
      if feat is not None:
        z = torch.cat([z, feat], 1)
      for layer in self.classifier_ext:
        z = layer(z)
      z_out = z
      for layer in self.classifier_out:
        z_out = layer(z_out)
      if not obtain_embedding:
        return z_out
      else:
        return z_out, z

class ClassifierLinear(nn.Module):
    def __init__(self, **kwargs):
        super(ClassifierLinear, self).__init__()
        # Initialization style
        self.cfg_general(**kwargs)
        self.cfg_optimizers(**kwargs)
        self.cfg_layers(**kwargs)
        self.create_layers(**kwargs)
        self.cfg_init(**kwargs)
        
    def cfg_general(self, dim_z=128, n_classes=1000, is_mlp = False, **kwargs):
        # Data/Latent dimension   
        self.dim_z      = dim_z
        self.n_classes  = n_classes
        self.is_mlp     = is_mlp   

    def cfg_layers(self, E_nl='relu', **kwargs):
        self.nl = E_nl

    def create_layers(self, **kwargs):
        scale_MLP = 3
        self.linear = torch.nn.Linear(self.dim_z if not self.is_mlp else scale_MLP * self.dim_z, self.n_classes)
        self.activation = glayers.activation_dict[self.nl]
        if self.is_mlp:
            self.linear0 = torch.nn.Linear(self.dim_z, scale_MLP * self.dim_z)
            self.linear1 = torch.nn.Linear(scale_MLP * self.dim_z, scale_MLP * self.dim_z)
        self.log_softmax = torch.nn.LogSoftmax()

    def cfg_optimizers(self, optimizer_type='adam', E_lr=2e-4, E_B1=0.0, E_B2=0.999, 
                                                    adam_eps=1e-8, weight_decay=5e-4, **kwargs):
        self.lr, self.B1, self.B2, self.adam_eps, self.weight_decay = E_lr, E_B1, E_B2, adam_eps, weight_decay
        self.optimizer_type = optimizer_type

    def cfg_init(self, E_init='ortho', skip_init=False, **kwargs):
        self.init = E_init
        # Initialize weights
        if not skip_init:
            itmodel.init_weights(self)

    def reset_init(self):
        itmodel.init_weights(self)

    def forward(self, z):
        if self.is_mlp:
            z = self.activation(self.linear0(z))
            z = self.activation(self.linear1(z))
        z = self.linear(z)
        return z
    
    def log_latent_classifier(self, z):
        return self.log_softmax(self.forward(z))



