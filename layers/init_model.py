''' 
Init
'''
import torch
import torch.nn as nn
from torch.nn import init

# Initialize
def init_weights(this_model, model_name = ''):
    this_model.param_count = 0
    for module in this_model.modules():
        if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
          if this_model.init == 'ortho':
            init.orthogonal_(module.weight)
          elif this_model.init == 'N02':
            init.normal_(module.weight, 0, 0.02)
          elif this_model.init in ['glorot', 'xavier']:
            init.xavier_uniform_(module.weight)
          else:
            print('Init style not recognized...')
          this_model.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for %s''s initialized parameters: %d' % (model_name, this_model.param_count))