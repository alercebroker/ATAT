""" 
Representation learning on variable length and irregular sampling time series with Generative models
Code by Nicolás Astorga
"""
import torch
# Import my stuff
import utils
import losses
import datasets
import obtain_metrics
#import train_fns
import train_fns
import utils
import layers.optimizers as optim
import plots.all_plots as all_plots
import plots.all_scorer as pallsr
import importlib
import pytorch_lightning as pl
import plot_manager as pm
import numpy as np

class LitModel(pl.LightningModule):
    def __init__(self, pre_model = None, **kwargs):
        super().__init__()
        self.update_config(**{'pre_model': pre_model, **kwargs})

        if pre_model is not None:
            self.config['using_pretrained'] = True

        config = self.config
        self.create_extra_metrics()
        self.plot_config()
        self.SPM = pm.ScorePlotManager(config, self.plt_cfg)
        self.my_loss = losses.Loss_obj(**config)

        self.create_models(config, pre_model)
        self.op_dict = {**self.models_dict, 'config': config, 'my_loss': self.my_loss,
                         'SPM': self.SPM}

        self.update_steps(self.op_dict)

        self.automatic_optimization = False
        self.allow_prepare_data = True
        self.test_writting = False
        self.print_parameters()

    def create_extra_metrics(self, test_set_name = ''):
        self.config['dict_set_types'] = datasets.dir_dset_dict[self.config['dataset']]
        self.extra_dict_set_types = {key : self.config['dict_set_types'][key] for key in self.config['dict_set_types'] \
                                                                if not (key in  ['train', 'val', 'val_rec', 'test'])}
        self.metrics_extra_dict_set_types = {key : self.config['dict_set_types'][key] for key in self.config['dict_set_types'] \
                                                                if not (key in  ['train', 'val', 'val_rec'])}
        if self.config['using_val']:
            self.metrics_extra_dict_set_types['val'] = self.config['dict_set_types']['val']
        if self.config['using_val_rec']:
            self.metrics_extra_dict_set_types['val_rec'] = self.config['dict_set_types']['val_rec']
        if test_set_name != '':
            self.metrics_extra_dict_set_types = {}
            self.metrics_extra_dict_set_types[test_set_name] = self.config['dict_set_types']['test']

    def create_models(self, config):
        return 

    def update_steps(self, config):
        return 
    
    def update_config(self, pre_model = None, **kwargs):
        self.config = kwargs
        if pre_model is None:
            self.save_hyperparameters() # Super IMPORTANT !!!, this allow you to write code without loading state_dict 
        else:
            self.save_hyperparameters(ignore=['pre_model']) # Super IMPORTANT !!!, this allow you to write code without loading state_dict 
        self.config = utils.update_config(kwargs)
        self.init_config = self.config.copy()

    def has_attr(self, pre_model =  None, key = ''):
        if pre_model is not None:
            return hasattr(pre_model, key)
        else:
            return False
            
    def print_parameters(self):
        for key in self.models_dict.keys():
            if self.models_dict[key] is not None:
                print('Number of params of %s: %s' % (key, sum([p.data.nelement() \
                                            for p in self.models_dict[key].parameters()] )  ))

    def plot_config(self):
        self.plt_cfg = \
        {'train':    {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False},
         'val' :     {'plt_h': False ,'plt_s': False , 'plt_r': False , 'plt_hv': False, 'plt_g': False},
         'val_rec' : {'plt_h': False , 'plt_s': False , 'plt_r': True , 'plt_hv': False, 'plt_g': False},
         'test'  :   {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False},
         'train_step':  {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False}}
        for set_type in self.extra_dict_set_types.keys():
            self.plt_cfg[set_type] = self.plt_cfg['test']

    def dict_to_device(self, all_data):
        for key, value in all_data.items():
            all_data[key] = all_data[key].to(self.device)
        return all_data
        
    def metrics_loop(self, dataloader, step_fn, to_device = False):
        list_metrics = []
        for all_data in dataloader:
          all_data = self.dict_to_device(all_data) if to_device else all_data
          list_metrics += [self.forward_step_fn(all_data, step_fn)]
        return obtain_metrics.reduce_metrics(list_metrics)

    def forward(self, x):
        return self.forward_step_fn(x, self.metrics_step)

    ### To be implemented
    def forward_step_fn(self, all_data, step_fn, **kwargs):
        return

    def training_step(self, batch, batch_idx):
        opt     = self.optimizers()
        metrics = self.forward_step_fn(batch, self.train_step)
        self.log_metrics(metrics, set_type = 'train')
        opt.zero_grad()
        self.manual_backward(metrics['loss'])
        opt.step()

    def configure_optimizers(self):
        return tuple([optim.obtain_optimizer(model) for model in self.models_dict.values() if model is not None])

    def validation_step(self, batch, batch_idx, dataloader_idx = None):
        kwargs  = self.validation_kwargs(dataloader_idx)
        metrics = self.forward_step_fn(batch, self.metrics_step, **kwargs)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx = None):
        metrics = self.forward_step_fn(batch, self.metrics_step)
        return metrics

    def validation_kwargs(self, dataloader_idx = None):
        set_type = self.idx_to_set[str(dataloader_idx)] if dataloader_idx is not None else 'test'
        return {'obtain_metrics_plot': self.plt_cfg[set_type]['plt_r']} 

    def prepare_data(self, is_reloaded = False, test_set_name = ''):
        if self.allow_prepare_data:
            self.create_extra_metrics(test_set_name)
            self.train_used = datasets.get_data(**{**self.init_config, 'set_type': 'train'})
            self.all_datasets_used = {}

            for set_type in self.metrics_extra_dict_set_types.keys():
                self.all_datasets_used[set_type] = datasets.get_data(**{**self.init_config, 'set_type': set_type}) 
                if self.config['eval_multiple_metrics'] != '' and is_reloaded:
                    all_metrics_to_eval = datasets.obtain_all_metrics(self.config['eval_multiple_metrics'])
                    for eval_metric in all_metrics_to_eval:
                        this_set_type = '%s_%s_%s' % (set_type, self.config['eval_multiple_metrics'], eval_metric)
                        self.all_datasets_used[this_set_type] = datasets.get_data(**{**self.init_config,
                                                                    'set_type': this_set_type,
                                                                    'eval_metric': eval_metric,
                                                                    'list_eval_metrics': all_metrics_to_eval})

            self.fixed_prepare_data()

    def reset_prepare_data(self):
        self.allow_prepare_data = True

    def fixed_prepare_data(self):
        self.allow_prepare_data = False

    def train_dataloader(self):
        self.train_used = datasets.get_data(**{**self.init_config, 'set_type': 'train'})
        return datasets.get_data_loaders(**{**self.init_config, 'dataset_used': self.train_used, 'set_type': 'train', 'drop_last': True})

    def val_dataloader(self, is_reloaded = False):
        loader_list = []
        self.loader_index, self.stack = {}, 0
        if self.config['using_train_step']:
            loader_list  += [datasets.get_data_loaders(**{**self.init_config, 'dataset_used': self.train_used,
                                                     'set_type': 'train', 'drop_last': False, 'is_train_step': True})]
            self.loader_index['train_step'], self.stack = self.stack, self.stack + 1

        for set_type in self.all_datasets_used.keys():
            this_dataset  = self.all_datasets_used[set_type]
            loader_list  += [datasets.get_data_loaders(**{**self.init_config, 'dataset_used': this_dataset,
                                                    'set_type': set_type, 'drop_last': False})]
            self.loader_index[set_type], self.stack = self.stack, self.stack + 1

        self.idx_to_set = {str(v): k for k, v in self.loader_index.items()}
        return loader_list

    def test_dataloader(self):
        self.test_used  = datasets.get_data(**{**self.init_config, 'set_type': 'test'})
        return datasets.get_data_loaders(**{**self.init_config, 'dataset_used': self.test_used,
                                                      'set_type': 'test', 'drop_last': False})

    def log_simple(self, metrics, set_type):
        for key in metrics.keys():
            self.log('%s_%s' % (set_type, metrics[key]), metrics[key])

    def log_metrics(self, metrics, set_type = 'train', kwargs = {}):
        for key in metrics.keys():
            self.log('%s/%s' % (key, set_type), metrics[key], **kwargs)
            self.log('%s/%s' % ('all_metrics', set_type), metrics, **kwargs)
        if set_type == 'train':
            for key in metrics.keys():
                self.log('%s/%s' % (key, 'mixed'), {set_type: metrics[key]})

    def log_mixed_metrics(self, m_metrics):
        this_set = list(m_metrics.keys())[-1]
        for key in m_metrics[this_set].keys():
            aux_dict = {set_type: m_metrics[set_type][key] for set_type in m_metrics.keys() if key in m_metrics[set_type].keys()}
            self.log('%s/%s' % (key, 'mixed'), aux_dict)

    def log_and_plot(self, metrics_stat, metrics_ext, set_type = 'val'):
        utils.obtain_general_root(self.config, self.global_step, set_type = set_type)
        self.SPM.oneset_dist_plot(metrics_stat, metrics_ext, set_type = set_type)
        self.log_metrics(metrics_stat, set_type = set_type)

    def log_and_plot_reloaded(self, metrics_stat, metrics_ext, set_type):
        pallsr.plot_reloaded_scores(metrics_stat, metrics_ext, self.config, set_type)
        self.log_simple(metrics_stat, set_type)

    def obtain_set_list(self, metrics_list, set_type = 'test'):
        this_list = metrics_list[self.loader_index[set_type]] \
                    if set_type is not None and self.stack > 1 else metrics_list[:]
        return this_list

    def obtain_post_metrics(self, metrics_list, set_type = 'test'):
        metrics_ext   = obtain_metrics.reduce_metrics(self.obtain_set_list(metrics_list, set_type))
        return self.SPM.post_metrics(metrics_ext, set_type)

    def metrics_log_and_plot(self, metrics_list, set_type = 'test', is_reloaded = False):
        metrics_stat, metrics_ext = self.obtain_post_metrics(metrics_list, set_type = set_type)
        if not is_reloaded:
            self.log_and_plot(metrics_stat, metrics_ext, set_type)
        else:
            self.log_and_plot_reloaded(metrics_stat, metrics_ext, set_type)
        return metrics_stat, metrics_ext

    def metrics_plot_across_sets(self, m_metrics_stat, is_reloaded = False, set_type = 'test'):
        if is_reloaded:
            if self.config['eval_multiple_metrics']:
                pallsr.plot_multiple_set_type(m_metrics_stat, self.config,
                                                is_reloaded = is_reloaded,
                                                metric_name = self.config['eval_multiple_metrics'],
                                                set_type    = set_type)

    def validation_epoch_end(self, val_step_outputs, is_reloaded = False):
        m_metrics_stat = {}
        
        if self.config['using_train_step']:
            train_metrics_stat, _ = self.metrics_log_and_plot(val_step_outputs, 'train_step',
                                                                 is_reloaded = is_reloaded)
            m_metrics_stat['train_step'] = train_metrics_stat

        for set_type in self.all_datasets_used.keys():
            aux_metrics_stat, _ = self.metrics_log_and_plot(val_step_outputs, set_type = set_type,
                                                                is_reloaded = is_reloaded)
            m_metrics_stat[set_type] = aux_metrics_stat.copy()

        for set_type in self.metrics_extra_dict_set_types.keys():
            self.metrics_plot_across_sets(m_metrics_stat, is_reloaded = is_reloaded, set_type = set_type)
        if not is_reloaded:
            self.log_mixed_metrics(m_metrics_stat)

    def test_epoch_end(self, test_step_outputs):
        self.validation_epoch_end(test_step_outputs, is_reloaded = True)


class ClassifierModel(LitModel):
    def __init__(self, pre_model = None, **kwargs):
        super().__init__(pre_model = pre_model, **kwargs)

    def create_models(self, config, pre_model = None):
        encoder = importlib.import_module(config['encoder'])
        self.E = encoder.Encoder(**config) if not self.has_attr(pre_model, 'E') else pre_model.E
        if pre_model:
            self.E.reset_some_params(**config)
        self.models_dict = {'E': self.E}

    def update_steps(self, config):
        self.train_step   = \
            getattr(importlib.import_module('train_fns'), 'classifier_training_function_SSL')(**config)
        self.metrics_step = \
            getattr(importlib.import_module('obtain_metrics'), 'classifier_metrics_fns_SSL')(**config)

    def forward_step_fn(self, all_data, step_fn, **kwargs):
        these_kwargs = {}
        data_var  = None
        data, time, labels, mask, mask_detection = all_data['data'].float(), \
                                                all_data['time'].float(), all_data['labels'].long(), \
                                                all_data['mask'].float(), all_data['mask_detection'].float()
        
        these_kwargs = {'data': data, 'time': time, 'labels': labels,
                        'mask': mask, 'mask_detection': mask_detection}
        
        these_kwargs.update({'global_step': self.global_step})

        if 'data_var' in all_data.keys():
            these_kwargs['data_var'] = all_data['data_var'].float()

        if 'tabular_feat' in all_data.keys():
            these_kwargs['tabular_feat'] = all_data['tabular_feat'].float().unsqueeze(2)
            if 'add_tabular_feat' in all_data.keys():
                these_kwargs['tabular_feat'] = torch.cat([these_kwargs['tabular_feat'],
                                                all_data['add_tabular_feat'].float().unsqueeze(2)], 1)
        elif 'add_tabular_feat' in all_data.keys():
            these_kwargs['tabular_feat'] = all_data['add_tabular_feat'].float().unsqueeze(2)

        these_kwargs.update(kwargs)
        metrics = step_fn(**these_kwargs)
        return metrics

    def order_by_time(self, this_dict):
        high_value = 999999
        for ss in ['', '_for']:
            if 'time%s'%ss in this_dict.keys():
                aux = (1 - this_dict['mask%s'%ss]) * high_value \
                                    +  this_dict['time%s'%ss] * this_dict['mask%s'%ss]
                atime = aux.argsort(1)
                for key in this_dict.keys():
                    if key in ['data%s'%ss, 'mask%s'%ss, 'data_var%s'%ss, 'time%s'%ss]:
                        this_dict[key] = this_dict[key].gather(1, atime)

    def plot_config(self):
        self.plt_cfg = \
        {'train':    {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False},
         'val' :     {'plt_h': False ,'plt_s': False , 'plt_r': False , 'plt_hv': False, 'plt_g': False},
         'val_rec' : {'plt_h': False , 'plt_s': False , 'plt_r': False , 'plt_hv': False, 'plt_g': False},
         'test'  :   {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False},
         'train_step':  {'plt_h': False, 'plt_s': False, 'plt_r': False, 'plt_hv': False, 'plt_g': False}}
        for set_type in self.extra_dict_set_types.keys():
            self.plt_cfg[set_type] = self.plt_cfg['test']