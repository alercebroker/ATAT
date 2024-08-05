import numpy as np
import os
from filelock import FileLock
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest import Repeater
from argparse import ArgumentParser
import parser
import pickle
import json
import main_model
import plots.all_scorer as pallsr
import plots.plot_scorer as psr
import utils
import pandas as pd
import importlib
from pytorch_lightning import Trainer, seed_everything

ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = float(300)

#exp_setting['eval_loss']         = 'error_rec_mean1/train_step'#'error_rec/val'
ABS_PATH          = os.path.abspath('.')
CHECKPOINT_NAME   = 'my_best_checkpoint'
RESULT_DIR        = 'results'

def make_dir(root, key): 
    aux_path      = '%s/%s' % (root, key)
    if not os.path.exists(aux_path):
        os.mkdir(aux_path)

def best_model_dir_and_step(some_dir):
    list_dir        = os.listdir(some_dir)
    index_model     = [i for i, this_dir in enumerate(list_dir) if CHECKPOINT_NAME in this_dir]
    best_model_name = list_dir[index_model[0]]
    return '%s/%s' % (some_dir, best_model_name), ''.join(x for x in best_model_name if x.isdigit())

def trial_name_string(trial): 
    name_str = 'Exp_cfg_' 
    seed_str = 'seed' 
    for key in trial.config.keys(): 
        if key != seed_str: 
            if type(trial.config[key]) == str: 
                name_str = '%s-%s=%s' % (name_str, key, trial.config[key]) 
            else: 
                name_str = '%s/%s=%3.2e' % (name_str, key, trial.config[key]) 
    if seed_str in trial.config.keys(): 
        name_str = '%s-%s=%d' % (name_str, 'seed', trial.config['seed']) 
    return name_str

def update_input_str(arch_spec, config):
    aux_str  = ''
    for key in config.keys():
        aux_str += arch_spec['dict_tune'][key][config[key]] if key != 'seed' else ' --seed %s' % config[key]
    return aux_str

def training(config, exp_setting = None, arch_gen = None, arch_spec = None, load_pretrain = ''):
    if 'seed' in config.keys():
        seed_everything(config['seed'], workers=True)
    updated_input              = exp_setting['default_string']
    updated_input             += arch_gen
    updated_input             += update_input_str(arch_spec, config)
    config_used                = vars(parser.prepare_parser(abs_path = ABS_PATH).parse_args(updated_input.split() ))
    path_logger                = tune.get_trial_dir() #"logs", #tune.get_trial_dir()
    config_used['abs_path']    = ABS_PATH
    config_used['current_dir'] = path_logger 
    model_module = getattr(importlib.import_module('main_model'), config_used['pl_model'])
    if load_pretrain == '':
        model = model_module(**config_used)
    else:
        pre_model_module = getattr(importlib.import_module('main_model'), config_used['pl_pre_model'])
        checkpoint_model_path, _ = best_model_dir_and_step('%s/%s/%s' % (ABS_PATH, load_pretrain, path_logger.split('/', -1)[-2]))
        pre_model = pre_model_module.load_from_checkpoint(checkpoint_model_path)
        model = model_module(**{'pre_model': pre_model, **config_used})
        #model.update_config(**config_used)

    mode_used    = exp_setting['mode'] if 'mode' in exp_setting.keys() else 'min'
    all_callbacks = []
    all_callbacks += [ModelCheckpoint(monitor = exp_setting['eval_loss'], dirpath='',
                                            save_top_k = 1, mode = mode_used,#)]
                                            filename='my_best_checkpoint-{step}')]
    all_callbacks += [EarlyStopping(monitor = exp_setting['eval_loss'], min_delta=0.00,
                                       patience = 4, verbose=False, mode= mode_used)]
    all_callbacks += [TuneReportCheckpointCallback({exp_setting['eval_loss']: exp_setting['eval_loss']},
                                                        filename = "my_check" ,  on="validation_end")]
   
    for callback in config_used['callbacks']:
        all_callbacks += [getattr(importlib.import_module('callbacks'), callback)(**config)]


    all_loggers  = []
    all_loggers += [pl_loggers.TensorBoardLogger(save_dir = path_logger,
                                        name="tensorboard", version=".")]
    all_loggers += [pl_loggers.CSVLogger(save_dir = path_logger,
                                        name=".", version=".")]

    trainer = Trainer(callbacks = all_callbacks, logger = all_loggers,
                    val_check_interval = 20000, #check_val_every_n_epoch = int(config_used['check_every_n_epochs']),
                    log_every_n_steps= 100, #val_check_interval = config_used['check_every_n_epochs'],
                    gpus=1, min_epochs = config_used['min_epochs'],
                    max_epochs = config_used['max_epochs'], num_sanity_val_steps = 0)
    trainer.running_sanity_check = False
    #trainer = Trainer(gpus=1, min_epochs = 3, max_epochs = 30, num_sanity_val_steps = 0)

    trainer.fit(model)

def run(experiment_name, exp_setting, arch_gen, arch_spec, search_setting, load_pretrain = ''):
    config = {key: tune.choice(arch_spec['tune_choice'][key]) for key in arch_spec['tune_choice'].keys()}
    config.update({'seed':  tune.grid_search(search_setting['grid_search_cfg'])})
    train_fn_with_parameters = tune.with_parameters(training,
                                                    exp_setting = exp_setting,
                                                    arch_gen = arch_gen,
                                                    arch_spec = arch_spec,
                                                    load_pretrain = load_pretrain)

    resources_per_trial = {"cpu": 4, "gpu": .25}
    num_samples  = 1
    search_alg   = BasicVariantGenerator(points_to_evaluate = search_setting['search_config'])

    scheduler    = None
    reporter     = None
    mode_used    = 'max' if 'mode' in exp_setting.keys() and  exp_setting['mode'] != 'min' else 'min'

    analysis = tune.run(train_fn_with_parameters, #training,
        resources_per_trial = resources_per_trial,
        metric      = "%s" % exp_setting['eval_loss'],
        mode        = mode_used,
        config      = config,
        num_samples = num_samples,
        scheduler   = scheduler,
        local_dir   = './%s' % RESULT_DIR,
        name        = experiment_name,
        resume      = "AUTO",
        search_alg  = search_alg,
        keep_checkpoints_num = 1,
        checkpoint_freq=0,
        trial_name_creator = trial_name_string,
        trial_dirname_creator = trial_name_string,
        checkpoint_score_attr= \
            ("%s%s" % (mode_used, exp_setting['eval_loss'])).replace('max', '').replace('min', 'min-'), #should add min- if looking the minimal score
        progress_reporter=reporter,
        max_failures=3)
    return analysis


def main():
  parser_aux = ArgumentParser()
  ### Dataset/Dataloader stuff ###
  parser_aux.add_argument('--exp_setting', type=str, default='exp_setting_1', help='')
  parser_aux.add_argument('--arch_gen', type=str, default='arch_gen1', help='')
  parser_aux.add_argument('--arch_spec', type=str, default='arch_spec1', help='')
  parser_aux.add_argument('--search_setting', type=str, default='search_dec', help='')
  parser_aux.add_argument('--selec_col', nargs='+', type=int)
  parser_aux.add_argument('--load_pretrain', type=str, default='', help='')

  tune_config = vars(parser_aux.parse_args())

  with open('scripts/exp_setting/%s.json' % tune_config['exp_setting'], 'r') as fread:
    exp_setting    = json.load(fread)
  with open('scripts/arch_gen/%s.json' % tune_config['arch_gen'], 'r') as fread:
    arch_gen       = json.load(fread)
  with open('scripts/arch_spec/%s.json' % tune_config['arch_spec'], 'r') as fread:
    arch_spec      = json.load(fread)  
  with open('scripts/search_setting/%s.json' % tune_config['search_setting'], 'r') as fread:
    search_setting = json.load(fread)  

  experiment_name = '%s_%s_%s_%s' % (tune_config['exp_setting'],
                                     tune_config['arch_gen'],
                                     tune_config['arch_spec'],
                                     tune_config['search_setting'])
  if tune_config['load_pretrain'] != '':
    experiment_name += '_LOAD_%s' % tune_config['load_pretrain'].split('/', 1)[-1]

  # Obtained updated config to set some general configurations
  default_input            = exp_setting['default_string']
  config_used              = vars(parser.prepare_parser(abs_path = ABS_PATH).parse_args(default_input.split() ))
  config_used['abs_path']  = ABS_PATH
  config_used              = utils.update_config(config_used)
  classes_names            = config_used['classes_names']
  list_set_type            = []
  for set_type in list(config_used['dict_set_types'].keys()):
        if not (set_type in ['val', 'train']):
            list_set_type += [set_type]
        elif config_used['using_val'] and set_type == 'val':
            list_set_type += [set_type]
        elif config_used['using_train_step'] and set_type == 'train':
            list_set_type += [set_type.replace('train', 'train_step')]


  analysis = run(experiment_name, exp_setting, arch_gen, arch_spec, search_setting,
                                                    load_pretrain = tune_config['load_pretrain'])


if __name__ == '__main__':
  main()
