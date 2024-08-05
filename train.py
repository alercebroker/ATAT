import numpy as np
import sys
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
import parser_1 as parser
import pickle
import json
import main_model
import plots.all_scorer as pallsr
import plots.plot_scorer as psr
import utils
import pandas as pd
import importlib
import datasets
from pytorch_lightning import Trainer, seed_everything

import warnings
warnings.filterwarnings("ignore")

ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = float(300)

ABS_PATH          = os.path.abspath('.')
CHECKPOINT_NAME   = 'my_best_checkpoint'
RESULT_DIR        = 'results'

def make_dir(root, key): 
    aux_path      = '%s/%s' % (root, key)
    if not os.path.exists(aux_path):
        os.mkdir(aux_path)
    return aux_path

def obtain_config(parser_aux):
    parsed, unknown    = parser_aux.parse_known_args()
    tune_config        = vars(parsed)
    valid_arguments    = [word.replace('--','') for word in unknown if '--' in word]
    config_for_update  = vars(parser.prepare_parser(abs_path = ABS_PATH).parse_args(unknown))
    config_for_update  = {key: config_for_update[key] for key in config_for_update.keys() if key in valid_arguments}
    return tune_config, config_for_update

def best_model_dir_and_step(some_dir):
    list_dir        = os.listdir(some_dir)
    index_model     = [i for i, this_dir in enumerate(list_dir) if CHECKPOINT_NAME in this_dir]
    best_model_name = list_dir[index_model[0]]
    return '%s/%s' % (some_dir, best_model_name), ''.join(x for x in best_model_name if x.isdigit())

def obtain_str_trial(config):
    name_str = 'Exp_cfg_' 
    seed_str = 'seed' 
    for key in config.keys(): 
        if key != seed_str: 
            if type(config[key]) == str: 
                name_str = '%s-%s=%s' % (name_str, key, config[key]) 
            else: 
                name_str = '%s/%s=%3.2e' % (name_str, key, config[key]) 
    if seed_str in config.keys(): 
        name_str = '%s-%s=%d' % (name_str, 'seed', config['seed']) 
    return name_str

def trial_name_string(trial):
    return obtain_str_trial(trial.config) 

def update_input_str(searching_spec, config):
    aux_str  = ''
    for key in config.keys():
        aux_str += searching_spec['dict_tune'][key][config[key]] if key != 'seed' else ' --seed %s' % config[key]
    return aux_str

def training(config, general_setting = None, searching_spec = None, config_for_update = None, load_pretrain = '',
                eval_again = '', is_testing = False, no_ray = False, experiment_name = None):
    if 'seed' in config.keys():
        seed_everything(config['seed'], workers=True)

    updated_input              = general_setting['default_string']
    updated_input             += update_input_str(searching_spec, config)
    config_used                = vars(parser.prepare_parser(abs_path = ABS_PATH).parse_args(updated_input.split() ))
    if not no_ray:
        path_logger = tune.get_trial_dir() 
    else:
        aux_path    = obtain_str_trial(config)       
        dir_results = make_dir(ABS_PATH, 'results')
        dir_exp     = make_dir(dir_results, experiment_name)
        path_logger = make_dir(dir_exp, aux_path)

    config_used['abs_path']    = ABS_PATH
    config_used['current_dir'] = path_logger

    config_used.update(config_for_update)
    if is_testing:
        config_used['use_small_subset'] = True
    model_module = getattr(importlib.import_module('main_model'), config_used['pl_model'])
    
    if load_pretrain == '':
        model = model_module(**config_used)
    else:
        pre_model_module = getattr(importlib.import_module('main_model'), config_used['pl_pre_model'])
        checkpoint_model_path, _ = best_model_dir_and_step('%s/%s/%s' % (ABS_PATH, load_pretrain, path_logger.split('/', -1)[-2]))
        pre_model = pre_model_module.load_from_checkpoint(checkpoint_model_path)
        model = model_module(**{'pre_model': pre_model, **config_used})

    mode_used    = general_setting['mode'] if 'mode' in general_setting.keys() else 'min'
    all_callbacks = []
    path_model_checkpoint = '' if not no_ray else path_logger
    if not is_testing:
        save_top_k = 1
        every_n_train_steps = None
    else: 
        save_top_k = -1
        every_n_train_steps = 0

    all_callbacks += [ModelCheckpoint(monitor=general_setting['eval_loss'], dirpath=path_model_checkpoint,
                                      save_top_k=save_top_k, mode=mode_used, every_n_train_steps=every_n_train_steps,#)]
                                      filename='my_best_checkpoint-{step}')]
        
    all_callbacks += [EarlyStopping(monitor = general_setting['eval_loss'], min_delta=0.00,
                                    patience = 3, verbose=False, mode=mode_used)]

    if not no_ray:
        all_callbacks += [TuneReportCheckpointCallback({general_setting['eval_loss']: general_setting['eval_loss']},
                                                        filename = "my_check" ,  on="validation_end")]
   
    for callback in config_used['callbacks']:
        all_callbacks += [getattr(importlib.import_module('callbacks'), callback)(**config)]

    all_loggers  = []
    all_loggers += [pl_loggers.TensorBoardLogger(save_dir = path_logger,
                                        name="tensorboard", version=".")]
    all_loggers += [pl_loggers.CSVLogger(save_dir = path_logger,
                                        name=".", version=".")]

    if not is_testing:
        trainer = Trainer(callbacks = all_callbacks, logger = all_loggers,
                        val_check_interval = 20000, #check_val_every_n_epoch = 1, #val_check_interval = 20000, #check_val_every_n_epoch = int(config_used['check_every_n_epochs']),
                        log_every_n_steps= 100, #val_check_interval = config_used['check_every_n_epochs'],
                        gpus=1,
                        min_epochs = config_used['min_epochs'],
                        max_epochs = config_used['max_epochs'],
                        num_sanity_val_steps = 0)
    else:
        trainer = Trainer(callbacks = all_callbacks, logger = all_loggers,
                        check_val_every_n_epoch = 1,
                        log_every_n_steps= 10, #val_check_interval = config_used['check_every_n_epochs'],
                        gpus=1,
                        min_epochs = 0,
                        max_epochs = 2,
                        num_sanity_val_steps = 0)
    trainer.running_sanity_check = False

    trainer.fit(model)
    del model
    del trainer
    this_logdir = path_logger
    best_model_dir_path, step_number = best_model_dir_and_step(this_logdir)
    print("reloading models to obtain metrics")
    dir_cfg = utils.obtain_final_reloaded_root({'current_dir': this_logdir})
    trainer  = Trainer(gpus=1,  num_sanity_val_steps = 0)
    trainer.running_sanity_check = False
    pre_model_module = getattr(importlib.import_module('main_model'), config_used['pl_model'])
    print("model checkpoint ", best_model_dir_path)
    pre_model = pre_model_module.load_from_checkpoint(best_model_dir_path)
    pre_model.config.update(dir_cfg)
    pre_model.prepare_data()
    all_dataloaders = pre_model.val_dataloader(is_reloaded = True)
    trainer.test(model = pre_model, dataloaders = all_dataloaders)

def run(experiment_name, general_setting, searching_spec, search_setting, config_for_update,
         load_pretrain = '', eval_again = '', is_testing = False, no_ray = False):    
    
    config = {key: [i[key] for i in search_setting['search_config']] for key in search_setting['search_config'][0]}
    config.update({'seed':  tune.grid_search(search_setting['grid_search_cfg'])})
   
    if not no_ray:
        train_fn_with_parameters = tune.with_parameters(training,
                                                        general_setting = general_setting,
                                                        searching_spec = searching_spec,
                                                        config_for_update = config_for_update,
                                                        load_pretrain = load_pretrain,
                                                        eval_again = eval_again,
                                                        is_testing = is_testing)
    else:
        training(search_setting['search_config'][0], general_setting = general_setting,
                                                    searching_spec = searching_spec, config_for_update = config_for_update,
                                                    load_pretrain = load_pretrain, eval_again = eval_again,
                                                    is_testing = is_testing, no_ray = no_ray, experiment_name = experiment_name)
        return


    resources_per_trial = {"cpu": 4, "gpu": 1}
    num_samples  = 1
    search_alg   = BasicVariantGenerator(points_to_evaluate = search_setting['search_config'])

    scheduler    = None
    reporter     = None
    mode_used    = 'max' if 'mode' in general_setting.keys() and  general_setting['mode'] != 'min' else 'min'

    analysis = tune.run(train_fn_with_parameters, #training,
        resources_per_trial = resources_per_trial,
        metric      = "%s" % general_setting['eval_loss'],
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
            ("%s%s" % (mode_used, general_setting['eval_loss'])).replace('max', '').replace('min', 'min-'), #should add min- if looking the minimal score
        progress_reporter=reporter,
        max_failures=10)
    
    return analysis

def exp_summary(experiment_name, classes_names, list_set_type, list_set_type_non_metrics, analysis, search_setting,
                selec_col = None, eval_again = '', eval_multiple_metrics = '',
                model_name = '', test_set_name = '', early_return = False):

    print("Summarizing data")
    experiment_root = '%s/%s/%s' % (ABS_PATH, RESULT_DIR, experiment_name)
    def add_string_key(this_dict):
        return {f'config/{k}': v for k, v in this_dict.items()}
    def filter_pd(df1, filter_v):
        return df1.loc[(df1[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
    def drop_str(this_str, str_dropped = 'seed'):
        return this_str[ : this_str.find(str_dropped) - 1 ]

    summary_root = utils.make_root(experiment_root, 'summary')
    metrics_dict = {key: [] for key in list_set_type}

    for this_config  in search_setting['search_config']:

        for seed in search_setting['grid_search_cfg']:
            this_cfg           = {**this_config, 'seed': seed}
            pd_filtered        = filter_pd(analysis.dataframe(), add_string_key(this_cfg))
            this_logdir        = pd_filtered['logdir'].item()
            glogdir            = drop_str(this_logdir).replace(experiment_root, '')
            this_cfg['logdir'] = utils.make_root(summary_root, glogdir)
            best_model_dir_path, step_number = best_model_dir_and_step(this_logdir)
            dir_cfg = utils.obtain_final_reloaded_root({'current_dir': this_logdir})

            if eval_again != '':
                print("reloading models to obtain metrics")
                print("Trainer")
                trainer  = Trainer(gpus=1,  num_sanity_val_steps = 0)
                trainer.running_sanity_check = False
                print("eval model: ", eval_again)
                pre_model_module = getattr(importlib.import_module('main_model'), model_name)
                print("model checkpoint ", best_model_dir_path)
                pre_model = pre_model_module.load_from_checkpoint(best_model_dir_path)
                pre_model.config.update(dir_cfg)
                pre_model.config.update({'eval_multiple_metrics': eval_multiple_metrics})
                print("Preparing data")
                pre_model.prepare_data(is_reloaded = True, test_set_name = test_set_name)
                print("reset prepare adata")
                pre_model.reset_prepare_data()
                print("Creating all dataloaders")
                all_dataloaders = pre_model.val_dataloader(is_reloaded = True)
                print("testing model")
                trainer.test(model = pre_model, dataloaders = all_dataloaders)

            for set_type in list_set_type:
                this_cfg_aux  = this_cfg.copy()
                this_cfg_aux.update(utils.load_json(dir_cfg['reloaded_root'], '%s_results%s' % (set_type, '')))
                this_cfg_aux.update(utils.load_pickle(dir_cfg['reloaded_root'], '%s_results%s' % (set_type, '_vec')))
                metrics_dict[set_type] += [this_cfg_aux]

    pd_metrics_dict = {key: pd.DataFrame(metrics_dict[key]) for key in list_set_type}
    pd_mean = {key: pd_metrics_dict[key].groupby(by = search_setting['groupby']).mean().drop(columns = ['seed'])\
                 for key in list_set_type}
    pd_std  = {key: pd_metrics_dict[key].groupby(by = search_setting['groupby']).std().drop(columns = ['seed'])\
                 for key in list_set_type}

    if early_return:
        return pd_mean, pd_std

    for set_type in list_set_type:
        psr.print_latex_table(pd_mean[set_type], search_setting, summary_root,
                            '%s_scorer_table' % set_type, pd_grouped_std = pd_std[set_type])
        if selec_col is not None:
            cols_used = pd_mean[set_type].columns[selec_col]
            psr.print_latex_table(pd_mean[set_type][cols_used], search_setting, summary_root,
                                '%s_scorer_table_short' % set_type,
                                pd_grouped_std = pd_std[set_type][cols_used])

    for set_type in list_set_type_non_metrics:
        if eval_multiple_metrics != '' and set_type != 'train' and set_type != 'train_step':
            psr.summarize_eval_metric_many(pd_mean, summary_root, is_multiple_models = True,
                                                        set_type = set_type, metric_name = eval_multiple_metrics)

    for set_type in list_set_type:
        for this_config  in search_setting['search_config']:
            pd_filtered = filter_pd(pd_metrics_dict[set_type], this_config)
            metrics_mean  = {}
            metrics_std   = {}
            for key in pallsr.VEC_KEYS:
                for metric_name in pd_filtered.columns:
                    if key in metric_name:
                        metrics_mean[metric_name] = np.stack(pd_filtered[metric_name].to_list(), 0).mean(0)
                        metrics_std[metric_name]  = np.stack(pd_filtered[metric_name].to_list(), 0).std(0)
            config_aux = this_config.copy()
            config_aux['scorer_root'] = pd_filtered['logdir'].iloc[0]
            config_aux['classes_names'] = classes_names
            pallsr.plot_vec_metrics(metrics_mean, set_type, config_aux, metrics_std)
    return pd_mean, pd_std


def prepare_parser_ray():
    parser_aux = ArgumentParser()
    ### Dataset/Dataloader stuff ###
    parser_aux.add_argument('--general', type=str, default='general_setting_1', help='')
    parser_aux.add_argument('--general_e', type=str, default='general_setting_1', help='')
    parser_aux.add_argument('--searching', type=str, default='search_dec', help='')
    parser_aux.add_argument('--searching_e', type=str, default='searching_spec1', help='')
    parser_aux.add_argument('--name_exp', type=str, default='name exp', help='')
    parser_aux.add_argument('--selec_col', nargs='+', type=int)
    parser_aux.add_argument('--load_pretrain', type=str, default='', help='')
    parser_aux.add_argument('--model_name', type=str, default='', help='')
    parser_aux.add_argument('--test_set_name', type=str, default='', help='')
    parser_aux.add_argument('--eval_again', type=str, default='', help='')
    parser_aux.add_argument('--eval_multiple_metrics', type=str, default='', help='')
    parser_aux.add_argument('--is_testing', action='store_true', default=False, help='',)
    parser_aux.add_argument('--no_ray', action='store_true', default=False, help='',)
    return parser_aux


def analysis_and_summary(tune_config, config_for_update, early_return = False, add_name_rroot = ''):
    with open('scripts/general/%s.json' % tune_config['general'], 'r') as fread:
        general_setting    = json.load(fread)
    with open('scripts/searching/%s.json' % tune_config['searching'], 'r') as fread:
        search_setting = json.load(fread)  
    with open('scripts/searching_e/%s.json' % tune_config['searching_e'], 'r') as fread:
        searching_spec      = json.load(fread)  

    setting_spec = general_setting[tune_config['general_e']]
    general_setting['default_string'] += setting_spec['details']
    general_setting.update(setting_spec)

    experiment_name = tune_config['name_exp']

    if tune_config['load_pretrain'] != '':
        experiment_name += '_LOAD_%s' % tune_config['load_pretrain'].split('/', 1)[-1]

    # Obtained updated config to set some general configurations
    default_input            = general_setting['default_string']
    config_used              = vars(parser.prepare_parser(abs_path = ABS_PATH).parse_args(default_input.split() ))
    config_used.update(config_for_update)
    config_used['abs_path']  = ABS_PATH
    config_used              = utils.update_config(config_used)
    classes_names            = config_used['classes_names']
    list_set_type            = []

    for set_type in list(config_used['dict_set_types'].keys()):
        if not (set_type in ['val', 'train']):
            list_set_type          += [set_type]
        elif config_used['using_val'] and set_type == 'val':
            list_set_type         += [set_type]
        elif config_used['using_train_step'] and set_type == 'train':
            list_set_type += [set_type.replace('train', 'train_step')]

    if tune_config['test_set_name'] != '':
        list_set_type = [tune_config['test_set_name']]

    list_set_type_non_metrics = list_set_type.copy()
    if config_used['eval_multiple_metrics'] != '':
        all_metrics_to_eval = datasets.obtain_all_metrics(config_used['eval_multiple_metrics'])
        for set_type in list_set_type_non_metrics:
            if set_type != 'train' and set_type != 'train_step':
                for metric_value in all_metrics_to_eval:
                    list_set_type += ['%s_%s_%s' % (set_type, config_used['eval_multiple_metrics'], metric_value)]

    analysis = run(experiment_name, general_setting, searching_spec, search_setting, config_for_update,
                                            load_pretrain = tune_config['load_pretrain'],
                                            eval_again = tune_config['eval_again'],
                                            is_testing = tune_config['is_testing'],
                                            no_ray = tune_config['no_ray'])

    try:
        if not tune_config['no_ray']:
            eval_multiple_metrics = tune_config['eval_multiple_metrics'] if tune_config['eval_multiple_metrics']  != '' else \
                                    config_used['eval_multiple_metrics']

            return exp_summary(experiment_name, classes_names, list_set_type, list_set_type_non_metrics, analysis, search_setting,
                        eval_again = tune_config['eval_again'],
                        selec_col = tune_config['selec_col'],
                        eval_multiple_metrics = eval_multiple_metrics,
                        model_name   = tune_config['model_name'],
                        test_set_name = tune_config['test_set_name'],
                        early_return = early_return)

    except FileNotFoundError:
        print('Metrics must be obtained manually using the get_metrics.py script and notebooks. Follow the README.MD')


def main():
    parser_aux = prepare_parser_ray()
    tune_config, config_for_update = obtain_config(parser_aux)
    analysis_and_summary(tune_config, config_for_update, early_return = False)


if __name__ == '__main__':
  main()
