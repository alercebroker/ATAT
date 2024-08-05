import glob
import h5py
import torch
import json

from joblib import load

import importlib
import yaml
import tqdm

import numpy as np
import copy

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def obtain_valid_mask(sample, mask, time_alert, eval_time):
    mask_time = (time_alert <= eval_time).float()
    sample['mask'] = mask * mask_time
    return sample


def get_tabular_data(feat_col, dict_add_feat_col, config_used, eval_time):
    new_feat_col = copy.deepcopy(feat_col)

    if config_used['using_metadata']:

        if not config_used['not_quantile_transformer']:
            metadata_qt = load('./final_dataset/QT-New/md_fold_{}.joblib'.format(config_used['seed']))
            new_feat_col  = torch.from_numpy(metadata_qt.transform(feat_col))

        if config_used['using_features']:
            add_feat_col   = dict_add_feat_col['{}'.format(eval_time)]

            if config_used['not_quantile_transformer']:
                new_add_feat_col = copy.deepcopy(add_feat_col)

            else:
                features_qt = load('./final_dataset/QT-New/fe_2048_fold_{}.joblib'.format(config_used['seed']))
                new_add_feat_col  = torch.from_numpy(features_qt.transform(add_feat_col))
                
            new_feat_col_total = torch.cat([new_feat_col, new_add_feat_col], dim=1)

        else:
            new_feat_col_total = copy.deepcopy(new_feat_col)
            
    else:
        if config_used['using_features']:
            add_feat_col   = dict_add_feat_col['{}'.format(eval_time)]

            if config_used['not_quantile_transformer']:
                new_feat_col_total = copy.deepcopy(add_feat_col)
            else:
                features_qt = load('./final_dataset/QT-New/fe_2048_fold_{}.joblib'.format(config_used['seed']))
                new_feat_col_total  = torch.from_numpy(features_qt.transform(add_feat_col))

        else:
            new_feat_col_total = copy.deepcopy(new_feat_col)
       
    return new_feat_col_total


def load_data(file_path, use_mask_alert=True, use_time_phot=True, use_time_alert=False):
    print('Loading data ...')
    h5_file   = h5py.File(file_path)
    these_idx  = h5_file.get('test')[:]

    data           = torch.from_numpy(h5_file.get('data')[:][these_idx])
    data_var       = torch.from_numpy(h5_file.get('data-var')[:][these_idx])
    if not use_mask_alert:
        mask       = torch.from_numpy(h5_file.get('mask')[:][these_idx])
    else:
        mask       = torch.from_numpy(h5_file.get('mask_alert')[:][these_idx])
    mask_detection = torch.from_numpy(h5_file.get('mask_detection')[:][these_idx])
    time           = torch.from_numpy(h5_file.get('time')[:][these_idx])
    time_alert     = torch.from_numpy(h5_file.get('time_alert')[:][these_idx])
    time_phot      = torch.from_numpy(h5_file.get('time_phot')[:][these_idx])
    target         = torch.from_numpy(h5_file.get('labels')[:][these_idx])
    feat_col       = torch.from_numpy(h5_file.get('norm_feat_col')[:][these_idx])

    if use_time_alert:
        time = time_alert
    if use_time_phot:
        time = time_phot

    print('- loading calculated features ...')
    dict_add_feat_col = dict()
    for eval_time in list_eval_time:
        dict_add_feat_col['{}'.format(eval_time)] = torch.from_numpy(h5_file.get('norm_add_feat_col_{}'.format(eval_time))[:][these_idx])

    h5_file.close()

    data_dict = {
        "data": data.float(),
        "data_var": data_var.float(),
        "time": time.float(),
        "mask": mask.float(),
        "mask_detection": mask_detection.float(),
        "time_alert": time_alert.float(),
        "mask": mask.float(),
        "tabular_feat": feat_col.float().unsqueeze(2),
        "labels": target.long(),
    }

    return data_dict, feat_col, dict_add_feat_col, target


def get_chunks(data_dict_eval_time, eval_time, batch_size):
    data_dict_eval_time['test_{}'.format(eval_time)]['data'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['data'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['data']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['data_var'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['data_var'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['data_var']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['time'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['time'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['time']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['mask'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['mask'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['mask']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['mask_detection'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['mask_detection'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['mask_detection']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['time_alert'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['time_alert'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['time_alert']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat']), batch_size)]
    data_dict_eval_time['test_{}'.format(eval_time)]['labels'] = \
        [data_dict_eval_time['test_{}'.format(eval_time)]['labels'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['labels']), batch_size)]

    return data_dict_eval_time


def get_predictions(path_results, target, batch_size):

    path_files_results = glob.glob('{}/*'.format(path_results))
    for path_results in path_files_results:
        path_exps = glob.glob('{}/Exp_cfg_*'.format(path_results))

        for path_exp in path_exps:
            print('\nLoading model ...')
            print('path_exp: {}'.format(path_exp))

            with open('./{}/hparams.yaml'.format(path_exp), "r") as stream:
                config_used = yaml.safe_load(stream)

            model_name = 'ClassifierModel'
            model_module = getattr(importlib.import_module('main_model'), model_name)
            model = model_module(**config_used)
            model_loaded = model.load_from_checkpoint('{}'.format(glob.glob('./{}/my_best*'.format(path_exp))[0])).E.eval().to(gpu)

            print('Use static features? {}'.format(config_used['using_metadata']))
            print('Use calculated features? {}'.format(config_used['using_features']))
            print('Use QT? {}'.format(not config_used['not_quantile_transformer']))

            # Generate batches over time
            data_dict_eval_time = dict()
            for eval_time in list_eval_time:
                new_feat_col_total = get_tabular_data(feat_col, dict_add_feat_col, config_used, eval_time)
                data_dict['tabular_feat'] = new_feat_col_total.float().unsqueeze(2)

                data_dict_eval_time['test_{}'.format(eval_time)]  = obtain_valid_mask(data_dict.copy(), 
                                                                                    data_dict.copy()['mask'], 
                                                                                    data_dict.copy()['time_alert'], 
                                                                                    eval_time)

                data_dict_eval_time = get_chunks(data_dict_eval_time, eval_time, batch_size)


            pred_time = dict()
            for _, set_eval_time in enumerate(data_dict_eval_time.keys()):
                print('#----------------- Testing in evaluation time: {} -----------------#'.format(set_eval_time.upper()))

                pred = []
                for i in range(len(data_dict_eval_time[set_eval_time]['data'])):
                    print('Batch numero {}'.format(i))

                    batch = dict()
                    batch['data'] = data_dict_eval_time[set_eval_time]['data'][i].to(gpu)
                    batch['data_var'] = data_dict_eval_time[set_eval_time]['data_var'][i].to(gpu)
                    batch['time'] = data_dict_eval_time[set_eval_time]['time'][i].to(gpu)
                    batch['mask'] = data_dict_eval_time[set_eval_time]['mask'][i].to(gpu)
                    batch['mask_detection'] = data_dict_eval_time[set_eval_time]['mask_detection'][i].to(gpu)
                    batch['time_alert'] = data_dict_eval_time[set_eval_time]['time_alert'][i].to(gpu)
                    batch['tabular_feat'] = data_dict_eval_time[set_eval_time]['tabular_feat'][i].to(gpu)

                    if path_exp.split('/')[-1].find('lc') != -1:
                        with torch.no_grad():
                            pred.append(model_loaded.predict_mix(**batch))    
                    else:
                        print('We are using Tabular Classifier...')
                        with torch.no_grad():
                            pred.append(model_loaded.predict_tab(**batch))

                pred_time[set_eval_time] = pred

            ################################################################################################

            # Concatening the predictions
            dict_pred_time = dict()
            for key, data_eval_time in pred_time.items():
                pred_time_concat = []
                
                for i in range(len(data_eval_time)):
                    if path_exp.split('/')[-1].find('lc') != -1:
                        pred_time_concat.append(data_eval_time[i]['MLPMix'].to('cpu'))
                    else:
                        pred_time_concat.append(data_eval_time[i]['MLPTab'].to('cpu'))
                    

                dict_pred_time[key] = np.exp(np.concatenate(pred_time_concat))

            ################################################################################################

            # Save predictions
            print('Predictions saved in {}'.format(path_exp))
            final_dict = {
                'y_test':target.numpy(),
                'list_y_pred_times':dict_pred_time.copy(),
            }

            torch.save(final_dict, "./{}/predictions_times.pt".format(path_exp)) 

            del model_loaded
            


if __name__ == '__main__':
    gpu = 'cuda'

    path_results = 'results_paper'

    list_eval_time = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    batch_size = 256

    data_root = 'final_dataset'
    file_path = './{}/{}'.format(data_root, 'elasticc_final.h5')

    data_dict, feat_col, dict_add_feat_col, target = load_data(file_path)

    get_predictions(path_results, target, batch_size)
