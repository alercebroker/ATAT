
'''
Utilities file
'''
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import json
import pickle

import torch
import matplotlib.pyplot as plt
import datasets

import glob
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Utility file to seed rngs
def seed_rng(seed): 
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

def make_root(root, key, config = None, root_name = None):
  aux_path      = '%s/%s' % (root, key)
  if not os.path.exists(aux_path):
    os.mkdir(aux_path)
  if config is not None:
    if root_name is None:
      config[key]        = aux_path
    else:
      config[root_name]  = aux_path
  return aux_path

def obtain_general_root(config, folder_number, set_type = 'val'):
  exp_root     = config['current_dir']
  exp_root     = make_root(exp_root, 'samples', config)
  fn_root      = make_root(exp_root, folder_number, config)
  scorer_root  = make_root(fn_root, 'scorer_root', config)
  fn_root      = make_root(fn_root, set_type, config)
  return config

def obtain_final_reloaded_root(config, add_name = ''):
  exp_root     = config['current_dir']
  exp_root     = make_root(exp_root, 'reloaded_root' + add_name, config, 'reloaded_root')
  return config


def save_json(file_to_save, folder_root, file_name):
  out_file = open('%s/%s' % (folder_root, '%s.json' % file_name), "w")
  json.dump(file_to_save, out_file)
  out_file.close()  
def load_json(folder_root, file_name):
  out_file = open('%s/%s' % (folder_root, '%s.json' % file_name), "r")
  aux_file = json.load(out_file)
  out_file.close()
  return aux_file  
def save_pickle(file_to_save, folder_root, file_name):
  out_file = open('%s/%s' % (folder_root, '%s.pkl' % file_name), "wb")
  pickle.dump(file_to_save, out_file)
  out_file.close()  
def load_pickle(folder_root, file_name):
  out_file = open('%s/%s' % (folder_root, '%s.pkl' % file_name), "rb")
  aux_file = pickle.load(out_file)
  out_file.close()  
  return aux_file

# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off

def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])

# Convenience function to count the number of parameters in a module
def count_parameters(module):
  print('Number of parameters: {}'.format(
    sum([p.data.nelement() for p in module.parameters()])))
   
# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
  return torch.randint(low=0, high=num_classes, size=(batch_size,),
          device=device, dtype=torch.int64, requires_grad=False)

### Plot stuff ####
def cluster_acc_torch(Y_pred, Y):
    D = int(torch.max(torch.max(Y_pred), torch.max(Y))+1)
    ww = torch.zeros(D,D)
    for i in range(Y_pred.size()[0]):
        ww[int(Y_pred[i]), int(Y[i])] += 1
    accuracy = torch.sum(torch.max(ww, dim = 1)[0])/torch.sum(ww)
    return accuracy

def obtain_cluster_transformation(Y_pred, Y):
    D = int(torch.max(torch.max(Y_pred), torch.max(Y))+1)
    ww = torch.zeros(D,D)
    for i in range(Y_pred.size()[0]):
        ww[int(Y_pred[i]), int(Y[i])] += 1
    return ww.argmax(1)

def this_print(a):
  print(a)

def update_config(config):
  config['data_root'] = '%s/%s' % (config['abs_path'], datasets.root_dict[config['dataset']])
  print('Using dataset root location %s' % config['data_root'])
  config['n_classes']       = datasets.nclass_dict[config['dataset']]  
  config['classes_names']   = np.array(datasets.classes_names[config['dataset']])
  if config['is_anomaly_detection']:
    config['n_classes']     = config['n_classes'] - 1
    config = datasets.anomaly_reindexation(config)
  config['dataset_channel'] = datasets.channel_dict[config['dataset']]
  config['dict_set_types']  = datasets.dir_dset_dict[config['dataset']]
  config['largest_seq']     = datasets.seq_dict[config['dataset']]
  config['T_max']           = datasets.T_max[config['dataset']]
  config['band_colors_obs'] = np.array(datasets.band_colors_obs[config['dataset']])
  config['band_colors_mod'] = np.array(datasets.band_colors_mod[config['dataset']])
  config['band_legend']     = np.array(datasets.band_legend[config['dataset']])
  config['noise_data']      = datasets.noise_data[config['dataset']]
  config['n_total_bands']   = config['n_total_bands'] if config['n_total_bands'] \
                                                      else config['n_classes']
  if config['using_metadata'] and config['using_features']:
    config['F_max'] = datasets.elasticc_feat_values[config['which_tabular_feat']] + \
                                      datasets.elasticc_feat_values['add_' + config['which_tabular_feat']]
  elif config['using_metadata']:
    config['F_max'] = datasets.elasticc_feat_values[config['which_tabular_feat']]
  elif config['using_features']:
    config['F_max'] = datasets.elasticc_feat_values['add_' + config['which_tabular_feat']]
  else:
    config['F_max'] = datasets.elasticc_feat_values[config['which_tabular_feat']] + \
                                      datasets.elasticc_feat_values['add_' + config['which_tabular_feat']]
  if 'F_max' in config.keys():
    config['F_len'] = len(config['F_max'])

  config['elasticc_feat_names']  = datasets.elasticc_feat_names[config['which_tabular_feat']]
  config['abs_path']        = os.path.abspath('.')
  return config

#---------------------------------------- PLOT FUNCTIONS ----------------------------------------#

def get_metrics(list_path_predictions):

    f1_results_time = dict()
    acc_results_time = dict()
    precision_results_time = dict()
    recall_results_time = dict()
    for i, path_predictions in enumerate(sorted(glob.glob(list_path_predictions))):

        try:
            predictions = torch.load('./{}/predictions_times.pt'.format(path_predictions))

            target = predictions['y_test']
            dict_pred_times = predictions['list_y_pred_times']

            f1_results_time['fold_{}'.format(i)] = dict()
            acc_results_time['fold_{}'.format(i)] = dict()
            precision_results_time['fold_{}'.format(i)] = dict()
            recall_results_time['fold_{}'.format(i)] = dict()

            for set_type, batches_pred in dict_pred_times.items():
                f1_results_time['fold_{}'.format(i)][set_type] = f1_score(target, 
                                                                          np.argmax(batches_pred, axis = 1), 
                                                                          average='macro')

                acc_results_time['fold_{}'.format(i)][set_type] = accuracy_score(target, 
                                                                                np.argmax(batches_pred, axis = 1))

                precision_results_time['fold_{}'.format(i)][set_type] = precision_score(target, 
                                                                                        np.argmax(batches_pred, axis = 1), 
                                                                                        average='macro')

                recall_results_time['fold_{}'.format(i)][set_type] = recall_score(target, 
                                                                                np.argmax(batches_pred, axis = 1), 
                                                                                average='macro')
        except FileNotFoundError:
            print('We dont have the prediction file of: {} yet'.format(path_predictions))

    return f1_results_time, acc_results_time, precision_results_time, recall_results_time


def plot_result_ablation(list_eval_time, models_f1_results_time, ylim_min=0, ylim_max=90):
    fs = 17

    plt.figure(figsize=(8, 7))

    for name_model, f1_results_time in models_f1_results_time.items():

        list_f1_mean_fold = []
        list_f1_std_fold = []
        for eval_time in list_eval_time:
            # Max length
            if eval_time == 1105:
               eval_time = 2048

            f1_folds = []
            for _, results in f1_results_time.items():
                f1_folds.append(results['test_{}'.format(eval_time)])

            list_f1_mean_fold.append(np.mean(f1_folds)*100)
            list_f1_std_fold.append(np.std(f1_folds)*100)

        list_f1_mean_fold = np.array(list_f1_mean_fold)
        list_f1_std_fold = np.array(list_f1_std_fold)

        plt.plot(np.log(list_eval_time)/np.log(2), list_f1_mean_fold, label=name_model)  
        plt.fill_between(np.log(list_eval_time)/np.log(2), 
                            list_f1_mean_fold + list_f1_std_fold, 
                            list_f1_mean_fold - list_f1_std_fold, 
                            alpha=0.5)
            
    list_eval_time = list_eval_time[:-1]
    plt.yticks(fontsize=fs)
    plt.xticks(np.log(list_eval_time)/np.log(2), list_eval_time, fontsize=fs)

    plt.ylabel("F1-Score", fontsize=fs, labelpad=13)
    plt.xlabel("Evaluated time (days after first alert)", fontsize=fs, labelpad=13)

    plt.ylim((ylim_min, ylim_max))
    plt.legend(fontsize=fs, loc='lower right')

    #plt.savefig("all_curves_teval_a_last.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_values_times(list_eval_time, models_f1_results_time, title=None, ylim_min=0, ylim_max=90, colors=None, f1_added=None):
    fs = 17

    plt.figure(figsize=(8, 7))

    if title is not None:
        plt.title(title, size=20, pad=13)

    for name_model, f1_results_time in models_f1_results_time.items():

        list_f1_mean_fold = []
        list_f1_std_fold = []
        for eval_time in list_eval_time:

            # Max length
            if eval_time == 1105:
               eval_time = 2048

            f1_folds = []
            for _, results in f1_results_time.items():
                f1_folds.append(results['test_{}'.format(eval_time)])

            list_f1_mean_fold.append(np.mean(f1_folds)*100)
            list_f1_std_fold.append(np.std(f1_folds)*100)

        list_f1_mean_fold = np.array(list_f1_mean_fold)
        list_f1_std_fold = np.array(list_f1_std_fold)
        
        if name_model.find('MTA') != -1: 
            plt.plot(np.log(list_eval_time)/np.log(2), list_f1_mean_fold, label=name_model, linestyle='dashed', color=colors[name_model])
            plt.fill_between(np.log(list_eval_time)/np.log(2), 
                            list_f1_mean_fold + list_f1_std_fold, 
                            list_f1_mean_fold - list_f1_std_fold, 
                            alpha=0.4, color=colors[name_model])

        else:
          if name_model != 'ATAT (MD)':
            plt.plot(np.log(list_eval_time)/np.log(2), list_f1_mean_fold, label=name_model, color=colors[name_model])
            plt.fill_between(np.log(list_eval_time)/np.log(2), 
                             list_f1_mean_fold + list_f1_std_fold, 
                             list_f1_mean_fold - list_f1_std_fold, 
                             alpha=0.5, color=colors[name_model])
            
          else:
            list_f1_mean_fold = np.insert(list_f1_mean_fold, 0, list_f1_mean_fold[0])
            list_f1_std_fold = np.insert(list_f1_std_fold, 0, list_f1_std_fold[0])
            plt.plot(np.log([0.7] + list_eval_time)/np.log(2), list_f1_mean_fold, label=name_model, color=colors[name_model])
            plt.fill_between(np.log([0.7] + list_eval_time)/np.log(2), 
                             list_f1_mean_fold + list_f1_std_fold, 
                             list_f1_mean_fold - list_f1_std_fold, 
                             alpha=0.5, color=colors[name_model])
            
    if f1_added is not None:
        plt.plot(np.log(list_eval_time)/np.log(2), f1_added['mean']*100, label='RF (Features + MD)', linestyle='-', color='#e41a1c')
        plt.fill_between(np.log(list_eval_time)/np.log(2), 
                        f1_added['mean']*100 + f1_added['std']*100, 
                        f1_added['mean']*100 - f1_added['std']*100, 
                        alpha=0.4, color='#e41a1c')
  
    list_eval_time = list_eval_time[:-1]
    plt.yticks(fontsize=fs)
    plt.xticks(np.log(list_eval_time)/np.log(2), list_eval_time, fontsize=fs)

    plt.ylabel("F1-Score", fontsize=fs, labelpad=13)
    plt.xlabel("Evaluated time (days after first alert)", fontsize=fs, labelpad=13)

    plt.ylim((ylim_min, ylim_max))
    plt.xlim(left=-0.5)
    plt.legend(fontsize=fs, loc='lower right')

    plt.show()

def f1_values(list_eval_time, models_f1_results_time):
    dict_mean = dict()
    dict_std = dict()

    for name_model, f1_results_time in models_f1_results_time.items():

        list_f1_mean_fold = []
        list_f1_std_fold = []
        for eval_time in list_eval_time:
            f1_folds = []
            for _, results in f1_results_time.items():
                f1_folds.append(results['test_{}'.format(eval_time)])

            list_f1_mean_fold.append(np.mean(f1_folds)*100)
            list_f1_std_fold.append(np.std(f1_folds)*100)

        list_f1_mean_fold = np.array(list_f1_mean_fold)
        list_f1_std_fold = np.array(list_f1_std_fold)

        dict_mean[name_model] = list_f1_mean_fold
        dict_std[name_model] = list_f1_std_fold

    return dict_mean, dict_std


#---------------------------------------- TABLE FUNCTIONS ----------------------------------------#

def get_pred_and_real(path_root, path_lc_md_feat_mta, classes):
    test_proba_model = []
    num_fold = 5
    for i_fold in range(num_fold):
        predictions = torch.load('./{}/{}-seed={}/predictions_times.pt'.format(path_root, path_lc_md_feat_mta, i_fold))
        df_y_pred = pd.DataFrame(np.array(predictions['list_y_pred_times']['test_2048']),
                                columns=classes)
        
        test_proba_model.append(df_y_pred)

        for idx_row in range(test_proba_model[i_fold].shape[0]):
            idx_col = test_proba_model[i_fold].iloc[idx_row].argmax()
            test_proba_model[i_fold].iloc[idx_row][idx_col] = 1

        test_proba_model[i_fold][test_proba_model[i_fold] != 1] = 0
        
    # Same test for each fold
    n_values = np.max(predictions['y_test']) + 1
    y_test = np.eye(n_values)[predictions['y_test']]
    df_y_test = pd.DataFrame(y_test,
                            columns=classes)

    return test_proba_model, df_y_test

  
def get_classification_report(test_proba_model, df_y_test, num_folds):
    report_metrics = []
    for i_fold in range(num_folds):
        report_metrics.append(classification_report(df_y_test.reindex(test_proba_model[i_fold].index).values, test_proba_model[i_fold].values, 
                            target_names=df_y_test.columns, output_dict=True))

    dict_mean_classes = dict()
    dict_std_classes = dict()

    for metric in ['f1-score', 'precision', 'recall']:
        dict_mean_classes[metric] = dict()
        dict_std_classes[metric] = dict()

        for label in df_y_test.columns: 
            list_mean_class = []
            list_std_class = []
            for i_fold in range(num_folds):
                list_mean_class.append(report_metrics[i_fold][label][metric] * 100)
                list_std_class.append(report_metrics[i_fold][label][metric] * 100)

            dict_mean_classes[metric][label] = round(np.mean(list_mean_class), 2)
            dict_std_classes[metric][label] = round(np.std(list_std_class), 2)

    return dict_mean_classes, dict_std_classes, report_metrics

#---------------------------------------- CONFUSION MATRIX ----------------------------------------#

def get_y_test_pred_folds(path_exp, dict_classes, folds):
    list_pred_folds = []
    for fold in folds:
        path_exp_fold = '{}-seed={}/predictions_times.pt'.format(path_exp, fold)
        pred = torch.load(path_exp_fold)

        y_test = pred['y_test']
        y_prob_pred = pred['list_y_pred_times']['test_2048']
        y_pred = np.argmax(y_prob_pred, axis=1)

        y_test_label = [dict_classes[x] for x in y_test]
        y_pred_label = [dict_classes[x] for x in y_pred]

        list_pred_folds.append((y_test_label, y_pred_label))

    return list_pred_folds


def get_confusion_matrix(mean_matrix, std_matrix, order_classes):
    # Graficando la matriz media
    cmap = plt.cm.Blues
    fig, ax = plt.subplots(figsize=(13, 13)) #, dpi=110)
    im = ax.imshow(np.around(mean_matrix, decimals=2), interpolation='nearest', cmap=cmap)

    # color map
    new_color = cmap(1.0) 

    # Añadiendo manualmente las anotaciones con la media y desviación estándar
    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            if mean_matrix[i, j] >= 0.005:
                #print(mean_matrix[i, j])
                text = f'{np.around(mean_matrix[i, j], decimals=2)}\n±{np.around(std_matrix[i, j], decimals=2)}'
                color = "white" if mean_matrix[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10.5)
            else:
                text = f'{np.around(mean_matrix[i, j], decimals=2)}'
                color = "white" if mean_matrix[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10.5)

    # Ajustes finales y mostrar la gráfica
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(np.arange(len(order_classes)))
    ax.set_yticks(np.arange(len(order_classes)))
    ax.set_xticklabels(order_classes)
    ax.set_yticklabels(order_classes)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.xaxis.labelpad = 13
    ax.yaxis.labelpad = 13

    plt.tight_layout()
    plt.show()