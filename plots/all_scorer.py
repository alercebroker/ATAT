import sys
sys.path.append('../')
import utils
import numpy as np
from . import plot_scorer as psr
from sklearn.metrics import classification_report
#VEC_KEYS    = ['ConfusionMatrix', 'AllReport']
VEC_KEYS    = ['ConfusionMatrix']
TIMES_SCORE = ['']


def is_substring_of_list(key):
    for this_string in VEC_KEYS:
        if this_string in key:
            return True
    return False

def compute_all_report(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    aux_str   = classification_report(y_true, y_pred)
    all_list  = aux_str.split('\n')[2 : n_classes + 2]
    all_data  = []
    for idx in range(len(all_list)):
        #import pdb
        #pdb.set_trace()
        this_sent  = all_list[idx].split('     ')
        all_data  += [[float(this_sent[3]), float(this_sent[4]), float(this_sent[5])]]
    all_data  = np.array(all_data)
    return all_data

def plot_vec_metrics(metrics_vec, set_type, config, metrics_vec_std = None, root_name = 'scorer_root'):
    for key in VEC_KEYS:
        for metric_name in metrics_vec.keys():
            if 'ConfusionMatrix' in metric_name:
                psr.plot_confusion_matrix(metrics_vec[metric_name], config['classes_names'],
                            filename = '%s/%s' % (config[root_name], '%s_confusion_matrix_%s' % (set_type, metric_name) ),
                            cm_std   =  metrics_vec_std[metric_name] if metrics_vec_std is not None else None) 
                psr.plot_confusion_matrix(metrics_vec[metric_name], config['classes_names'],
                            filename = '%s/%s' % (config[root_name], 'JustMean_%s_confusion_matrix_%s' % (set_type, metric_name) ),
                            cm_std   =  None) 
            if 'AllReport' in metric_name:
                psr.plot_all_report(metrics_vec[metric_name], config['classes_names'],
                            root = config[root_name],
                            filename =  'Table_%s_all_report_%s' % (set_type, metric_name),
                            cm_std   =  None) 

def plot_common(metrics_stat, metrics_ext, config, set_type):
    utils.save_json(metrics_stat, config['scorer_root'], '%s_results' % set_type)
    metrics_dict_aux = {key: metrics_ext[key] for key in metrics_ext.keys() if is_substring_of_list(key)}
    utils.save_pickle(metrics_dict_aux, config['scorer_root'], '%s_results_vec' % set_type)
    plot_vec_metrics(metrics_dict_aux, set_type, config, root_name = 'scorer_root')

def plot_reloaded_scores(metrics_stat, metrics_ext, config, set_type):
    utils.save_json(metrics_stat, config['reloaded_root'], '%s_results' % set_type)
    metrics_dict_aux = {key: metrics_ext[key] for key in metrics_ext.keys() if is_substring_of_list(key)}
    utils.save_pickle(metrics_dict_aux, config['reloaded_root'], '%s_results_vec' % set_type)
    plot_vec_metrics(metrics_dict_aux, set_type, config, root_name = 'reloaded_root')

def plot_multiple_set_type(m_metrics_stat, config, is_reloaded = False, metric_name = 'time', set_type = 'test'):
    root_name = config['reloaded_root'] if is_reloaded else config['scorer_root'] 
    psr.summarize_eval_metric_many(m_metrics_stat, root_name, metric_name = metric_name, set_type = set_type)