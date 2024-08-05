'''
Obtain metrics script
'''
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time as computing_time
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, ConfusionMatrix, \
                         AUROC, AveragePrecision

def rmetrics_post_loop(list_dict_metrics_ext, config):
  return metrics_post_loop(reduce_metrics(list_dict_metrics_ext), config)

def reduce_metrics(list_dict_metrics_ext):
  pd_metrics_ext   = pd.DataFrame(list_dict_metrics_ext)
  metrics_ext  = {}
  for key in pd_metrics_ext.columns:
    metrics_ext[key]  = np.nan_to_num(np.concatenate(pd_metrics_ext[key].to_numpy() ))
  return metrics_ext

def stack_metrics(list_dict_metrics_ext):
  pd_metrics_ext   = pd.DataFrame(list_dict_metrics_ext)
  metrics_ext  = {}
  for key in pd_metrics_ext.columns:
    metrics_ext[key]  = np.nan_to_num(np.stack(pd_metrics_ext[key].to_numpy(), 0))
  return metrics_ext

def classifier_metrics_fns_SSL(E, config, my_loss, SPM, **kwargs):
  def metrics_step(data, time, labels, mask, mask_detection = None, data_var = None,
                   tabular_feat = None, global_step = 0, **kwargs):
    dict_m = {}
    SPM.bound_dict(dict_m)
    init_time = computing_time.time()
    SPM.am('y', labels.data.cpu().numpy(), score_type = 'label')
    #SPM.am('y', labels.data.cpu(), score_type = 'label')
    with torch.no_grad():
        log_y_pred_dict    = E.predict_all(data = data,  data_var = data_var,
                                                  time = time,
                                                  mask = mask,
                                                  tabular_feat = tabular_feat,
                                                  global_step = global_step)
        if config['predict_obj'] == 'tab':
          log_y_pred_dict  = {'MLPMix': log_y_pred_dict['MLPTab']}
        for key in log_y_pred_dict.keys():
          SPM.amal('y_pred_vec_%s' % (key),
                    log_y_pred_dict[key].exp().data.cpu().numpy(), score_type = 'prob')
                    #log_y_pred_dict[key].exp().data.cpu(), score_type = 'prob')
    lapse_time = computing_time.time() - init_time
    SPM.amal('TimeComputed', (np.ones(data.shape[0]) * lapse_time)/data.shape[0], score_type = None)    
    SPM.delete_bound()
    return dict_m
  return metrics_step