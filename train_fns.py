''' train_fns.py
Functions for the main loop of training
'''
import torch

def classifier_training_function_SSL(E, config, my_loss, **kwargs):

  if config['predict_obj'] == 'lc':
      predict_obj = E.predict_lc
  elif config['predict_obj'] == 'tab':
      predict_obj = E.predict_tab
  elif config['predict_obj'] == 'mix':
      predict_obj = E.predict_mix
  elif config['predict_obj'] == 'all':
      predict_obj = E.predict_all

  def train_step(data, time, labels, mask, data_var = None, tabular_feat = None, global_step = 0, **kwargs):
    loss = 0
    log_y_pred_dict = predict_obj(data = data, data_var = data_var,
                                        time = time, mask = mask,
                                        tabular_feat = tabular_feat,
                                        global_step = global_step)

    for key in log_y_pred_dict.keys():
      loss += my_loss.cross_entropy_st(labels, log_y_pred_dict[key])

    out = {'loss': loss}

    return out
  
  return train_step