import plots.all_plots as pall
import plots.all_scorer as pallsr
import plots.all_holoview as pallhv
import plots
import numpy as np
import torch
import pandas as pd
import sys
sys.path.append('../')
import utils
from sklearn.metrics import precision_recall_curve
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, ConfusionMatrix, \
                         AUROC, AveragePrecision
from torchmetrics import F1Score
from sklearn import metrics
import plots.plot_hist as plot_hist
import os

class ScorePlotManager:
  def __init__(self, config, plt_cfg):
    self.is_first_iter = True
    self.mmanager     = pd.DataFrame()
    self.config       = config
    self.plt_cfg      = plt_cfg
    self.key_tlabel   = 'y'
    self.anomaly_list = ['test']
    self.posible_dist = ['train_step', 'val', 'test']
    self.score_types  = [None, 'label', 'prob', 'latent', 'obs_rep']
    self.key_list     = []

  def is_update_manager(self, key):
    if key in self.key_list:
      return False
    else:
      self.key_list += [key]
      return True
  
  def get_mmanager(self, metrics_ext):
    list_metrics_ext = list(metrics_ext.keys())
    return self.mmanager[self.mmanager['key'].isin(list_metrics_ext) ]

  def bound_dict(self, this_dict):
    self.metrics_dict = this_dict

  def delete_bound(self):
    delattr(self, 'metrics_dict')

  def get_n_classes(self, set_type):
    if self.config['is_anomaly_detection'] and set_type in self.anomaly_list:
      return self.config['n_classes'] + 1
    else:
      return self.config['n_classes']

  def plot_anomaly(self, set_type):
    if self.config['is_anomaly_detection'] and set_type in self.anomaly_list:
      return True
    else:
      return False

  def get_list_an_score(self):
    m_to_process =  self.mmanager[self.mmanager['anomaly_score'] != 0]
    return m_to_process[m_to_process['len_shape_score'] == 1]['key'].to_list()

  def add_to_manager(self, key = '', value = None, score_type = None, process_metric =  False,
                      anomaly_score = 0.0, to_hist = False, to_scatter = False,
                      plot_with = None, **kwargs):
    if self.is_update_manager(key):
      len_shape_score = len(value.shape) if torch.is_tensor(value) or type(value) is np.ndarray else 0
      is_class_score =  True if len_shape_score == 2 and value.shape[1] == self.config['n_classes'] else False
      # we assume that value is either a tensor or a float
      self.mmanager = self.mmanager.append({'key': key,
                          'len_shape_score': len_shape_score,
                          'is_class_score': is_class_score,
                          'score_type': score_type,
                          'process_metric': process_metric,
                          'anomaly_score': anomaly_score,
                          'to_hist': to_hist,
                          'to_scatter': to_scatter,
                          'plot_with': plot_with}, ignore_index = True)

  def am_ext(self, metrics_ext, key, value, **kwargs):
    metrics_ext[key] = value
    self.add_to_manager(**{'key':key, 'value': value, **kwargs})

  def amal_ext(self, metrics_ext, key, value, **kwargs):
    metrics_ext[key] = value
    self.add_to_manager(**{'key':key, 'value': value, 'process_metric': True, 'to_scatter': True, 'to_hist': True, **kwargs})

  def am_justp_ext(self, metrics_ext, key, value, **kwargs):
    metrics_ext[key] = value
    self.add_to_manager(**{'key':key, 'value': value, 'process_metric': True, 'to_scatter': False, 'to_hist': False, **kwargs})

  def am(self, key, value, **kwargs):
    self.metrics_dict[key] = value
    self.add_to_manager(**{'key':key, 'value': value, **kwargs})

  def amal(self, key, value, **kwargs):
    self.am(**{'key':key, 'value': value, 'process_metric': True, 'to_scatter': True, 'to_hist': True, **kwargs})

  def amp(self, key, value, **kwargs):
    self.am(**{'key':key, 'value': value, 'to_scatter': True, 'to_hist': True, **kwargs})

  def obtain_metrics_stat(self, metrics_ext):
      key_list_stat = self.mmanager[self.mmanager['len_shape_score'] == 0]['key'].to_list()
      return {key: float(metrics_ext[key]) for key in key_list_stat if key in metrics_ext.keys()}

  def post_metrics(self, metrics_ext, set_type = 'val'):
    # we should process al the scores process 1d and 2d, metrics for plotting are not necessary to be here
    self.process_2d_metrics(metrics_ext, set_type)
    self.process_1d_metrics(metrics_ext, set_type)
    if set_type is not None and set_type in self.plt_cfg.keys():
      if self.plt_cfg[set_type]['plt_r'] and self.config['is_lc_dataset']:
        self.scale_metrics(metrics_ext)
    return self.obtain_metrics_stat(metrics_ext), metrics_ext

  def process_2d_metrics(self, metrics_ext, set_type):
      mmanager = self.get_mmanager(metrics_ext)
      m_to_process = mmanager[mmanager['process_metric'] == True]
      m_to_process = m_to_process[m_to_process['len_shape_score'] == 2]
      for i, row in m_to_process.iterrows():
          if row['score_type'] is None or row['score_type'] == 'obs_rep':
              self.process_any(metrics_ext, row, '_mean2', metrics_ext[row['key']].mean(1))
          if row['score_type'] == 'prob':
              self.process_y_pred_2d(row['key'], metrics_ext, set_type)

  def process_1d_metrics(self, metrics_ext, set_type):
      mmanager = self.get_mmanager(metrics_ext) 
      m_to_process = mmanager[mmanager['process_metric'] == True]
      m_to_process = m_to_process[m_to_process['len_shape_score'] == 1]
      for i, row in m_to_process.iterrows():
          if row['score_type'] is None or row['score_type'] == 'obs_rep':
              self.process_any(metrics_ext, row, '_mean1', metrics_ext[row['key']].mean(0))
          if row['score_type'] == 'label' and row['key'] != 'y' and not self.plot_anomaly(set_type):
              self.process_y_pred_1d(row['key'], metrics_ext)
          if row['anomaly_score'] != 0 and self.plot_anomaly(set_type):
              self.process_anomaly_1d(row['key'], row['anomaly_score'], metrics_ext)

  def process_any(self, metrics_ext, row, add_name, value):
      row_copy = dict(row).copy()
      row_copy.update({'key': '%s%s' % (row['key'], add_name), 'value': value})
      self.am_ext(metrics_ext, **row_copy)

  def process_y_pred_2d(self, key, metrics_ext, set_type):
    # y_pred_vec is the vector form
    add_name   = key.replace('y_pred_vec', '')
    y_pred_v   = metrics_ext[key]
    n_classes  = self.get_n_classes(set_type)
    self.am_justp_ext(metrics_ext,'%s%s' % ('y_agmax', add_name), y_pred_v.argmax(1), score_type = 'label' )
    self.amal_ext(metrics_ext,'%s%s' % ('y_entropy', add_name), (-y_pred_v * np.log(y_pred_v)).sum(1) )
    if not self.plot_anomaly(set_type):
      metrics_float   = MetricCollection([AUROC(num_classes = n_classes), AveragePrecision(num_classes = n_classes),
                                                    F1Score(average = 'macro', num_classes = self.config['n_classes'])])  
      metrics_results = metrics_float(torch.tensor(metrics_ext[key]), torch.tensor(metrics_ext['y']) )
      metrics_results['AvgPrecision'] = metrics_results.pop('AveragePrecision')
      PONDER_METRIC = 100
      for this_key in metrics_results.keys():
        self.am_ext(metrics_ext, '%s%s' % (this_key, add_name),
                             PONDER_METRIC * metrics_results[this_key].item())
      # y_pred_vec is the vector form
      metrics_vec     = MetricCollection([ConfusionMatrix(num_classes = n_classes, normalize = 'true')])
      metrics_results = metrics_vec(torch.tensor(metrics_ext[key]), torch.tensor(metrics_ext['y']) )
      for this_key in metrics_results.keys():
        self.am_ext(metrics_ext, '%s%s' % (this_key, add_name), metrics_results[this_key])

  def process_y_pred_1d(self, key, metrics_ext):
    add_name   = key.replace('y_agmax', '')
    y_pred     = metrics_ext[key]
    metrics_float   = MetricCollection([Accuracy(), Precision(), Recall()]) #, F1Score()])  
    metrics_results = metrics_float(torch.tensor(y_pred), torch.tensor(metrics_ext['y']) )
    PONDER_METRIC = 100
    for this_key in metrics_results.keys():
      self.am_ext(metrics_ext, '%s%s' % (this_key, add_name),
                      PONDER_METRIC * metrics_results[this_key].item())
    self.am_ext(metrics_ext, '%s%s' % ('Accuracy_array', add_name), (y_pred == metrics_ext['y']).astype(float) )
    self.am_ext(metrics_ext, '%s%s' % ('AllReport', add_name),
                                               pallsr.compute_all_report(metrics_ext['y'], y_pred))

  def process_anomaly_1d(self, key, sign_an_score, metrics_ext):
    if not ('labels_ad' in metrics_ext.keys()):
      labels_ad                  = np.ones(len(metrics_ext['y']))
      labels_ad[metrics_ext['y'] == self.config['n_classes']] = 0 #After changin the indexes      
      self.am_ext(metrics_ext, 'labels_ad', labels_ad)
    else:
      labels_ad = metrics_ext['labels_ad']
    an_score             = metrics_ext[key]  * sign_an_score
    norm_an_score        = (an_score - an_score.min() )/(an_score.max() - an_score.min())
    rank_an_score        = np.arange(len(an_score))[an_score.argsort().argsort()]/len(an_score)
    # precision recall don't support pos_label 0 so we have to invert the prediction
    precision, recall, _ = precision_recall_curve(1 - labels_ad, 1 - norm_an_score, pos_label=1)
    fpr, tpr, _          = metrics.roc_curve(1 - labels_ad, 1 - norm_an_score, pos_label=1)
    aucroc_an            = metrics.auc(fpr, tpr)
    aucpr_an             = metrics.auc(recall, precision)
    self.amal_ext(metrics_ext, 'an_score_' + key, an_score)
    self.amal_ext(metrics_ext, 'rank_an_score_' + key, rank_an_score)
    self.am_ext(metrics_ext, 'an_aucroc_' + key, aucroc_an)
    self.am_ext(metrics_ext, 'an_aucpr_'  + key, aucpr_an)
    
  def scale_metrics(self, metrics):
    ss = 'scaled_'
    means       = np.expand_dims(metrics['means'], 2)
    scales      = np.expand_dims(metrics['scales'],2)
    metrics[ss + 'D_mu']       = metrics['D_mu']    * scales + means
    metrics[ss + 'D_var']      = metrics['D_var']   * scales**2
    metrics[ss + 'D_sigma']    = metrics['D_sigma'] * scales
    metrics[ss + 'data']       = metrics['data']  * scales + means
    if self.config['which_post_decoder'] != '':
      metrics[ss + 'D_prot_x']   = metrics['D_prot_x']  * scales + means
    if self.config['is_dec_var']:
      metrics[ss + 'data_var']   = metrics['data_var'] * scales**2
      metrics[ss + 'data_sigma'] = metrics['data_sigma'] * scales
      if self.config['which_post_decoder'] != '':
        metrics[ss + 'D_prot_x_var'] = metrics['D_prot_x_var'] * scales**2
    if 'time_for' in metrics.keys():
      metrics[ss + 'D_mu_for'] = metrics['D_mu_for']*scales + means
      if self.config['is_dec_var']:
        metrics[ss + 'D_sigma_for'] = metrics['D_sigma_for']*scales
        metrics[ss + 'D_var_for'] = metrics['D_var_for']*scales**2
      if 'data_for' in metrics.keys():
        metrics[ss + 'data_for'] = metrics['data_for']*scales + means
      if 'data_var_for' in metrics.keys():
        metrics[ss + 'data_sigma_for'] = metrics['data_sigma_for']*scales
        metrics[ss + 'data_var_for'] = metrics['data_var_for']*scales**2

  def oneset_dist_plot(self, metrics_stat, metrics_ext, set_type = 'val'):
    pallsr.plot_common(metrics_stat, metrics_ext, self.config, set_type)
    folder_rec = utils.make_root(self.config[set_type], 'reconstruction_root')
    pall.plot_rec(metrics_ext, self.config, folder_rec, self.get_n_classes(set_type)) if self.plt_cfg[set_type]['plt_r'] else None
    self.hist_plots(metrics_ext, set_type) if self.plt_cfg[set_type]['plt_h'] else None
    self.scatter_plots(metrics_ext, set_type) if self.plt_cfg[set_type]['plt_s'] else None
    if self.plot_anomaly(set_type):
      list_scores = self.get_list_an_score()
      folder_an = utils.make_root(self.config[set_type], 'anomaly_root')
      #m_stat  = self.mmanager[self.mmanager['len_shape_score']  0]['key'].to_list()
      metrics_an = {key: metrics_ext[key] for key in metrics_ext.keys() if not (key in metrics_stat.keys())}
      for key_an_score in list_scores:
        pall.plot_rec_ascore(key_an_score, metrics_an, self.config, folder_an) if self.plt_cfg[set_type]['plt_r'] else None

  def multiset_dist_plot(self, dm_ext):
    # Plot histograms for N x C scores for SSL
    if 'train_step' in dm_ext.keys() and 'train_step_u' in dm_ext.keys():
      folder_3scores_ssl = utils.make_root(self.config[set_type], 'Hist_3scores_ssl')
      mmanager = self.get_mmanager(metrics_ext)
      m_to_process  = mmanager[mmanager['len_shape_score'] == 2]
      m_to_process  = m_to_process[m_to_process['is_scatter'] == True]
      m_to_process  = m_to_process[m_to_process['is_class_score'] == True]
      ssl_dict = \
        {'score_1': {'metrics': dm_ext['train_step'], 'filter': lambda lab, y_c: lab == y_c,'name': 'class %d (labeled)' },
        'score_2': {'metrics': dm_ext['train_step_u'], 'filter': lambda lab, y_c: lab == y, 'name': 'class %d (unlabeled)' },
        'score_3': {'metrics': dm_ext['train_step_u'], 'filter': lambda lab, y_c: lab != y_c, 'name': 'not class %d (unlabeled)'},
        'name': 'ssl_exp'}
      self.loop_plot_oneout(ssl_dict, m_to_process['key'].to_list(), folder_3scores_ssl, is_vec = True)

  def hist_plots(self, metrics_ext, set_type):
    mmanager = self.get_mmanager(metrics_ext)
    # Plot histograms 1d
    folder_scores = utils.make_root(self.config[set_type], 'Hist_scores')
    # Plot histograms 1d for acc
    list_acc_correct = [key for key in metrics_ext.keys() if 'Accuracy_array' in  key]
    folder_scores_acc = {key_name: utils.make_root(self.config[set_type], 'Hist_%s_scores' % key_name)\
                                    for key_name in list_acc_correct}
    # Plot histograms 1d for anomaly
    folder_scores_an = utils.make_root(self.config[set_type], 'Hist_scores_anomaly')
    m_to_process_h   = mmanager[mmanager['to_hist'] == True]
    m_to_process_1d  = m_to_process_h[m_to_process_h['len_shape_score'] == 1]

    for i, row in m_to_process_1d.iterrows():
      pall.plot_hist_1d(row['key'], metrics_ext, self.config, folder_scores)
      # Plot accuracy in histograms
      for acc_key in list_acc_correct:
        try:
          pall.plot_hist_1d_acc(row['key'], acc_key, metrics_ext, self.config, folder_scores_acc[acc_key])
        except:
          print("plot not possible")
      # Plot anomaly scores in histograms
      if self.plot_anomaly(set_type):
        pall.plot_hist_1d_labelad(row['key'], metrics_ext, self.config, folder_scores_an)
    # Plot histograms for N x C scores
    if self.plot_anomaly(set_type):
      folder_3scores_an = utils.make_root(self.config[set_type], 'Hist_3scores_anomaly')
      m_to_process_2d  = m_to_process_h[m_to_process_h['len_shape_score'] == 2]
      m_to_process_2d  = m_to_process_2d[m_to_process_2d['is_class_score'] == True]
      an_dict = \
        {'score_1': {'metrics': metrics_ext, 'filter': lambda lab, y_c: lab == y_c,'name': 'class %d' },
        'score_2': {'metrics': metrics_ext, 'filter': lambda lab, y_c: (lab != y_c) * (lab != self.config['n_classes']), 'name': 'not class %d' },
        'score_3': {'metrics': metrics_ext, 'filter': lambda lab, y_c: lab == self.config['n_classes'], 'name': 'outliers %d'},
        'name': 'anomaly_exp'}
      self.loop_plot_oneout(an_dict, m_to_process_2d['key'].to_list(), folder_3scores_an, is_vec = True)


  def scatter_plots(self, metrics_ext, set_type):
    # Plot scatters of z with label information, y, y_pred_amax, etc..
    mmanager = self.get_mmanager(metrics_ext)
    m_to_process_s = mmanager[mmanager['to_scatter'] == True]
    list_z  = m_to_process_s[m_to_process_s['score_type'] == 'latent']['key'].to_list()
    list_y  = mmanager[mmanager['score_type'] == 'label']['key'].to_list()
    for mapper in ['umap', 'pca']:
      for z_emb in list_z:
        total_z = pall.obtain_tranform_z(metrics_ext, z_emb, mapper = mapper)
        folder_scatter_z = utils.make_root(self.config[set_type], 'Scatter_%s_mapper_%s_with_labels' % (mapper, z_emb))
        for y_used_key in list_y:
          pall.plot_scatter(y_used_key, total_z, metrics_ext, self.config, folder_scatter_z,
                                                                        self.get_n_classes(set_type))
    # Plot scatter with 2 distributions 
    folder_scatter_shared = utils.make_root(self.config[set_type], 'Scatter_two_dist')
    m_to_process_shared  = m_to_process_s[~m_to_process_s['plot_with'].isnull()]
    for i, row in m_to_process_shared.iterrows():
      plot_with = row['plot_with']
      total_z1, total_z2 = pall.obtain_transform_z_doble(metrics_ext, plot_with['latent'], row['key'])
      for y_used_key in plot_with['latent_labels']:
        pall.plot_scatter_shared(metrics_ext, total_z1, total_z2, y_used_key, plot_with['label'],
                                                       self.config, folder_scatter_shared)
    # Plot scatter of score vs represetation for Accuracy
    list_acc_correct = [key for key in metrics_ext.keys() if 'Accuracy_array' in  key]
    folder_acc_scores_rep = {key_name: utils.make_root(self.config[set_type], \
                   'Scatter_%s_scores_rep' % key_name) for key_name in list_acc_correct}
    # Plot scatter of score vs represetation for anomaly
    folder_an_scores_rep = utils.make_root(self.config[set_type], 'Scatter_An_scores_rep')
    scores_list  = m_to_process_s[(m_to_process_s['len_shape_score'] == 1) & \
                  (m_to_process_s['score_type'] != 'obs_rep')]['key'].to_list()
    rep_list     = m_to_process_s[(m_to_process_s['len_shape_score'] == 1) & \
                  (m_to_process_s['score_type'] == 'obs_rep')]['key'].to_list()
    for score_name in scores_list:
      for rep_name in rep_list:
        for acc_key in list_acc_correct:
          pall.plot_scatter_score_rep_acc(metrics_ext, score_name, rep_name, acc_key,\
                                         self.config, folder_acc_scores_rep[acc_key], 2)
        if self.plot_anomaly(set_type):
          pall.plot_scatter_score_rep_labelad(metrics_ext, score_name, rep_name, self.config,
                                                       folder_an_scores_rep, 2)

  def loop_plot_oneout(self, score_dict, keys, folder_root, is_vec = False):
      #Input should have the following form
      #{'score_1': {'metrics': metrics_ext, 'filter': lambda(x):x,'name': 'some_name with %d' },
      # 'score_2': {'metrics': metrics_ext, 'filter': lambda(x):x, 'name': 'some_name with %d' },
      # 'score_3': {'metrics': metrics_ext, 'filter': lambda(x):x, 'name': 'some_name with %d' },
      # 'name': 'some_name' }

      def filter_data(score_dict, score_name, key, class_idx):
        batch_idx = score_dict[score_name]['filter'](score_dict[score_name]['metrics']['y'], class_idx)
        if is_vec:
          return score_dict[score_name]['metrics'][key][batch_idx, class_idx]
        else:
          return score_dict[score_name]['metrics'][key][batch_idx]

      for key in keys:
        folder_root_class  = '%s/%s_oneclass_out_%s' % (folder_root, score_dict['name'], key)
        if not os.path.isdir(folder_root_class):
            os.mkdir(folder_root_class)
        for i in range(self.config['n_classes']):
          save_path = '%s/histogram_%d.jpg' % (folder_root_class, i)
          score1 = filter_data(score_dict, 'score_1', key, i)
          score2 = filter_data(score_dict, 'score_2', key, i)
          if 'score_3' in score_dict.keys():
            score3 = filter_data(score_dict, 'score_3', key, i)
            plot_hist.plot_3histogram(score1, score_dict['score_1']['name'] % i,
                            score2, score_dict['score_2']['name'] % i,
                            score3, score_dict['score_3']['name'] % i,
                            score_dict['name'],
                            save_path, density = True)
          else:
            plot_hist.plot_2histogram(score1, score_dict['score_1']['name'] % i,
                            score2, score_dict['score_2']['name'] % i,
                            score_dict['name'],
                            save_path, density = True)