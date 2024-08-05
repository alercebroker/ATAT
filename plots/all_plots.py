import os
import numpy as np
from . import plot_hist as hplot
from . import plot_scatter as splot
from . import reconstruction_plots as recplot

import umap
from sklearn.decomposition import PCA
###################### Plot histograms ######################

def plot_hist_1d(key, metrics_ext, config, folder_root):
  save_path = '%s/histrogram_%s.jpg' % (folder_root, key) 
  hplot.plot_histogram(metrics_ext[key], key, key, save_path)

def plot_hist_1d_acc(key, accuracy_name, metrics_ext, config, folder_root):
    save_path = '%s/histrogram_classified_correctly_%s.jpg' % (folder_root, key)
    hplot.plot_2histogram(metrics_ext[key][metrics_ext[accuracy_name] == 1], 'Data classified correctly',
                          metrics_ext[key][metrics_ext[accuracy_name] == 0], 'Data classified incorrectly',
                          key, save_path)

def plot_hist_1d_labelad(key, metrics_ext, config, folder_root):
    save_path  = '%s/histrogram_in_out_%s.jpg' % (folder_root, key)
    hplot.plot_2histogram(metrics_ext[key][metrics_ext['labels_ad'] == 1], 'inlier',
                          metrics_ext[key][metrics_ext['labels_ad'] == 0], 'outlier',
                          key, save_path)

    save_path = '%s/histogram_%s_nonorm_in_and_out_centered_in.jpg' % (folder_root, key)
    hplot.plot_2histogram(metrics_ext[key][metrics_ext['labels_ad'] == 1], 'inlier',
                          metrics_ext[key][metrics_ext['labels_ad'] == 0], 'outlier',
                          key, save_path, fixed = True, centered = 0)


################### Plot scatters ############################

def obtain_tranform_z(metrics_ext, key, mapper = 'umap'):
  if mapper == 'umap':
    return umap.UMAP().fit_transform(metrics_ext[key]) \
                          if metrics_ext[key].shape[-1] > 2 else metrics_ext[key]
  elif mapper == 'pca':
    return PCA(n_components=2).fit_transform(metrics_ext[key]) \
                          if metrics_ext[key].shape[-1] > 2 else metrics_ext[key]

def plot_scatter(key, total_z_umap, metrics_ext, config, folder_root, n_colors):
  ### Histograms entropy of latent variables ###
  save_path = '%s/UMAP_embedding_%s.jpg' % (folder_root, key)
  splot.plot_z(total_z_umap, metrics_ext[key], config, save_path, n_colors)


def plot_scatter_score_rep_acc(metrics_ext, key_score, key_rep, accuracy_name, config, folder_root, n_colors):
  save_path  = '%s/scatter_rep_%s_vs_%s_.jpg' % (folder_root, key_score, key_rep)
  splot.scatter_color(metrics_ext[key_rep], metrics_ext[key_score], key_rep, key_score, metrics_ext[accuracy_name],
                             save_path, ['Classified incorrectly', 'Classified correctly'], n_colors )
             
def plot_scatter_score_rep_labelad(metrics_ext, key_score, key_rep, config, folder_root, n_colors):
  save_path  = '%s/scatter_Anomaly_rep_%s_vs_%s.jpg' % (folder_root, key_score, key_score)
  splot.scatter_color(metrics_ext[key_rep], metrics_ext[key_score], key_rep, key_score,
             metrics_ext['labels_ad'], save_path, ['Inlier data', 'Outlier data'], n_colors)


################## Reconstruction plot ########################
# Filter indexes
def filter_metrics(metrics, index):
  new_metrics = {}
  for key in metrics.keys():
    aux_var = metrics[key]
    if aux_var is not None and type(aux_var) != bool and len(aux_var) > 1:
      new_metrics[key] = metrics[key][index]
  return new_metrics

def plot_rec(metrics_ext, config, folder_root, n_classes):
  config_aux = config.copy()

  ### Reconstruction plot ####
  recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = False, n_classes = n_classes)
  if config['is_lc_dataset']:
    recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = True, n_classes = n_classes)
    recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = True, n_classes = n_classes,
                                  simplify_plot = True)
    recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = True, n_classes = n_classes,
                                  simplify_plot = True, is_legend = False, add_name = 'NoLeg')


def plot_rec_ascore(key_an_score, metrics_ext, config, folder_root):
  config_aux               = config.copy()
  config_aux['is_plot_supervised'] = False
  config_aux['samples_per_class'] = 4
  an_score                 = metrics_ext['an_score_' + key_an_score]
  an_score_sort_index      = an_score.argsort()
  num_rows, num_cols       = 4, 4
  num_chosen               = (num_rows * num_cols)//2
  an_score_sort_index_min  = an_score_sort_index[ :num_chosen]
  an_score_sort_index_max  = an_score_sort_index[-num_chosen:]
  an_score_sort_index_cat  = np.concatenate([an_score_sort_index_min, an_score_sort_index_max])

  metrics_ext_ordered      = filter_metrics(metrics_ext,  an_score_sort_index_cat)
  an_score_min_max_scores  = an_score[an_score_sort_index_cat]

  scaled_plots = True if config['is_lc_dataset'] else False
  recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = scaled_plots,
                      simplify_plot = True, add_name = '%s_' % key_an_score, n_classes = num_rows,
                      score_with_title = an_score_min_max_scores, score_with_title2 = metrics_ext['id'][an_score_sort_index_cat])
  recplot.reconstruction_sheet(metrics_ext, config_aux, folder_root, is_scaled = scaled_plots, n_classes = num_rows,
                      simplify_plot = True, add_name = 'NoLeg_%s_' % key_an_score, is_legend = False,
                      score_with_title = an_score_min_max_scores, score_with_title2 = None)