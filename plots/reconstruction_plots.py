#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Reconstruction plot file
'''
import os
import numpy as np
import random
import gc
import matplotlib.pyplot as plt
#from astropy.visualization import hist
# Sample function for sample sheets
def reconstruction_sheet(metrics_ext, config, folder_root, is_scaled = False, simplify_plot = False,
                         add_name = '', use_prediction_to_plot = False, score_with_title = None,
                         score_with_title2 = None, is_legend = True, n_classes = 0):


  # loop over total number of sheets
  if config['is_plot_supervised']:
    label_used = metrics_ext['y']
  else:
    if config['use_prediction_gmm_to_plot']:
      label_used = metrics_ext['y_pred_zgmm']

  if config['plot_cov'] and config['which_post_decoder'] == 'AGP':
    for i in range(config['dataset_channel']):
      plot_covariances(metrics_ext['cov11_plot_band_%d' % i], config, label_used,
                                              config['n_classes'], samples_per_class,
                                              is_plot_supervised = is_plot_supervised, name = 'cov11_band_%d' % i)
      plot_covariances(metrics_ext['cov12_plot_band_%d' % i], config, label_used,
                                              config['n_classes'], samples_per_class,
                                              is_plot_supervised = is_plot_supervised, name = 'cov12_band_%d' % i)

  rows = config['n_classes'] if n_classes == 0 else n_classes
  cols = samples_per_class    = config['samples_per_class']
  is_plot_supervised          = config['is_plot_supervised']
  use_prediction_gmm_to_plot  = config['use_prediction_gmm_to_plot']
  plt.clf()
  f, axarr = plt.subplots(rows, samples_per_class, figsize=(3 * samples_per_class, 3 * rows))
  for cc in range(config['dataset_channel']):

      index = 0
      for i in range(rows):
        if use_prediction_gmm_to_plot or is_plot_supervised:
          filter_id    = label_used == i
          arange_index = np.arange(len(label_used)) [filter_id]
          lab_index    = 0
        for j in range(cols):
            if (use_prediction_gmm_to_plot or is_plot_supervised) and lab_index == len(arange_index):
              break
            if use_prediction_gmm_to_plot  or is_plot_supervised:
              index = arange_index[lab_index]
              
            if_label = ((j == 0) or not simplify_plot) and is_legend
            if_leg   = ((i == 0 and j == cols - 1) or not simplify_plot) and is_legend

            plot_rec(axarr[i,j], metrics_ext, index, cc, config, is_scaled = is_scaled,
                          if_leg = if_leg, simplify_plot = simplify_plot)

            if if_leg:
              axarr[i, j].legend(fontsize = 16)
            if simplify_plot  and if_leg:
              axarr[i, j].legend(fontsize = 16)
            if  (if_label and simplify_plot) or score_with_title is not None:
              frow_add_name = ' (outlier)' if metrics_ext['y'][index] == config['n_classes'] else ''
              this_font = 13
              if score_with_title is not None:
                frow_add_name += '\n MI Bound: %4.2f' % score_with_title[index]
                this_font = 15
                if score_with_title2 is not None:
                  frow_add_name += '\n id: %d' % score_with_title2[index]
              axarr[i, j].legend(title = str(metrics_ext['y'][index]) + frow_add_name , 
                     title_fontsize = this_font, fontsize = this_font, loc = 'lower left')
            #axarr[i, j].axis('off')
            index += 1
            if use_prediction_gmm_to_plot or is_plot_supervised:
              lab_index += 1
            if is_scaled:
              axarr[i, j].invert_yaxis()

  plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
  image_filename = '%s/reconstruction_%s' % (folder_root, add_name) + '.jpg'
  if is_scaled:
    image_filename = '%s/reconstruction_scaled_%s'% (folder_root, add_name) + '.jpg' 
  if simplify_plot:
    image_filename = '%s/simp_reconstruction_scaled_%s' % (folder_root, add_name) + '.jpg' 
  f.savefig(image_filename)
  plt.close(f)
  plt.clf()
  import gc;
  gc.collect()
  #pdb.set_trace()
def plot_covariances(cov11_plot, config, label_used, num_classes, samples_per_class, is_plot_supervised = False, name = 'cov11'):
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  for cc in range(config['dataset_channel']):
      plt.clf()
      f, axarr = plt.subplots(num_classes, samples_per_class, figsize=(2 * samples_per_class , 2 * num_classes))

      index = 0
      for i in range(num_classes):
        if config['which_train_fn'] == 'VADE' or is_plot_supervised:
          arange_index = np.arange(len(label_used)) [label_used == i]
          lab_index    = 0
        for j in range(int(samples_per_class) ):
            if (config['which_train_fn'] == 'VADE' or is_plot_supervised) and lab_index == len(arange_index):
              break
            if config['which_train_fn'] == 'VADE' or is_plot_supervised:
              index = arange_index[lab_index]

            im = axarr[i, j].imshow(cov11_plot[cc][index])
            divider = make_axes_locatable(axarr[i, j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax=cax, orientation='vertical')

            index += 1
            if config['which_train_fn'] == 'VADE' or is_plot_supervised:
              lab_index += 1

      image_filename = '%s/covariances_%s_' % (config['a_rec_root'], name) + str(cc) + '.jpg'

      f.savefig(image_filename)
      plt.close(f)
      plt.clf()
      import gc;
      gc.collect()

# Sample function for sample sheets
def single_curve(index, metrics_ext, config, folder_root,
                                        saved_name = '', is_scaled = False, is_obs_data = True):
  # loop over total number of sheets
  plt.clf()
  f, axarr = plt.subplots(1, 1, figsize=(12, 12))
  for cc in range(config['dataset_channel']):  
    plot_rec(axarr, metrics_ext, index, cc, config, is_scaled = is_scaled, is_obs_data = is_obs_data)
  axarr.legend(title = str(metrics_ext['id'][index]), title_fontsize = 12, loc = 'lower left')
  if is_scaled:
    axarr.invert_yaxis()
  image_filename = '%s/%s%s' % (folder_root, saved_name, metrics_ext['id'][index]) + '.jpg'
  f.savefig(image_filename, dpi = 30)
  plt.close(f)
  plt.clf()
  gc.collect

def plot_rec(axarr, metrics_ext, index, cc, config, if_leg = True, simplify_plot = False,
                                                             is_scaled = False, is_obs_data = True):
  svar    = ''
  slarge  = ''
  ss = ''
  if is_scaled:
    ss = 'scaled_'
  if not simplify_plot and config['which_post_decoder'] == 'AGP':
    if config['is_learn_pvar']:
      svar    = ', ' + '$\\sigma^2$' + '=%2.3f' % metrics_ext['pvar'][index][cc]
    if config['is_learn_plarge']:
      slarge  = ', ' + '$l^2$' + '=%2.3f' % metrics_ext['plarge'][index][cc] 

  if not is_obs_data:
    mask_cc = np.arange(len(metrics_ext['time'][index, :, cc]))
  else:
    mask_cc = metrics_ext['mask'][index, :, cc].astype(bool)
  ### Class name ###
  class_name = '' if simplify_plot else  ' ' + str(config['classes_names'][metrics_ext['y'][index]])

  ### Reconstruction Variables ###
  time    = metrics_ext['time'][index, :, cc][mask_cc]
  D_mu    = metrics_ext[ss + 'D_mu'][index, :, cc][mask_cc]
  if config['is_dec_var']:
    D_sigma = metrics_ext[ss + 'D_sigma'][index, :, cc][mask_cc]

  ### Update variables when using  ###
  if 'time_for' in metrics_ext.keys():
    mask_cc_for = metrics_ext['mask_for'][index, :, cc].astype(bool)
    time  = np.concatenate([time, metrics_ext['time_for'][index, :, cc][mask_cc_for]])
    t_arg = time.argsort()
    time  = time[t_arg]
    D_mu  = np.concatenate([D_mu, metrics_ext['D_mu_for'][index, :, cc][mask_cc_for]])[t_arg]
    if 'D_sigma_for' in metrics_ext.keys():
      D_sigma  = np.concatenate([D_sigma, metrics_ext['D_sigma_for'][index, :, cc][mask_cc_for]])[t_arg]
    if 'data_for' in metrics_ext.keys():
      axarr.errorbar(metrics_ext['time_for'][index, :, cc][mask_cc_for], metrics_ext[ss + 'data_for'][index,:, cc][mask_cc_for],
          yerr = metrics_ext[ss + 'data_sigma_for'][index, :, cc][mask_cc_for] if config['noise_data'] else None,
          fmt = '+', color = config['band_colors_obs'][cc],
          alpha = 0.3,  markersize = 4,  label = config['band_legend'][cc] + 'fc obs.' + class_name if if_leg else None)

  ### Plotting data ####
  if is_obs_data:
    axarr.errorbar(metrics_ext['time'][index, :, cc][mask_cc], metrics_ext[ss + 'data'][index,:, cc][mask_cc],
        yerr = metrics_ext[ss + 'data_sigma'][index, :, cc][mask_cc] if config['noise_data'] else None,
        fmt = 'o', color = config['band_colors_obs'][cc],
        alpha = 0.3,  markersize = 4,  label = config['band_legend'][cc] + 'obs.' + class_name if if_leg else None)
  ### Reconstruction ###
  axarr.plot(time, D_mu, '--', color = config['band_colors_mod'][cc],  linewidth = 2.,
      alpha = 0.8, markersize = 16, label = config['band_legend'][cc] + 'pred.' +  svar + slarge if if_leg else None )
  ### Std reconstruction area ###
  if config['is_dec_var']:
    axarr.fill_between(time, D_mu + D_sigma, D_mu - D_sigma, color = config['band_colors_mod'][cc], alpha=0.08)
  ### Induction points ###
  if config['which_post_decoder'] == 'AGP':
    prot_time_used = metrics_ext['time_prot'][index]
    axarr.errorbar(prot_time_used, metrics_ext[ss + 'D_prot_x'][index, :, cc], fmt = 'x', capsize = 2,
        yerr  = np.sqrt(metrics_ext[ss + 'D_prot_x_var'][index, :, cc]) if config['is_dec_var'] else None,
        color = config['band_colors_mod'][cc], alpha = 0.8,  markersize = 5,
        label = config['band_legend'][cc] + 'ind.' if if_leg else None)

