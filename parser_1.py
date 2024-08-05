#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import random
import gc
#import seaborn as sns
import pandas as pd
import importlib

#from astropy.visualization import hist

def prepare_parser(abs_path = None):
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  
  #### Lightning modules ###
  parser.add_argument(
    '--pl_model', type=str, default='AEModel',
    help='pytorch lightning module'
         '(default: %(default)s)')
  #### Lightning modules ###
  parser.add_argument(
    '--pl_pre_model', type=str, default='AEModel',
    help='pytorch lightning module when loading'
         '(default: %(default)s)')
  #### Use linear callbacks ###
  parser.add_argument(
    '--callbacks', nargs='+', default = [],
    help='callbacks used (default: %(default)s)') 
  ### Dataset/Dataloader stuff ###
  parser.add_argument(
    '--dataset', type=str, default='Astro_200',
    help='Which Dataset to train on'
         '(default: %(default)s)')
  parser.add_argument(
    '--is_lc_dataset', action='store_true', default=False,
    help='concatenate test set to the training (default: %(default)s)')
  parser.add_argument(
    '--cat_noise_to_E', action='store_true', default=False,
    help='concatenate the noise to the flux or magnitude (default: %(default)s)')
  parser.add_argument(
    '--concat', action='store_true', default=False,
    help='concatenate test set to the training (default: %(default)s)')
  parser.add_argument(
    '--n_total_bands', type=int, default = 0,
    help='number total bands used for classifications '
         '(default: %(default)s)')
  parser.add_argument(
    '--dropout', type=float, default= 0.0,
    help='dropout? (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers; consider using less for HDF5 '
         '(default: %(default)s)')
  parser.add_argument(
    '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
    help='Pin data into memory through dataloader? (default: %(default)s)') 
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data (strongly recommended)? (default: %(default)s)')
  parser.add_argument(
    '--equally_sample', action='store_true', default=False,
    help='Shuffle the data (strongly recommended)? (default: %(default)s)')
  parser.add_argument(
    '--add_name', type=str, default='',
    help='Put name (default: %(default)s)')
  parser.add_argument(
    '--add_root', type=str, default='',
    help='additional root (default: %(default)s)')
  ##### early stopping ####
  parser.add_argument(
    '--use_early_stopping', action='store_true', default=False,
    help='use early stopping? (default: %(default)s)') 
  ##### training settings ####
  parser.add_argument(
    '--use_post_linear', action='store_true', default=False,
    help='use post linear (default: %(default)s)')
  parser.add_argument(
    '--using_train_step', action='store_true', default=False,
    help='using train step (default: %(default)s)')
  parser.add_argument(
    '--using_val', action='store_true', default=False,
    help='use a validation set? (default: %(default)s)')
  parser.add_argument(
    '--check_every_n_epochs', type=float, default=1.0,
    help='check validation every n epochs? (default: %(default)s)')
  parser.add_argument(
    '--min_epochs', type=int, default=200,
    help='min_epochs? (default: %(default)s)')
  parser.add_argument(
    '--max_epochs', type=int, default=300,
    help='max_epochs? (default: %(default)s)')
  parser.add_argument(
    '--ad_name', type=str, default='',
    help='is anomaly detection (default: %(default)s)')


  ### Model stuff ###
  parser.add_argument(
    '--encoder', type=str, default='./layers/encoder',
    help='Name of the encoder model module (default: %(default)s)')
  parser.add_argument(
    '--decoder', type=str, default='./layers/decoder',
    help='Name of the decoder model module (default: %(default)s)')
  parser.add_argument(
    '--which_encoder', type=str, default='attn',
    help='which encoder is used? (default: %(default)s)')
  parser.add_argument(
    '--which_decoder', type=str, default='attn',
    help='which decoder used?(default: %(default)s)')
  parser.add_argument(
    '--which_pre_encoder', type=str, default='',
    help='which pre encoder is used? (default: %(default)s)')
  parser.add_argument(
    '--which_post_decoder', type=str, default='',
    help='which post decoder used?(default: %(default)s)')
  parser.add_argument(
    '--loss_type', type=str, default='hinge_dis',
    help='Type of loss, hinge_dis or dc gan (default: %(default)s)')
  parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--cluster_per_class', type=int, default=1,
    help='Cluster_per_class: %(default)s)')
  parser.add_argument(
    '--norm_style', type=str, default='in',
    help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
         'ln [layernorm], gn [groupnorm] (default: %(default)s)')
  parser.add_argument(
    '--D_param', type=str, default='SN',
    help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
          ' or None (default: %(default)s)')
  parser.add_argument(
    '--E_param', type=str, default='SN',
    help='Parameterization style to use for E, spectral norm (SN) or SVD (SVD)'
         ' or None (default: %(default)s)')    
  parser.add_argument(
    '--D_nl', type=str, default='relu',
    help='Activation function for D (default: %(default)s)')
  parser.add_argument(
    '--E_nl', type=str, default='relu',
    help='Activation function for E (default: %(default)s)')



  ### Autoencoding classifier ###
  parser.add_argument(
    '--embedding_supervised', type=float, default=0.0,
    help='Do we add a cross-entropy term?(default: %(default)s)')
  parser.add_argument(
    '--num_neural_classifier', type=int, default = 0,
    help='Do the classifier is computed using a neural network?(default: %(default)s)')

  ### VADE ###
  parser.add_argument(
    '--l_p_z', type= float, default=1.0,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--l_q_z_x', type= float, default=1.0,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--l_p_z_c', type= float, default=1.0,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--l_p_c', type= float, default=1.0,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--l_q_y_z', type= float, default=1.0,
    help='Noise dimensionality: %(default)s)')

  ### Anomaly detection stuff ###
  parser.add_argument(
    '--is_anomaly_detection', action='store_true', default=False,
    help='Is anomaly_detection?(default: %(default)s)')
  parser.add_argument(
    '--ad_class_name', type=str, default='',
    help='Name of the class for anomaly detection (default: %(default)s)')
  parser.add_argument(
    '--dim_f', type=int, default = 0,
    help='dimensionality of extra features: %(default)s)')
  parser.add_argument(
    '--use_extra_feat', action='store_true', default=False,
    help='Is anomaly_detection?(default: %(default)s)')  
  parser.add_argument(
    '--is_standarize_feat', action='store_true', default=False,
    help='standarize the extra features?(default: %(default)s)')
  parser.add_argument(
    '--extensive_plot', action='store_true', default=False,
    help='plot extensively?(default: %(default)s)')   

  ### amortized gaussian process stuff ###
  parser.add_argument(
    '--filter_data', action='store_true', default=False,
    help='filter data?(default: %(default)s)')
  parser.add_argument(
    '--full_cov_loss', action='store_true', default=False,
    help='full cov loss?(default: %(default)s)')
  parser.add_argument(
    '--is_dec_var', action='store_true', default=False,
    help='is dec var?(default: %(default)s)')
  parser.add_argument(
    '--act_decvar', type=str, default='exp',
    help='Activation function for the variance of the prototypes'
         '(default: %(default)s)')
  parser.add_argument(
    '--bias_decvar', type=float, default= 1e-6,
    help='bias in the induction points variance'
         ' (default: %(default)s)')
  parser.add_argument(
    '--samples_per_class', type=int, default=6,
    help='Random seed to use; affects both initialization and '
         ' dataloading. (default: %(default)s)')

  ### Training stuff ####
  parser.add_argument(
    '--notTSNE', action='store_true', default=False,
    help='plot TSNE?(default: %(default)s)')
  parser.add_argument(
    '--is_double_loss', type=float, default= 0.0,
    help='use double loss?(default: %(default)s)')
  parser.add_argument(
    '--detach_double', action='store_true', default=False,
    help='detach double?(default: %(default)s)')
  parser.add_argument(
    '--double_loss_cte', type=float, default= 0.5,
    help='constant in '
         '(default: %(default)s)')
  parser.add_argument(
    '--is_ccbn', action='store_true', default=False,
    help='not use conditional batch norm?(default: %(default)s)')
  # Posterior gaussian process
  parser.add_argument(
    '--is_sharpen', type=float, default= 0.0,
    help='is sharpen? (default: %(default)s)')

  ### Model init stuff ###
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use; affects both initialization and '
         ' dataloading. (default: %(default)s)')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Init style to use for D (default: %(default)s)')
  parser.add_argument(
    '--E_init', type=str, default='ortho',
    help='Init style to use for D(default: %(default)s)')
  parser.add_argument(
    '--GMM_init', type=str, default='ortho',
    help='Init style to use for GMM Prior(default: %(default)s)')
  parser.add_argument(
    '--skip_init', action='store_true', default=False,
    help='Skip initialization, ideal for testing when ortho init was used '
          '(default: %(default)s)')


  ### Optimizer stuff ###
  parser.add_argument(
    '--optimizer_type', type=str, default='adam',
    help='optimizer type (default: %(default)s)')
  parser.add_argument(
    '--weight_decay', type=float, default=5e-4,
    help='weight decay of the optimizer (default: %(default)s)')
  parser.add_argument(
    '--D_lr', type=float, default=5e-5,
    help='Learning rate to use for Decoder (default: %(default)s)')
  parser.add_argument(
    '--E_lr', type=float, default=2e-4,
    help='Learning rate to use for Encoder (default: %(default)s)')
  parser.add_argument(
    '--P_lr', type=float, default=2e-4,
    help='Learning rate to use for Prior (default: %(default)s)')
  parser.add_argument(
    '--D_B1', type=float, default=0.0,
    help='Beta1 to use for Decoder (default: %(default)s)')
  parser.add_argument(
    '--E_B1', type=float, default=0.0,
    help='Beta1 to use for Encoder (default: %(default)s)')
  parser.add_argument(
    '--D_B2', type=float, default=0.999,
    help='Beta2 to use for Decoder (default: %(default)s)')
  parser.add_argument(
    '--E_B2', type=float, default=0.999,
    help='Beta2 to use for Encoder (default: %(default)s)')
    
  ### Batch size, parallel, and precision stuff ###
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--num_epochs', type=int, default=100,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
        
  ### Bookkeping stuff ###  
  parser.add_argument(
    '--save_every', type=int, default=2000,
    help='Save every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='How many copies to save (default: %(default)s)')
  parser.add_argument(
    '--num_best_copies', type=int, default=2,
    help='How many previous best checkpoints to save (default: %(default)s)')
  parser.add_argument(
    '--which_best', type=str, default='IS',
    help='Which metric to use to determine when to save new "best"'
         'checkpoints, one of IS or FID (default: %(default)s)')
  parser.add_argument(
    '--test_every', type=int, default=5000,
    help='Test every X iterations (default: %(default)s)')
  parser.add_argument(
    '--base_root', type=str, default='',
    help='Default location to store all weights, samples, data, and logs '
           ' (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--weights_root', type=str, default='weights',
    help='Default location to store weights (default: %(default)s)')
  parser.add_argument(
    '--logs_root', type=str, default='logs',
    help='Default location to store logs (default: %(default)s)')
  parser.add_argument(
    '--samples_root', type=str, default='samples',
    help='Default location to store samples (default: %(default)s)')  
  parser.add_argument(
    '--name_suffix', type=str, default='',
    help='Suffix for experiment name for loading weights for sampling '
         '(consider "best0") (default: %(default)s)')
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Optionally override the automatic experiment naming with this arg. '
         '(default: %(default)s)')
  parser.add_argument(
    '--config_from_name', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)')
  parser.add_argument(
    '--is_plot_supervised', action='store_true', default=False,
    help='This is to include supervision when plotting'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_prediction_gmm_to_plot', action='store_true', default=False,
    help='Use prediction of GMM to plot'
         '(default: %(default)s)')


  ### Plot Stuff ###
  parser.add_argument(
    '--umap_plot', action='store_true', default=False,
    help='plot umap?'
         '(default: %(default)s)')
  parser.add_argument(
    '--not_umap_train_plot', action='store_true', default=False,
    help='not plot umap obtained by trained?'
         '(default: %(default)s)')
  parser.add_argument(
    '--total_sample_GMM', type=int, default=10000,
    help='number of sample for umap?'
         '(default: %(default)s)')
  parser.add_argument(
    '--ellipses_plot', type=int, default= 0,
    help='number of sample for umap?'
         '(default: %(default)s)')

  ### EMA Stuff ###
  parser.add_argument(
    '--ema', action='store_true', default=False,
    help='Keep an ema of G''s weights? (default: %(default)s)')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=False,
    help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
  parser.add_argument(
    '--ema_start', type=int, default=0,
    help='When to start updating the EMA weights (default: %(default)s)')
  
  ### Numerical precision and SV stuff ### 
  parser.add_argument(
    '--adam_eps', type=float, default=1e-8,
    help='epsilon value to use for Adam (default: %(default)s)')
  parser.add_argument(
    '--BN_eps', type=float, default=1e-5,
    help='epsilon value to use for BatchNorm (default: %(default)s)')
  parser.add_argument(
    '--SN_eps', type=float, default=1e-8,
    help='epsilon value to use for Spectral Norm(default: %(default)s)')
  parser.add_argument(
    '--num_D_SVs', type=int, default=1,
    help='Number of SVs to track in D (default: %(default)s)')
  parser.add_argument(
    '--num_E_SVs', type=int, default=1,
    help='Number of SVs to track in E (default: %(default)s)')
  parser.add_argument(
    '--num_D_SV_itrs', type=int, default=1,
    help='Number of SV itrs in D (default: %(default)s)')
  parser.add_argument(
    '--num_E_SV_itrs', type=int, default=1,
    help='Number of SV itrs in E (default: %(default)s)')
  
  parser.add_argument(
    '--toggle_grads', action='store_true', default=True,
    help='Toggle E and D''s "requires_grad" settings when not training them? '
         ' (default: %(default)s)')
  ### Prior stuff ###
  parser.add_argument(
    '--prior_type', type=str, default='default',
    help='Type of prior, default, aux or GMM (default: %(default)s)')
  parser.add_argument(
    '--sigma', type=float, default=0.5,
    help='Sigma threshold  (default: %(default)s)')


  ### Which train function ###
  parser.add_argument(
    '--which_train_fn', type=str, default='AE',
    help='How2trainyourbois (default: %(default)s)') 
  parser.add_argument(
    '--reg_type', type=str, default='',
    help='regularization type (default: %(default)s)')  
  parser.add_argument(
    '--beta_1', type=float, default = 0.0,
    help='beta 1? (default: %(default)s)')
  parser.add_argument(
    '--beta_2', type=float, default = 0.0,
    help='beta 2? (default: %(default)s)')
  
  ### Resume training stuff
  parser.add_argument(
    '--load_weights', type=str, default='',
    help='Suffix for which weights to load (e.g. best0, copy0) '
         '(default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  
  ### Log stuff ###
  parser.add_argument(
    '--logstyle', type=str, default='%3.3e',
    help='What style to use when logging training metrics?'
         'One of: %#.#f/ %#.#e (float/exp, text),'
         'pickle (python pickle),'
         'npz (numpy zip),'
         'mat (MATLAB .mat file) (default: %(default)s)')
  parser.add_argument(
    '--log_D_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--log_E_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in E? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sv_log_interval', type=int, default=10,
    help='Iteration interval for logging singular values '
         ' (default: %(default)s)') 
  parser.add_argument(
    '--attn_layers', type=int, default = 1,
    help='Number of attentions layers'
         '(default: %(default)s)')
  parser.add_argument(
    '--emb_to_classifier', type=str, default = 'avg',   
    help='what embedding to use'
         '(default: %(default)s)')
  parser.add_argument(
    '--using_tabular_feat', action='store_true', default=False,
    help='using tabular features?')
  parser.add_argument(
    '--in_memory', action='store_true', default=False,
    help='load in memory h5?')
  parser.add_argument(
    '--which_tabular_feat', type=str, default = 'feat_alerts',   
    help='which tabular feat to use'
         '(default: %(default)s)') 
  parser.add_argument(
    '--tab_num_heads', type=int, default = 4,
    help='Number of heads'
         '(default: %(default)s)')
  parser.add_argument(
    '--tab_head_dim', type=int, default = 32,
    help='Number of head dim'
         '(default: %(default)s)')
  parser.add_argument(
    '--tab_detach', action='store_true', default=False,
    help='detach classifier?'
         '(default: %(default)s)')
  parser.add_argument(
    '--dropout_first_mha', type=float, default=0.0,
    help='Sigma threshold  (default: %(default)s)')
  parser.add_argument(
    '--dropout_second_mha', type=float, default=0.0,
    help='Sigma threshold  (default: %(default)s)')
  parser.add_argument(
    '--drop_mask_second_mha', action='store_true', default=False,
    help='using normalizing features?'
         '(default: %(default)s)')
  parser.add_argument(
    '--tab_output_dim', type=int, default = 0,
    help='Number of output dim'
         '(default: %(default)s)')
  parser.add_argument(
    '--classify_source', action='store_true', default=False,
    help='using normalizing features?'
         '(default: %(default)s)')
  parser.add_argument(
    '--combine_lc_tab', action='store_true', default=False,
    help='combine lightcurve with tabular data?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_feat_modulator', action='store_true', default=False,
    help='use feat modulator?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_add_feat', action='store_true', default=False,
    help='use additinal features?'
         '(default: %(default)s)')
  parser.add_argument(
    '--using_val_rec', action='store_true', default=False,
    help='use validation for reconstruction?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_causal_transformer', action='store_true', default=False,
    help='use causal transformer?'
         '(default: %(default)s)')
  parser.add_argument(
    '--train_multiple_times', action='store_true', default=False,
    help='eval in multiple days?'
         '(default: %(default)s)')
  parser.add_argument(
    '--eval_multiple_metrics', type=str, default = '', # 'time'   
    help='eval multiple metrics?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_base_loss', action='store_true', default=False,
    help='use base loss?'
         '(default: %(default)s)')
  parser.add_argument(
    '--predict_obj', type=str, default = 'all',   
    help='what embedding to use'
         '(default: %(default)s)')
  parser.add_argument(
    '--per_init_time', type=float, default=0.2,
    help='percentage init time  (default: %(default)s)')
  parser.add_argument(
    '--per_final_time', type=float, default=0.2,
    help='percentage final time  (default: %(default)s)')
  parser.add_argument(
    '--per_base_maintain', type=float, default=0.8,
    help='percentage final time  (default: %(default)s)')
  parser.add_argument(
    '--per_shared_pred_base', type=float, default=0.0,
    help='percentage final time  (default: %(default)s)')
  parser.add_argument(
    '--pre_aug_type', type=str, default = '',   
    help='what embedding to use'
         '(default: %(default)s)')
  parser.add_argument(
    '--force_online_opt', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--online_opt_tt', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_mask_alert', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_small_subset', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_time_alert', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_time_phot', action='store_true', default=False,
    help='force online optimizations?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_detection_token', action='store_true', default=False,
    help='use detection token?'
         '(default: %(default)s)')
  parser.add_argument(
    '--using_metadata', action='store_true', default=False,
    help='using metadata?'
         '(default: %(default)s)')
  parser.add_argument(
    '--using_features', action='store_true', default=False,
    help='using features?'
         '(default: %(default)s)')
  parser.add_argument(
    '--eval_again', type=str, default = '',   
    help='evaluate again'
         '(default: %(default)s)')
  parser.add_argument(
    '--head_dim', type=int, default = 128,
    help='Number of head dimensions'
         '(default: %(default)s)')
  parser.add_argument(
    '--num_heads', type=int, default = 4,
    help='Number of heads'
         '(default: %(default)s)')
  parser.add_argument(
    '--M', type = int, default = 16,
    help='Number of component of fourier modulator (time)?(default: %(default)s)')
  parser.add_argument(
    '--Feat_M', type = int, default = 16,
    help='Number of component of fourier modulator (feat)?(default: %(default)s)')
  parser.add_argument(
    '--which_SSL_lc', type=str, default = None,   
    help='which_SSL_lc'
         '(default: %(default)s)') 
  parser.add_argument(
    '--which_SSL_tab', type=str, default = None,   
    help='which_SSL_tab'
         '(default: %(default)s)') 
  parser.add_argument(
    '--overlap_pred_base', type=float, default=0.0,
    help='overlap between mask for base and for pred (default: %(default)s)')
  parser.add_argument(
    '--drop_per_SSL', type=float, default=0.2,
    help='drop_per_SSL (default: %(default)s)')
  parser.add_argument(
    '--which_SSL_tab_input', type = str, default = None,   
    help='which_SSL_tab_input'
         '(default: %(default)s)')
  parser.add_argument(
    '--embed_dim_mlp', type=int, default=32,
    help='Number of hidden units of MLP'
         '(default: %(default)s)')
  parser.add_argument(
    '--num_mlp_blocks', type=int, default=4,
    help='Number of hidden units of MLP'
         '(default: %(default)s)')
  parser.add_argument(
    '--label_per', type=float, default=0.0,
    help='Number of hidden units of MLP'
         '(default: %(default)s)')
  parser.add_argument(
    '--same_partition', action='store_true', default=False,
    help='same partition?'
         '(default: %(default)s)')
  parser.add_argument(
    '--use_common_positional_encoding', action='store_true', default=False,
    help='not using time modulation'
         '(default: %(default)s)')
  parser.add_argument(
    '--not_quantile_transformer', action='store_true', default=False,
    help='not using quantile transformer'
         '(default: %(default)s)')
  parser.add_argument(
    '--not_tabular_transformer', action='store_true', default=False,
    help='not using tabular transformer'
         '(default: %(default)s)')
  parser.add_argument(
    '--reset_tab_transformer', action='store_true', default=False,
    help='reset_tab_transformer'
         '(default: %(default)s)')
  parser.add_argument(
    '--reset_lc_transformer', action='store_true', default=False,
    help='reset_lc_transformer'
         '(default: %(default)s)')
  if abs_path is None:
    layer_dir = os.listdir('./layers')
  else:
    layer_dir = os.listdir('%s/layers' % abs_path)
  for this_dir in layer_dir:
    try:
      file_py = importlib.import_module('layers.' + this_dir[:-3])
      print("A: " + this_dir)
      parser = file_py.add_sample_parser(parser)
      print("B: " + this_dir)
    except:
      print("C1: " + this_dir)
      continue
      print("C2: " + this_dir)
  return parser