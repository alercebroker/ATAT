'''
Datasets
'''
import os
import os.path
import sys
from PIL import Image
from joblib import load
import numpy as np
import math
import pickle
import torch
import random
from sklearn.model_selection import train_test_split

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
try:
  from torchvision.datasets.utils import verify_str_arg
except:
  print("cuda too old")
import torch.utils.data as data
from torch.utils.data import DataLoader
import h5py

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def mask_errors(err, mask):
    filter_mask = err > np.median(err, -1, keepdims = True) + 5*np.std(err, -1, keepdims = True)
    mask[filter_mask] = 0
    return mask
    
def mask_mags(mag, mask):
    filter_mask = np.abs(mag - np.median(mag, -1, keepdims = True)) > 5*np.std(mag, -1, keepdims = True)
    mask[filter_mask] = 0
    return mask

def mask_all(mag, err, mask):
    mask_err = mask_errors(err, mask)
    mask_mag = mask_mags(mag, mask)
    return mask_err * mask_mag

def obtain_all_metrics(metric_name):
    def obtain_all_times():
        times_to_eval = []
        for i in range(3, 12):
            times_to_eval += [2**i]
        return np.array(times_to_eval)
    if metric_name == 'time':
      return obtain_all_times()

class StochasticMask(object):
    def __init__(self, per_init_time = .2, **kwargs):
        self.per_init_time = per_init_time

    def obtain_valid_mask(self, mask, time_alert):
        random_value  = random.uniform(0, 1)
        max_time      = (time_alert * mask).max()
        init_time     = self.per_init_time * max_time
        eval_time     = init_time + (max_time - init_time) * random_value 
        mask_time     = (time_alert <= eval_time).float()
        return mask * mask_time

    def __call__(self, sample):
        sample['mask'] = self.obtain_valid_mask(sample['mask'], sample['time_alert'])
        return sample

is_torchvision_installed = True
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

def anomaly_reindexation(config):
  which_dataset  = dset_dict[config['dataset']]
  return which_dataset.anomaly_reindexation(config)

# Convenience function to centralize all data loaders
def get_data(dataset, set_type = 'train', **kwargs):

  # Append /FILENAME.hdf5 to root if using hdf5
  which_dataset  = dset_dict[dataset]
  list_transform_normal = []
  train_transform       = transforms.Compose(list_transform_normal)
  target_transform      = transforms.Compose(list_transform_normal)
  dataset_used          = which_dataset(dataset = dataset, set_type = set_type, transform = train_transform,
                            target_transform = target_transform, **kwargs)
  return dataset_used

def get_data_loaders(dataset_used, set_type= 'train', batch_size=64, num_workers=8, shuffle=True,
                     pin_memory=True, drop_last=False, equally_sample = False, **kwargs):

  loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last} # Default, drop last incomplete batch
  if set_type == 'train':
    if not equally_sample:
      loader = DataLoader(dataset_used, batch_size=batch_size,
                            shuffle=shuffle, **loader_kwargs)
    else:
      loader = DataLoader(dataset_used, batch_size=batch_size,
                            sampler = BalancedBatchSampler(dataset_used, torch.tensor(dataset_used.labels)), **loader_kwargs)
  else:
      loader = DataLoader(dataset_used, batch_size=batch_size,
                            shuffle=shuffle, **loader_kwargs)
  return loader

# Convenience dicts
root_dict     = {'ELASTICC': 'final_dataset', 'ELASTICC_STREAM': 'New_All_h5'}
dir_dset_dict = {'ELASTICC': {'train': 'all_ELASTICC.pkl', 'val': 'all_ELASTICC.pkl', 'test': 'all_ELASTICC.pkl'},
                 'ELASTICC_STREAM': {'train': 'all_ELASTICC.pkl', 'val': 'all_ELASTICC.pkl', 'test': 'all_ELASTICC.pkl'} }
nclass_dict   = {'ELASTICC': 20, 'ELASTICC_STREAM': 20}
# Number of classes to put per sample sheet               
classes_per_sheet_dict = {'ELASTICC': 20, 'ELASTICC_STREAM': 20}
seq_dict = {'ELASTICC': 65, 'ELASTICC_STREAM': 100}
T_max    = {'ELASTICC': 1500, 'ELASTICC_STREAM': 1500}
#number of channel per datatset
channel_dict  = {'ELASTICC': 6, 'ELASTICC_STREAM': 6}
folded_dict   = {'ELASTICC': False, 'ELASTICC_STREAM': False}
classes_names = {'ELASTICC': [ 'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT',
                              'KN', 'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c',
                              'II', 'SN-like/Other', 'TDE', 'uLens']}
classes_names.update({'ELASTICC_STREAM': [ 'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT',
                              'KN', 'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c',
                              'II', 'SN-like/Other', 'TDE', 'uLens']})
band_colors_obs  = {'ELASTICC': ['b', 'g', 'r', 'orange', 'brown',  'k']}
band_colors_obs.update({'ELASTICC_STREAM': ['b', 'g', 'r', 'orange', 'brown',  'k']})

band_colors_mod  = {'ELASTICC': ['b', 'g', 'r', 'orange', 'brown',  'k']}
band_colors_mod.update({'ELASTICC_STREAM': ['b', 'g', 'r', 'orange', 'brown',  'k']})

band_legend    = {'ELASTICC': ['b', 'g', 'r', 'orange', 'brown',  'k']}
band_legend.update({'ELASTICC_STREAM': ['b', 'g', 'r', 'orange', 'brown',  'k']})

noise_data   = {'ELASTICC': True}
noise_data.update({'ELASTICC_STREAM': True})

elasticc_feat_names = {'feat_norm': 64  *  [''],
                       'add_feat_norm': 429 * ['']
}

elasticc_feat_values = {'feat_norm': 64  *  [''],
                        'add_feat_norm': 429 * ['']
}

class ELASTICC(data.Dataset):
  def __init__(self, data_root = '', dataset = '', set_type = 'train', transform=None,
                                target_transform=None, in_memory = False,
                                using_metadata = False, using_features = False, seed = 0,
                                eval_metric = None, list_eval_metrics = None,
                                force_online_opt = False, per_init_time = 0.2,
                                use_mask_alert = False, use_small_subset = False,
                                use_time_alert = False, use_time_phot = False,
                                online_opt_tt = False, predict_obj = 'lc',
                                F_max = [], label_per = 0.0, same_partition = False,
                                not_quantile_transformer = False, **kwargs):

    name  = 'training' if set_type == 'train' or set_type == 'train_step' else 'validation'
    partition_used = seed if not same_partition else 0
    name_target = '' if not dataset == 'ELASTICC_STREAM' else '-test'

    file_path = '%s/%s' % (data_root, 'elasticc_final.h5')
    h5_file   = h5py.File(file_path)

    if set_type == 'test' or 'test_real' in set_type:
      self.these_idx  = h5_file.get('test')[:]
    else:
      self.these_idx  = h5_file.get('%s_%s' % (name, partition_used))[:]

    self.using_metadata      = using_metadata
    self.using_features      = using_features
    self.set_type            = set_type
    self.list_eval_time      = list_eval_metrics
    self.per_init_time       = per_init_time
    self.force_online_opt    = force_online_opt
    self.online_opt_tt       = online_opt_tt
    self.use_mask_alert      = use_mask_alert
    self.use_time_alert      = use_time_alert
    self.use_time_phot       = use_time_phot
    self.use_small_subset    = use_small_subset
    self.predict_obj         = predict_obj 
    self.label_per           = label_per
    self.F_len               = len(F_max)
    self.same_partition      = same_partition
    self.not_quantile_transformer = not_quantile_transformer

    self.in_memory = in_memory

    self.data           = h5_file.get('data')
    self.data_var       = h5_file.get('data-var')
    if not use_mask_alert:
      self.mask         = h5_file.get('mask')
    else:
      self.mask         = h5_file.get('mask_alert')
    self.mask_detection = h5_file.get('mask_detection')
    self.time           = h5_file.get('time')
    self.time_alert     = h5_file.get('time_alert')
    self.time_phot      = h5_file.get('time_phot')
    self.target         = h5_file.get('labels')
    self.eval_time      = eval_metric
    if self.use_time_alert:
      self.time = self.time_alert
    if self.use_time_phot:
      self.time = self.time_phot

    self.labels         = torch.from_numpy(self.target[:][self.these_idx])

    #------------------- FEATURES ESTATICAS (METADATA) -------------------#

    if self.using_metadata:
      self.feat_col    = h5_file.get('norm_feat_col')
      self.metadata_qt = load('%s/QT-New%s/md_fold_%s.joblib' % (data_root, name_target, partition_used)) 

      if not_quantile_transformer: 
        self.feat_col  = torch.from_numpy(self.feat_col[:][self.these_idx])
      else:
        print('We are transforming the static features (metadata) using pretrained QT ...')
        self.feat_col  = torch.from_numpy(self.metadata_qt.transform(self.feat_col[:][self.these_idx]))

    #------------------- FEATURES DINAMICAS (CALCULADAS) -------------------#

    if self.using_features:
      print('We are using dynamic features (calculated) ...')
      self.add_feat_col  = h5_file.get('norm_add_feat_col_2048' \
                                  if self.eval_time is None else 'norm_add_feat_col_%s' % self.eval_time)
      self.features_qt   = load('%s/QT-New%s/fe_2048_fold_%s.joblib' % (data_root , name_target, partition_used))

      if not_quantile_transformer: 
        self.add_feat_col  = torch.from_numpy(self.add_feat_col[:][self.these_idx])
      else:
        print('We are transforming the dynamic features (calculated) using pretrained QT ...')
        self.add_feat_col  = torch.from_numpy(self.features_qt.transform(self.add_feat_col[:][self.these_idx]))

    #------------------- FEATURES DINAMICAS (CALCULADAS) TO APPLY MTA IN 8, 128 and 2048 DAYS -------------------#

      self.list_eval_time = np.array([8, 128, 2048])

      self.add_feat_col_list = {'time_%s' % this_eval_time: \
                                  h5_file.get('norm_add_feat_col_{}'.format(this_eval_time))  \
                                  for this_eval_time in self.list_eval_time}
      
      self.add_features_qt = {'time_{}'.format(this_eval_time): \
                                load('{}/QT-New/fe_{}_fold_{}.joblib'.format(data_root, this_eval_time, partition_used))  \
                                for this_eval_time in self.list_eval_time}

      if not_quantile_transformer: 
        self.add_feat_col_list = {'time_%s' % this_eval_time: \
                                    self.add_feat_col_list['time_{}'.format(this_eval_time)][:][self.these_idx]  \
                                    for this_eval_time in self.list_eval_time}
      else:
        print('We are transforming the dynamic features (calculated) for 8, 128 and 2048 days using pretrained QT ...')
        self.add_feat_col_list = {'time_{}'.format(this_eval_time): \
                                    self.add_features_qt['time_{}'.format(this_eval_time)].transform(self.add_feat_col_list['time_{}'.format(this_eval_time)][:][self.these_idx])  \
                                    for this_eval_time in self.list_eval_time}
    

    if self.set_type == 'val_rec' or self.in_memory or self.use_small_subset or \
            'val_real' in self.set_type or (self.set_type == 'train' and label_per != 0):

      ar = np.arange(len(self.labels))
      if self.set_type == 'train' and label_per != 0:
        label_amount_used = int(label_per) if label_per > 0 else label_per
        _, index_vr, _, _ = train_test_split(ar, ar, test_size = label_amount_used,
                                                    random_state = 0, stratify = self.labels)
        index_vr.sort()
      elif self.set_type == 'val_rec' or self.use_small_subset or 'val_real' in self.set_type:
        _, index_vr, _, _ = train_test_split(ar, ar, test_size = 45,
                                                    random_state = 0, stratify = self.labels)
        index_vr.sort()
      else:
        index_vr = ar

      index_vd = self.these_idx[index_vr]
      self.data           = torch.from_numpy(self.data[:][index_vr])
      self.data_var       = torch.from_numpy(self.data_var[:][index_vr])
      self.mask           = torch.from_numpy(self.mask[:][index_vr])
      self.mask_detection = torch.from_numpy(self.mask_detection[:][index_vr])
      self.time           = torch.from_numpy(self.time[:][index_vr])
      self.time_alert     = torch.from_numpy(self.time_alert[:][index_vr])
      self.target         = self.labels = torch.from_numpy(self.target[:][index_vr])

      if self.using_metadata:
        if not_quantile_transformer: 
          self.feat_col  = torch.from_numpy(self.feat_col[:][index_vr])
        else:
          self.feat_col  = torch.from_numpy(self.metadata_qt.transform(self.feat_col[:][index_vr]))
      if self.using_features:
        if not_quantile_transformer:
          self.add_feat_col = torch.from_numpy(self.add_feat_col[:][index_vr])
        else:
          self.add_feat_col = torch.from_numpy(self.features_qt.transform(self.add_feat_col[:][index_vr]))

    #--------------------------------------------------------------------------------------------------
          
    self.max_time = 1500
    self.transform = transform
    self.target_transform = target_transform

  def obtain_valid_mask(self, sample, mask, time_alert, index):
      ##### ESTA MALOOO #####
      mask_time = (time_alert <= self.eval_time).float()
      sample['mask'] = mask * mask_time
      if self.using_features:
          sample['add_tabular_feat'] = \
              torch.from_numpy(self.add_feat_col_list['time_{}'.format(self.eval_time)][index, :])
      return sample

  def sc_augmenation(self, sample, index):
      ##### ESTA MALOOO #####
      mask, time_alert = sample['mask'], sample['time_alert']
      random_value  = random.uniform(0, 1)
      max_time      = (time_alert * mask).max()
      init_time     = self.per_init_time * max_time
      eval_time     = init_time + (max_time - init_time) * random_value 
      mask_time     = (time_alert <= eval_time).float()
      if self.using_features:
          sample['add_tabular_feat'] = \
              torch.from_numpy(self.add_feat_col_list['time_{}'.format(eval_time)][index, :])
      sample['mask'] = mask * mask_time
      return sample

  def three_time_mask(self, sample, index):
      mask, time_alert = sample['mask'], sample['time_alert']
      eval_time     = np.random.choice([8, 128, 2048])
      mask_time     = (time_alert <= eval_time).float()
      if self.using_features:
          sample['add_tabular_feat'] = \
              torch.from_numpy(self.add_feat_col_list['time_{}'.format(eval_time)][index, :])
      sample['mask'] = mask * mask_time
      return sample

  def __getitem__(self, index_it):

    if not self.in_memory and not self.set_type == 'val_rec' and not self.use_small_subset and not ('val_real' in self.set_type):
      index = self.these_idx[index_it]
      data_dict = {'data': torch.from_numpy(self.data[index, :, :]),
                  'data_var': torch.from_numpy(self.data_var[index, :, :]),
                  'time': torch.from_numpy(self.time[index, :, :]),
                  'labels': torch.from_numpy(np.array(self.target[index])),
                  'mask': torch.from_numpy(self.mask[index, :, :]),
                  'mask_detection': torch.from_numpy(self.mask_detection[index, :, :]),
                  'time_alert': torch.from_numpy(self.time_alert[index, :, :]),
                  'index': index}
      
      if self.using_metadata:
        data_qt_md = self.feat_col[index_it, : ]
        data_dict.update({'tabular_feat': data_qt_md})

      if self.using_features:
        data_qt_fe = self.add_feat_col[index_it, : ]
        data_dict.update({'add_tabular_feat': data_qt_fe})

    else:
      data_dict = {'data': self.data[index],
                   'data_var': self.data_var[index],
                   'time': self.time[index],
                   'labels': self.target[index],
                   'mask': self.mask[index],
                   'mask_detection': self.mask_detection[index],
                   'time_alert': self.time_alert[index],
                   'index': index}
      
      if self.using_metadata:
        data_dict.update({'tabular_feat': self.feat_col[index_it]})

      if self.using_features:
        data_dict.update({'add_tabular_feat': self.add_feat_col[index_it]})

    if self.set_type == 'train':
      if self.online_opt_tt:
        data_dict = self.three_time_mask(data_dict, index_it)

    return data_dict


  def __len__(self):
    return len(self.labels)

dset_dict     = {'ELASTICC': ELASTICC, 'ELASTICC_STREAM': ELASTICC}
  