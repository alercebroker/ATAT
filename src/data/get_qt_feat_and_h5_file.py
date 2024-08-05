from sklearn.preprocessing import QuantileTransformer
from joblib import dump
import pandas as pd
import numpy as np
import h5py
import json 
import os

import warnings
warnings.filterwarnings("ignore")

feat_dict = {
    "hostgal2_ellipticity": "HOSTGAL2_ELLIPTICITY",
    "hostgal2_mag_Y": "HOSTGAL2_MAG_Y",
    "hostgal2_mag_g": "HOSTGAL2_MAG_g",
    "hostgal2_mag_i": "HOSTGAL2_MAG_i",
    "hostgal2_mag_r": "HOSTGAL2_MAG_r",
    "hostgal2_mag_u": "HOSTGAL2_MAG_u",
    "hostgal2_mag_z": "HOSTGAL2_MAG_z",
    "hostgal2_magerr_Y": "HOSTGAL2_MAGERR_Y",
    "hostgal2_magerr_g": "HOSTGAL2_MAGERR_g",
    "hostgal2_magerr_i": "HOSTGAL2_MAGERR_i",
    "hostgal2_magerr_r": "HOSTGAL2_MAGERR_r",
    "hostgal2_magerr_u": "HOSTGAL2_MAGERR_u",
    "hostgal2_magerr_z": "HOSTGAL2_MAGERR_z",
    "hostgal2_snsep": "HOSTGAL2_SNSEP",
    "hostgal2_sqradius": "HOSTGAL2_SQRADIUS",
    "hostgal2_zphot": "HOSTGAL2_PHOTOZ",
    "hostgal2_zphot_err": "HOSTGAL2_PHOTOZ_ERR",
    'hostgal2_zphot_q000': 'HOSTGAL2_ZPHOT_Q000',
    "hostgal2_zphot_q010": "HOSTGAL2_ZPHOT_Q010",
    "hostgal2_zphot_q020": "HOSTGAL2_ZPHOT_Q020",
    "hostgal2_zphot_q030": "HOSTGAL2_ZPHOT_Q030",
    "hostgal2_zphot_q040": "HOSTGAL2_ZPHOT_Q040",
    "hostgal2_zphot_q050": "HOSTGAL2_ZPHOT_Q050",
    "hostgal2_zphot_q060": "HOSTGAL2_ZPHOT_Q060",
    "hostgal2_zphot_q070": "HOSTGAL2_ZPHOT_Q070",
    "hostgal2_zphot_q080": "HOSTGAL2_ZPHOT_Q080",
    "hostgal2_zphot_q090": "HOSTGAL2_ZPHOT_Q090",
    "hostgal2_zphot_q100": "HOSTGAL2_ZPHOT_Q100",
    'hostgal2_zspec': 'HOSTGAL2_SPECZ',
    'hostgal2_zspec_err': 'HOSTGAL2_SPECZ_ERR',
    "hostgal_ellipticity": "HOSTGAL_ELLIPTICITY",
    "hostgal_mag_Y": "HOSTGAL_MAG_Y",
    "hostgal_mag_g": "HOSTGAL_MAG_g",
    "hostgal_mag_i": "HOSTGAL_MAG_i",
    "hostgal_mag_r": "HOSTGAL_MAG_r",
    "hostgal_mag_u": "HOSTGAL_MAG_u",
    "hostgal_mag_z": "HOSTGAL_MAG_z",
    "hostgal_magerr_Y": "HOSTGAL_MAGERR_Y",
    "hostgal_magerr_g": "HOSTGAL_MAGERR_g",
    "hostgal_magerr_i": "HOSTGAL_MAGERR_i",
    "hostgal_magerr_r": "HOSTGAL_MAGERR_r",
    "hostgal_magerr_u": "HOSTGAL_MAGERR_u",
    "hostgal_magerr_z": "HOSTGAL_MAGERR_z",
    "hostgal_snsep": "HOSTGAL_SNSEP",
    "hostgal_sqradius": "HOSTGAL_SQRADIUS",
    "hostgal_zphot": "HOSTGAL_PHOTOZ",
    "hostgal_zphot_err": "HOSTGAL_PHOTOZ_ERR",
    'hostgal_zphot_q000': 'HOSTGAL_ZPHOT_Q000',
    "hostgal_zphot_q010": "HOSTGAL_ZPHOT_Q010",
    "hostgal_zphot_q020": "HOSTGAL_ZPHOT_Q020",
    "hostgal_zphot_q030": "HOSTGAL_ZPHOT_Q030",
    "hostgal_zphot_q040": "HOSTGAL_ZPHOT_Q040",
    "hostgal_zphot_q050": "HOSTGAL_ZPHOT_Q050",
    "hostgal_zphot_q060": "HOSTGAL_ZPHOT_Q060",
    "hostgal_zphot_q070": "HOSTGAL_ZPHOT_Q070",
    "hostgal_zphot_q080": "HOSTGAL_ZPHOT_Q080",
    "hostgal_zphot_q090": "HOSTGAL_ZPHOT_Q090",
    "hostgal_zphot_q100": "HOSTGAL_ZPHOT_Q100",
    'hostgal_zspec': 'HOSTGAL_SPECZ',
    'hostgal_zspec_err': 'HOSTGAL_SPECZ_ERR',
    "mwebv": "MWEBV",
    "mwebv_err": "MWEBV_ERR",
    "z_final": "REDSHIFT_HELIO",
    "z_final_err": "REDSHIFT_HELIO_ERR",
}

feat_list = list(feat_dict.values())


def joint_classes(target_dir):
    print('- joining each lc file into one ...')
    index = 0
    list_dfs = []
    for fine_class in os.listdir(target_dir):
        df       = pd.read_pickle(os.path.join(target_dir, fine_class))
        list_dfs.append(df)
        print('-', fine_class, len(df))
        index   += 1

    return pd.concat(list_dfs)


def open_partitions(path_blind_partitions, path_partitions, path_save_final_data):
    total_partitions = pd.read_parquet(path_blind_partitions)
    total_partitions_not_blind = pd.read_parquet(path_partitions)

    mapping_to_int = {key: idx - 1  for idx, key in enumerate(total_partitions.label.unique()) }
    print('- ', total_partitions.label.unique())
    print('- ', mapping_to_int)

    with open("./{}/dict_classes.json".format(path_save_final_data), "w") as outfile:
        json.dump(mapping_to_int, outfile)

    def apply_mapping(label_str):
        return mapping_to_int[label_str]

    total_partitions_not_blind['label_int'] = total_partitions_not_blind.apply(lambda x: apply_mapping(x['label']), axis = 1)

    return total_partitions_not_blind


def ordered_partitions(df, total_partitions, fold):
    this_partition = total_partitions[(total_partitions['partition'] == 'training_%d' %  fold) | \
                                      (total_partitions['partition'] == 'validation_%d' % fold) | \
                                      (total_partitions['partition'] == 'test')]
    this_partition       = this_partition.set_index('SNID')
    this_partition_final = this_partition.filter(items = df.index, axis = 0)
    this_partition_final['unique_id'] = np.arange(len(this_partition_final))
    return this_partition_final


def create_dataset(all_partitions, all_data, md_norm_list, num_folds, path_save_final_data):
    with h5py.File('{}/elasticc_final.h5'.format(path_save_final_data), 'w') as hf:
        hf.create_dataset("data",     data = np.stack(all_data['data'].to_list(), 0))
        hf.create_dataset("data-var",  data = np.stack(all_data['data_var'].to_list(), 0))
        hf.create_dataset("mask",  data = np.stack(all_data['mask'].to_list(), 0))
        hf.create_dataset("time",  data = np.stack(all_data['time'].to_list(), 0))
        hf.create_dataset("labels",  data = np.stack(all_data['label_int'].to_list(), 0))
        hf.create_dataset("mask_alert",  data = np.stack(all_data['mask_alert'].to_list(), 0))
        hf.create_dataset("time_alert",  data = np.stack(all_data['time_alert'].to_list(), 0))
        hf.create_dataset("time_phot",  data = np.stack(all_data['time_phot'].to_list(), 0))
        hf.create_dataset("mask_detection",  data = np.stack(all_data['mask_detection'].to_list(), 0))
        hf.create_dataset("norm_feat_col",   data = all_data[md_norm_list].to_numpy())  
        hf.create_dataset("SNID", data = np.array(all_data.index))

        for fold in range(num_folds): 
            print("- fold, ", fold)
            aux_pd        = all_partitions['fold_%s' % fold]
            hf.create_dataset("training_%d" % fold,
                data = aux_pd[aux_pd['partition'] == 'training_%d' % fold]['unique_id'].to_numpy())
            hf.create_dataset("validation_%d" % fold,
                data = aux_pd[aux_pd['partition'] == 'validation_%d' % fold]['unique_id'].to_numpy())
            
        hf.create_dataset("test", 
            data = aux_pd[aux_pd['partition'] == 'test']['unique_id'].to_numpy())
        hf.close()


def create_qt(features, name, path_save):
    qt = QuantileTransformer(n_quantiles=10000,
                             random_state = 0,
                             output_distribution = 'normal')
    qt.fit(features)
    os.makedirs('./{}/QT-New'.format(path_save), exist_ok=True)
    dump(qt, './{}/QT-New/%s.joblib'.format(path_save) % name)


def get_qt_metadata(pd_final, total_partitions_not_blind, num_folds, path_save_final_data):
    all_partitions = {}
    for fold in range(num_folds):
        all_partitions['fold_%s' % fold] = ordered_partitions(pd_final, 
                                                              total_partitions_not_blind, 
                                                              fold)

    pd_final_v2 = pd.merge(pd_final, all_partitions['fold_0'][['label_int']], 
                           left_index=True, right_index=True, how = 'inner')
    
    norm_name_used = 'norm'
    md_norm_list   = ['%s_%s' % (norm_name_used, feat) for feat in feat_list]

    for col in feat_list:
        aux      = pd_final_v2[col].to_numpy().copy()
        new_name = '%s_%s' % (norm_name_used, col)
        print("- %s is Nan: %s" % (col, np.isnan(aux.mean())))
        if np.isnan(aux.mean()):
            pd_final_v2[new_name] = pd_final_v2[col].fillna(-9999)
        else:
            pd_final_v2[new_name] = pd_final_v2[col]
        pd_final_v2[new_name] = pd_final_v2[new_name].replace([np.inf, -np.inf], -9999)

    for fold in range(num_folds):
        name_used = '%s_fold_%s' % ('md', fold)
        aux_feat_list = md_norm_list
        aux_pd        = all_partitions['fold_%s' % fold]
        aux_idx       = aux_pd[aux_pd['partition'] == 'training_%s' % fold]['unique_id'].to_numpy()
        create_qt(pd_final_v2[aux_feat_list].iloc[aux_idx], name_used, path_save_final_data)

    return all_partitions, pd_final_v2, md_norm_list


def add_dynamic_features(All_SNID, times_to_eval, path_dataset):

    eval_time = times_to_eval[0]
    aux_feat = \
        pd.read_parquet(('./dynamic_features/features_20220817_%4d.parquet' % eval_time).replace(' ', '0'))
    norm_name_used = 'norm'
    fe_norm_list   = ['%s_%s' % (norm_name_used, feat) for feat in aux_feat.columns]
    del aux_feat

    for eval_time in times_to_eval:
        print('- {}'.format(eval_time))
        aux_feat = \
            pd.read_parquet(('./dynamic_features/features_20220817_%4d.parquet' % eval_time).replace(' ', '0'))
        aux_feat.index.names = ['SNID']    
        for col in aux_feat.columns:
            aux = aux_feat[col].to_numpy().copy()

            ##### Deleting nans #######
            if np.isnan(aux.mean()):
                aux_feat['%s_%s' % (norm_name_used, col)] = aux_feat[col].fillna(-9999)
            else:
                aux_feat['%s_%s' % (norm_name_used, col)] = aux_feat[col]
            aux_feat['%s_%s' % (norm_name_used, col)] = aux_feat['%s_%s' % (norm_name_used, col)].replace([np.inf, -np.inf], -9999)    

        aux_feat = aux_feat.filter(items = All_SNID.astype(str), axis = 0)

        all_features = aux_feat[fe_norm_list].to_numpy()
        with h5py.File('./{}'.format(path_dataset), 'a') as hf:
            hf.create_dataset("norm_add_feat_col_%s" % eval_time, data = all_features)
        hf.close()