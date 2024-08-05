import requests
import tarfile
import h5py
import wget
import os
import re

import torch 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.data.get_fits_elasticc_data import GetElasticcData
from src.data.get_lc_md import processing
from src.data.get_partitions import *
from src.data.get_qt_feat_and_h5_file import *

def check_directory(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "First you should download the data from "
            "`https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC_TRAINING_SAMPLES/FULL_ELASTICC_TRAIN.tar` "
            "and place it inside the project in a folder named `data_original_fits`."
        )

if __name__ == '__main__':
    num_cores = 20
    use_paper_partitions = True
    path_save_fits = 'data_original_fits'
    check_directory('{}/FULL_ELASTICC_TRAIN.tar'.format(path_save_fits))
    #os.makedirs('./{}'.format(path_save_fits), exist_ok=True)

    #-------------------------------- Download data --------------------------------#
    #print('Downloading ELASTICC data ...')
    #url = 'https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/TRAINING_SAMPLES/FULL_ELASTICC_TRAIN.tar'
    #filename = wget.download('{}'.format(url), out='./{}/'.format(path_save_fits)) 

    #-------------------------------- Unzip data fits --------------------------------#
    print('Unzipping ELASTICC data ...')
    path_save_tar_file = './{}/FULL_ELASTICC_TRAIN.tar'.format(path_save_fits)
    tar = tarfile.open(path_save_tar_file)
    tar.extractall('./{}'.format(path_save_fits))
    tar.close()
    os.remove('./{}'.format(path_save_tar_file))

    #-------------------------------- Extract data --------------------------------#
    print('Extracting info from FITS ...')

    # We recommend using 20 cores to extract the data by classes
    save_parent_file = 'data_extracted'
    classes_to_fix = ['Cepheid', 'd-Sct']
    sample_test = 1000

    get_data = GetElasticcData(load_parent_file = path_save_fits,
                               save_parent_file = save_parent_file,
                               dataset_name = 'ELASTICC',
                               subset_name = 'TRAIN',
                               there_are_exceptions = True,
                               aux_subset_name = 'TRAINFIX',
                               which_classes = classes_to_fix)
    
    get_data.run(multiprocess=True, num_cores=num_cores)

    # We had to fix the ID of classes ['d-Sct', 'Cepheid'] because they overlapped other IDs of other classes
    print('Fixing the IDs of classes Cepheid and d-Sct ...')
    list_new_ids = ['1000000', '2000000']
    for i in range(len(classes_to_fix)):
        header_obj = pd.read_pickle('./{}/{}/header_{}.pkl'.format(save_parent_file, classes_to_fix[i], classes_to_fix[i]))
        lc_obj = pd.read_pickle('./{}/{}/lc_{}.pkl'.format(save_parent_file, classes_to_fix[i], classes_to_fix[i]))

        header_obj.SNID = header_obj.SNID.apply(lambda row: list_new_ids[i]+row)
        lc_obj.SNID = lc_obj.SNID.apply(lambda row: list_new_ids[i]+row)

        header_obj.to_pickle('./{}/{}/header_{}.pkl'.format(save_parent_file, classes_to_fix[i], classes_to_fix[i]))
        lc_obj.to_pickle('./{}/{}/lc_{}.pkl'.format(save_parent_file, classes_to_fix[i], classes_to_fix[i]))

    #-------------------------------- Data processing --------------------------------#
    print('Data processing ...')
    target_dir = 'data_processed'
    os.makedirs(target_dir, exist_ok=True)

    processing(base_dir=save_parent_file, 
               target_dir=target_dir)

    #-------------------------------- Create data partition --------------------------------#
    print('Creating data paritions ...')
    save_dir = 'data_partition'
    num_folds = 5
    os.makedirs(save_dir, exist_ok=True)

    snid_per_fine_class = open_original_files(base_dir=save_parent_file)
    snids_df = create_df_ids_labels(snid_per_fine_class)
    partitions, blind_partitions, test_set = create_split(snids_df, sample_test, num_folds)
    #check_partitions(snids_df, partitions, test_set, sample_test, num_folds)
    save_partitions(save_dir, partitions, blind_partitions)

    #-------------------------------- Generate QT from static features (metadata) --------------------------------#
    print('Generating data files joined and getting QT from static features (metadata) ...') 
    if use_paper_partitions == True and os.path.exists('./data_partition_paper/partitions.parquet'):
        save_dir = 'data_partition_paper'
    else:
        print('- we are creating new partitions')

    path_blind_partitions = './{}/blind_partitions.parquet'.format(save_dir)
    path_partitions = './{}/partitions.parquet'.format(save_dir)

    path_save_final_data = 'final_dataset'
    os.makedirs('./{}'.format(path_save_final_data), exist_ok=True)

    pd_final = joint_classes(target_dir)
    print('- pd_final.shape: ', pd_final.shape)
    total_partitions_not_blind = open_partitions(path_blind_partitions, path_partitions, path_save_final_data)
    all_partitions, pd_final_v2, md_norm_list = get_qt_metadata(pd_final, total_partitions_not_blind, num_folds, path_save_final_data)

    #--------------------------------  Remove generated files --------------------------------# 
    os.remove(save_parent_file) # extracted data
    os.remove(target_dir) # processed data

    #--------------------------------  Create final dataset --------------------------------# 
    print('Creating dataset with lc and static features ...')
    create_dataset(all_partitions, pd_final_v2, md_norm_list, num_folds, path_save_final_data) 

    #--------------------------------  Add dynamic features within of the dataset --------------------------------# 
    if os.path.exists('./dynamic_features/features_20220817_2048.parquet'):
        print('Adding dynamic features to the dataset ...')
        path_dataset = '{}/elasticc_final.h5'.format(path_save_final_data)
        h5_file = h5py.File(path_dataset)
        All_SNID = h5_file.get('SNID')[:]
        h5_file.close()

        times_to_eval = []
        for i in range(11,-1,-1):
            times_to_eval += [2**i]
        print('- {}'.format(times_to_eval))

        add_dynamic_features(All_SNID, times_to_eval, path_dataset)
    
    #-------------------------------- Generate QT from dynamic features (calculated) --------------------------------# 
        print('Generating QT for dynamic features in different times [1, 2, 4, 8, 16, 32, ..., 1024, 2048] ...')
        path_dataset = '{}/elasticc_final.h5'.format(path_save_final_data)
        h5_file = h5py.File(path_dataset)

        for eval_time in times_to_eval:
            for fold in range(num_folds):
                print('- creating QT for dynamic features in {} days and fold {}'.format(eval_time, fold))
                these_idx  = h5_file.get('training_{}'.format(fold))[:]
                norm_add_feat_col = torch.from_numpy(h5_file.get('norm_add_feat_col_{}'.format(eval_time))[:][these_idx])
                create_qt(norm_add_feat_col, 'fe_{}_fold_{}'.format(eval_time, fold), path_save_final_data)

        h5_file.close()


