import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

fine_class_grouping_e1 = {
    'AGN': 'AGN',
    'CART': 'CART',
    'Cepheid': 'Cepheid',
    'd-Sct': 'Delta Scuti',
    'dwarf-nova': 'Dwarf Novae',
    'EB': 'EB',
    'ILOT': 'ILOT',
    'KN_B19': 'KN',
    'KN_K17': 'KN',
    'Mdwarf-flare': 'M-dwarf Flare',
    'PISN': 'PISN',
    'RRL': 'RR Lyrae',
    'SLSN-I+host': 'SLSN',
    'SLSN-I_no_host': 'SLSN',
    'SNIa-91bg': '91bg',
    'SNIa-SALT2': 'Ia',
    'SNIax': 'Iax',
    'SNIb+HostXT_V19': 'Ib/c',
    'SNIb-Templates': 'Ib/c',
    'SNIc+HostXT_V19': 'Ib/c',
    'SNIc-Templates': 'Ib/c',
    'SNIcBL+HostXT_V19': 'Ib/c',
    'SNII+HostXT_V19': 'II',
    'SNII-NMF': 'II',
    'SNII-Templates': 'II',
    'SNIIb+HostXT_V19': 'SN-like/Other',
    'SNIIn+HostXT_V19': 'II',
    'SNIIn-MOSFIT': 'II',
    'TDE': 'TDE',
    'uLens-Binary': 'uLens',
    'uLens-Single-GenLens': 'uLens',
    'uLens-Single_PyLIMA': 'uLens'
}

# We will use the group 1
fine_class_grouping = fine_class_grouping_e1

def open_original_files(base_dir):
    snid_per_fine_class = {}
    for fine_class in tqdm(os.listdir(base_dir)):
        try:
            class_lightcurves = pd.read_pickle(os.path.join(base_dir, fine_class, f'lc_{fine_class}.pkl'))

            snids = class_lightcurves['SNID'].unique()
            snid_per_fine_class[fine_class] = snids

        except FileNotFoundError:
            print('- files not found {}'.format(fine_class))

    msg = 'The grouping of the classes does not match. Please check the dictionary or the data_original file.'
    assert len(fine_class_grouping.keys()) == len(snid_per_fine_class.keys()), msg

    return snid_per_fine_class


def create_df_ids_labels(snid_per_fine_class):
    snids = []
    labels = []
    for fine_class in fine_class_grouping.keys():
        fine_class_snids = snid_per_fine_class[fine_class]
        snids.append(fine_class_snids)
        labels.append([fine_class_grouping[fine_class]]*len(fine_class_snids))
        
    snids_df = pd.DataFrame({
        'SNID': np.concatenate(snids),
        'label': np.concatenate(labels)
    })

    msg = 'There are SNIDs duplicated. Check the fix_ids_elasticc.py file.'
    assert snids_df.nunique().SNID == snids_df.shape[0], msg

    return snids_df


def create_split(snids_df, sample_test, num_folds):
    test_set = snids_df.groupby('label').sample(n=sample_test, random_state=0)
    test_set['partition'] = 'test'

    training_validation = snids_df[~snids_df['SNID'].isin(test_set['SNID'])]
    
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    training_set_list = []
    validation_set_list = []
    n_samples = len(training_validation)
    for i, indexes in enumerate(kf.split(np.zeros(n_samples), training_validation.label)):
        training_index, validation_index = indexes
        training_set = training_validation.iloc[training_index].copy()
        validation_set = training_validation.iloc[validation_index].copy()
        training_set['partition'] = f'training_{i}'
        validation_set['partition'] = f'validation_{i}'
        training_set_list.append(training_set)
        validation_set_list.append(validation_set)

    partitions = pd.concat([test_set]+training_set_list+validation_set_list, axis=0)

    blind_test = test_set.copy()
    blind_test['label'] = 'unknown'
    blind_partitions = pd.concat([blind_test]+training_set_list+validation_set_list, axis=0)

    return partitions, blind_partitions, test_set


def check_partitions(snids_df, partitions, test_set, sample_test, num_folds):
    n_classes = len(snids_df['label'].unique())
    print('- ', (len(snids_df)-n_classes*sample_test)*num_folds + n_classes*sample_test)
    print('-', len(partitions))

    snids_df[snids_df['label'].isin(['Delta Scuti', 'Cepheid'])]['SNID'].astype(float).sort_values()

    for astroclass in snids_df['label'].unique():
        print('- ', astroclass.ljust(15), snids_df[snids_df['label'] == astroclass]['SNID'].astype(float).mean())

    snids_df['SNID'] = snids_df['SNID'].astype(float)
    snids_df.sort_values('SNID')

    plt.hist(snids_df['SNID'], label='all', density=False)
    plt.hist(snids_df[snids_df['label'].isin(['Delta Scuti', 'Cepheid'])]['SNID'], label='dsct+ceph', density=False, alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    print('- ', snids_df[snids_df['SNID'].duplicated(keep=False)].sort_values('SNID').iloc[:30])

    assert (len(snids_df)-n_classes*sample_test)*num_folds + n_classes*sample_test == len(partitions)

    test_snids = partitions[partitions['partition'] == 'test']['SNID'].values
    non_test_snids = partitions[partitions['partition'] != 'test']['SNID'].unique()
    assert set(non_test_snids).intersection(set(test_set)) == set()


def save_partitions(base_dir, partitions, blind_partitions):
    partitions.to_parquet('./{}/partitions.parquet'.format(base_dir))
    blind_partitions.to_parquet('./{}/blind_partitions.parquet'.format(base_dir))
    

