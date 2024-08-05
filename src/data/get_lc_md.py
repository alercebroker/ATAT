import numpy as np
import pandas as pd
import os

from tqdm import tqdm

## Stream names to Training set names
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
    #'hostgal2_zphot_p50': 'HOSTGAL2_ZPHOT_P50',
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
    #'hostgal_zphot_p50': 'HOSTGAL_ZPHOT_P50',
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

def obtain_first(feature):
    return feature[0]

def obtain_feat(list_feat):
    return np.array(list_feat)

MAX_EPOCHS = 65 
def pad_list(lc, nepochs):
    pad_num = MAX_EPOCHS - nepochs
    if pad_num >= 0:
        return np.pad(lc, (0, pad_num), 'constant', constant_values=(0, 0))
    else:
        return np.array(lc)[np.linspace(0, nepochs - 1, num = MAX_EPOCHS).astype(int)]

# mask: 1 donde hay observaciones
def create_mask(lc):
    return (lc != 0).astype(float)

def normalizing_time(time_fid):
    mask_min =  9999999999 * (time_fid == 0).astype(float)
    t_min = np.min(time_fid + mask_min)
    return (time_fid - t_min) * (~(time_fid == 0)).astype(float)

# Matrix (Observaciones x filtros) 
def separate_by_filter(feat_time_series, bands):
    timef_array  = np.array(feat_time_series)
    band_array   = np.array(bands)
    colors = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'orange', 'z': 'brown', 'Y': 'k'}
    final_array = []
    for i, color in enumerate(colors.keys()):
        aux          = timef_array[band_array == color]
        nepochs      = len(aux)
        final_array += [pad_list(aux, nepochs)]
    return np.stack(final_array, 1)

def mask_alert(time, mask, photflags):
    time_flatten = time.reshape(-1,)[(photflags.reshape(-1,) != 0) & (photflags.reshape(-1,) != 1024)]
    aux_argsort  = time_flatten.argsort(0)
    time_ordered = time_flatten[aux_argsort]
    t_min        = time[photflags == 6144]
    mask_final   = mask.copy()
    mask_final[time < t_min - 30.0] = 0
    mask_final[time > time_ordered[-1]] = 0
    mask_final[photflags == 1024]       = 0
    return mask_final

def mask_detection(photflags):
    return (photflags >= 4096).astype(float)

def normalizing_time_alert(time, mask_alert, photflags):
    t_min = time[photflags == 6144]
    return (time - t_min) *  mask_alert

# Le restamos el primer MJD a todos los tiempos de todas las bandas
def normalizing_time_phot(time, mask_alert):
    mask_min =  999999999999 * (mask_alert == 0).astype(float)
    t_min = np.min(time + mask_min)
    return (time - t_min) *  mask_alert

### Auxiliary functions
def intersection_list(important_list, other_list):
    inter = []
    diff  = []
    for obj_imp in important_list:
        if obj_imp in other_list:
            inter += [obj_imp]
        else:
            diff  += [obj_imp]
    return inter, diff

def check_if_nan_in_list(pd_used, columns):
    nan_cols = []
    for col in columns:
        #print(col)
        if np.isnan(pd_used[col].to_numpy().mean()):
            nan_cols += [col]
    return nan_cols


def processing(base_dir, target_dir):

    for fine_class in tqdm(os.listdir(base_dir)):
        
        ###### Light curves observations ######
        # Reading and organizing
        print("- reading %s" % fine_class)
        df  = pd.read_pickle('%s/lc_%s.pkl' % (os.path.join(base_dir, fine_class), fine_class))
        df2 = df[['SNID', 'MJD', 'FLUXCAL', 'FLUXCALERR', 'BAND', 'PHOTFLAG']]

        # Group by ID (values in a list by ID)
        print("- grouping")
        df_grouped = df2.groupby(['SNID']).agg(lambda x: list(x))
        pd_dataset = pd.DataFrame()
        dict_transform = {'MJD': 'time', 'FLUXCAL': 'data', 'FLUXCALERR': 'data_var', 'PHOTFLAG': 'photflag'}
        list_time_feat = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'PHOTFLAG']
        band_key = 'BAND'

        # Separating
        print("- separating")
        for key_used in list_time_feat:
            pd_dataset[dict_transform[key_used]] =  \
                df_grouped.apply(lambda x: separate_by_filter(x[key_used], x[band_key] ), axis = 1)
        pd_dataset['time'] = pd_dataset.apply(lambda x: normalizing_time(x['time']), axis = 1)
        pd_dataset['mask'] = pd_dataset.apply(lambda x: create_mask(x['data']), axis = 1)

        print("- masking")
        pd_dataset['mask_alert']     = pd_dataset.apply(lambda x: mask_alert(x['time'], x['mask'], x['photflag']), axis = 1)
        pd_dataset['time_alert']     = pd_dataset.apply(lambda x: normalizing_time_alert(x['time'], x['mask_alert'], x['photflag']), axis = 1)
        pd_dataset['time_phot']      = pd_dataset.apply(lambda x: normalizing_time_phot(x['time'], x['mask_alert']), axis = 1)
        pd_dataset['mask_detection']  = pd_dataset.apply(lambda x: mask_detection(x['photflag']), axis = 1)

        ###### Metadata by astronomical object ######
        df_aux  = pd.read_pickle('%s/header_%s.pkl' % (os.path.join(base_dir, fine_class), fine_class))
        inter, diff = intersection_list(feat_list, df_aux.columns)
        df_aux = df_aux[inter + ['SNID']]

        for column in diff:
            df_aux[column] = np.nan

        pd_dataset = pd_dataset.reset_index()
        pd_dataset = pd_dataset.merge(df_aux, on = "SNID", how="left")
        pd_dataset = pd_dataset.set_index('SNID')

        print('- ', check_if_nan_in_list(pd_dataset, feat_list))
        print("")

        os.makedirs(target_dir, exist_ok=True)
        pd_dataset.to_pickle(os.path.join(target_dir, '%s.pkl' % fine_class))


