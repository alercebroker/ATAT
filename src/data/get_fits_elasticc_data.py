from astropy.io import fits

import multiprocessing
import pandas as pd
import tqdm
import glob
import os


class GetElasticcData(object):
    '''
    This code transform the original ELASTICC data to data in a friendly format in a 
    multiprocessing or normal environment. 
    
    Args:
        load_parent_file (str): Directory where the objects are stored in the original format.
        save_parent_file (str): Directory where the objects will be saved in a new format.
        dataset_path (str): Path of the objects in a dataset. In our case ELASTICC.
        toy_example (bool): True if you want to create a toy example with 2-pair (HEAD - PHOT) cores .

    '''

    def __init__(
            self, 
            load_parent_file='0.0_data_original', 
            save_parent_file='data_toy', 
            dataset_name='ELASTICC', 
            subset_name='TRAIN', 
            there_are_exceptions=True,
            aux_subset_name='TRAINFIX',
            which_classes=['Cepheid', 'd-Sct'],
            toy_example=False
        ):
        
        # Primary variables
        self.load_parent_file = load_parent_file
        self.save_parent_file = save_parent_file
        self.dataset_name = dataset_name
        self.subset_name = subset_name

        self.dataset_path = './{}/{}_{}_*'.format(self.load_parent_file, 
                                                  self.dataset_name, 
                                                  self.subset_name)
        
        # Exceptions
        self.there_are_exceptions = there_are_exceptions
        self.aux_subset_name = aux_subset_name
        self.which_classes = which_classes

        self.toy_example = toy_example
        

        self.all_object_files = glob.glob(self.dataset_path)
        print('- all_object_files:\n{}\n'.format(self.all_object_files))

    def run(self, multiprocess, num_cores=8):
        '''
        Generate the files of the light curves and metadata of a specific astronomical object.
        
        Args:
            multiprocess (bool): True if it simultaneously preprocesses data from different cores of a specific astronomical object..
            num_cores (int): Number of cores to be used.

        Output:
            Two dataframe saved in a pickle file of all the cores of an object.

        '''

        if self.toy_example:
            if len(self.all_object_files) < num_cores: num_cores = len(self.all_object_files)
            print('- we are using {} cores'.format(num_cores))

            with multiprocessing.Pool(processes=num_cores) as pool:
                _ = list(tqdm.tqdm(pool.imap_unordered(self.processing_fits_by_class, self.all_object_files), 
                                               total=len(self.all_object_files)))
            
        else:
            for object_file in self.all_object_files:

                all_cores_files, name_object = self.__get_cores(object_file)

                path_save_lc = './{}/lc_{}.pkl'.format(self.save_path, name_object)
                path_save_header = './{}/header_{}.pkl'.format(self.save_path, name_object)

                if multiprocess:
                    if len(all_cores_files) < num_cores: num_cores = len(all_cores_files)
                    print('- we are using {} cores'.format(num_cores))

                    with multiprocessing.Pool(processes=num_cores) as pool:
                        mapped_values = list(tqdm.tqdm(pool.imap_unordered(self.processing_fits, all_cores_files), 
                                                    total=len(all_cores_files)))

                        df_cores_lc, df_cores_header = [], []
                        for df_lc, df_header in mapped_values:
                            df_cores_lc.append(df_lc)
                            df_cores_header.append(df_header)

                        df_cores_lc = pd.concat(df_cores_lc, axis=0).reset_index(drop=True)   
                        df_cores_header = pd.concat(df_cores_header, axis=0).reset_index(drop=True)

                        df_cores_lc.to_pickle(path_save_lc)
                        df_cores_header.to_pickle(path_save_header)

                else:
                    df_cores_lc, df_cores_header = [], []
                    for core_file in tqdm.tqdm(all_cores_files):
                        df_lc, df_header = self.processing_fits(core_file)  

                        df_cores_lc.append(df_lc)
                        df_cores_header.append(df_header)

                    df_cores_lc = pd.concat(df_cores_lc, axis=0).reset_index(drop=True)   
                    df_cores_header = pd.concat(df_cores_header, axis=0).reset_index(drop=True)

                    df_cores_lc.to_pickle(path_save_lc)
                    df_cores_header.to_pickle(path_save_header)

    def processing_fits(self, core_file):
        '''
        Gets the light curves of a core of an astronomical object. 
        * Core corresponds to a HEAD and PHOT file.

        Args:
            core_file (tuple): A tuple of a HEAD and PHOT path of a specific object. 

        Output:
            df_object_lc (pd.DataFrame): DataFrame of the light curves of a core. 
            df_object_header (pd.DataFrame): DataFrame of the metadata of a core.

        '''

        # Header
        metadata_fits = fits.open('./{}'.format(core_file[0]))
        data_core_head = metadata_fits[1].data
        columns_head = metadata_fits[1].columns.names

        # Phot
        data_fits = fits.open('./{}'.format(core_file[1]))
        data_core_phot= data_fits[1].data
        columns_phot = data_fits[1].columns.names

        # Gets all the light curves of a core
        list_core_lc = []
        list_core_header = []
        for idx_sn in range(data_core_head.shape[0]):
            row_mdata = data_core_head[idx_sn]

            ptrobs_min = row_mdata['PTROBS_MIN']
            ptrobs_max = row_mdata['PTROBS_MAX']

            lightcurve = pd.DataFrame([list(row) for row in data_core_phot[ptrobs_min-1:ptrobs_max]], 
                                        columns=columns_phot)

            lightcurve.insert(0, 'SNID', row_mdata['SNID'])
            lightcurve['SNTYPE'] = row_mdata['SNTYPE']

            header = pd.DataFrame([list(row_mdata)], columns=columns_head)

            list_core_lc.append(lightcurve)
            list_core_header.append(header)
       
        df_object_lc = pd.concat(list_core_lc, axis=0).reset_index(drop=True)   
        df_object_header = pd.concat(list_core_header, axis=0).reset_index(drop=True)  

        return df_object_lc, df_object_header
    
    def processing_fits_by_class(self, object_file):
        '''
        Gets the light curves of a core of an astronomical object. 
        * Core corresponds to a HEAD and PHOT file.

        Args:
            core_file (tuple): A tuple of a HEAD and PHOT path of a specific object. 

        Output:
            df_object_lc (pd.DataFrame): DataFrame of the light curves of a core. 
            df_object_header (pd.DataFrame): DataFrame of the metadata of a core.

        '''

        object_file = object_file.replace('\\', '/').split('/')[-1]
        name_object = object_file.replace('{}_{}_'.format(self.dataset_name, self.subset_name), "")

        self.save_path = './{}/{}'.format(self.save_parent_file, name_object)
        os.makedirs(self.save_path, exist_ok=True)

        path_save_lc = './{}/lc_{}.pkl'.format(self.save_path, name_object)
        path_save_header = './{}/header_{}.pkl'.format(self.save_path, name_object)

        # Files of one class
        if name_object in self.which_classes:
            files_path = './{}/{}/{}_{}_NONIaMODEL0*'.format(self.load_parent_file, 
                                                             object_file,
                                                             self.dataset_name,
                                                             self.aux_subset_name)         
        else: 
            files_path = './{}/{}/{}_{}_NONIaMODEL0*'.format(self.load_parent_file, 
                                                             object_file,
                                                             self.dataset_name,
                                                             self.subset_name)

        all_cores_files = glob.glob(files_path)
        all_cores_files.sort()

        try:
            assert len(all_cores_files) != 0
        except AssertionError:
            print("\n- object {} has a exception in its name file that you do not consider".format(name_object))
            exit()

        # Take two parts by class
        if self.toy_example:
            all_cores_files = all_cores_files[:4]

        # Tuple of the phot and header files
        it = iter(all_cores_files)
        all_cores_files = [*zip(it, it)]

        df_object_lc = []
        df_object_header = []
        for core_file in all_cores_files:
            
            # Header
            metadata_fits = fits.open('./{}'.format(core_file[0]))
            data_core_head = metadata_fits[1].data
            columns_head = metadata_fits[1].columns.names

            # Phot
            data_fits = fits.open('./{}'.format(core_file[1]))
            data_core_phot= data_fits[1].data
            columns_phot = data_fits[1].columns.names

            # Gets all the light curves of a core
            list_core_lc = []
            list_core_header = []
            for idx_sn in range(data_core_head.shape[0]):
                row_mdata = data_core_head[idx_sn]

                ptrobs_min = row_mdata['PTROBS_MIN']
                ptrobs_max = row_mdata['PTROBS_MAX']

                lightcurve = pd.DataFrame([list(row) for row in data_core_phot[ptrobs_min-1:ptrobs_max]], 
                                            columns=columns_phot)

                lightcurve.insert(0, 'SNID', row_mdata['SNID'])
                lightcurve['SNTYPE'] = row_mdata['SNTYPE']

                header = pd.DataFrame([list(row_mdata)], columns=columns_head)

                list_core_lc.append(lightcurve)
                list_core_header.append(header)
        
            df_object_lc.append(pd.concat(list_core_lc, axis=0).reset_index(drop=True))   
            df_object_header.append(pd.concat(list_core_header, axis=0).reset_index(drop=True))

        df_cores_lc = pd.concat(df_object_lc, axis=0).reset_index(drop=True)   
        df_cores_header = pd.concat(df_object_header, axis=0).reset_index(drop=True)

        df_cores_lc.to_pickle(path_save_lc)
        df_cores_header.to_pickle(path_save_header)

    def __get_cores(self, object_file):
        '''
        Gets the name and path of the cores (header, phot) of the preprocessed astronomical object.

        Args:
            object_file (str): Path of an astronomical object. 
            dataset_name (str): Name of the dataset in which the objects are stored.

        Output:
            object_file (str): Path of the cores of an astronomical object. 
            name_object (str): Object name.

        '''

        object_file = object_file.replace('\\', '/').split('/')[-1]
        name_object = object_file.replace('{}_{}_'.format(self.dataset_name, self.subset_name), "")

        self.save_path = './{}/{}'.format(self.save_parent_file, name_object)
        os.makedirs(self.save_path, exist_ok=True)

        print('- DATASET {} - PREPROCESSING OBJECT: {}'.format(self.dataset_name, name_object))
        # Files of one class
        if name_object in self.which_classes and self.there_are_exceptions:
            print('- we are considering exceptions in the file of this object')
            files_path = './{}/{}/{}_{}_NONIaMODEL0*'.format(self.load_parent_file, 
                                                                object_file,
                                                                self.dataset_name,
                                                                self.aux_subset_name)         
        else: 
            files_path = './{}/{}/{}_{}_NONIaMODEL0*'.format(self.load_parent_file, 
                                                                object_file,
                                                                self.dataset_name,
                                                                self.subset_name)
        all_cores_files = glob.glob(files_path)
        all_cores_files.sort()

        try:
            assert len(all_cores_files) != 0
        except AssertionError:
            print("\n- object {} has a exception in its name file that you do not consider".format(name_object))
            exit()

        # Tuple of the phot and header files
        it = iter(all_cores_files)
        all_cores_files = [*zip(it, it)]

        return all_cores_files, name_object


#if __name__ == '__main__':
#    # Note: toy_example is only multiprocess
#    load_parent_file = 'data_original_fits'
#    save_parent_file = 'data_preprocessed'
#    dataset_name = 'ELASTICC'
#    subset_name = 'TRAIN'
#
#    # Exceptions
#    there_are_exceptions = True
#    aux_subset_name = 'TRAINFIX'
#    which_classes = ['d-Sct', 'Cepheid']
#
#    toy_example = False
#
#    get_data = GetElasticcData(load_parent_file, 
#                               save_parent_file, 
#                               dataset_name, 
#                               subset_name, 
#                               there_are_exceptions,
#                               aux_subset_name,
#                               which_classes,
#                               toy_example=toy_example)
#    
#    get_data.run(multiprocess=True, num_cores=20)

    