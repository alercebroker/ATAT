import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
import pickle
import math
old_class_names = [ 'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT',
                              'KN', 'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c',
                              'II', 'SN-like/Other', 'TDE', 'uLens']

new_class_names = [ 'CART', 'Iax', '91bg', 'Ia', 'Ib/c', 'II', 'SN-like/Other', 'SLSN', 'PISN',
                    'TDE', 'ILOT', 'KN', 'M-dwarf Flare', 'Dwarf Novae', 'AGN',
                    'uLens', 'Delta Scuti', 'RR Lyrae', 'Cepheid', 'EB']

new_class_names2 = [ 'CART', 'Iax', '91bg', 'Ia', 'Ib/c', 'II', 'SN-like/Other', 'SLSN', 'PISN',
                    'TDE', 'ILOT', 'KN', 'M-dwarf Flare', 'uLens', 'Dwarf Novae', 'AGN', 'Delta Scuti', 'RR Lyrae', 'Cepheid', 'EB']

idx_transform  = {idx: old_class_names.index(key) for idx, key in enumerate(new_class_names)}
idx_transform2 = {idx: old_class_names.index(key) for idx, key in enumerate(new_class_names2)}
idx_transform3 = {idx: new_class_names.index(key) for idx, key in enumerate(new_class_names2)}

def summarize_eval_metric_many(metrics_stat_dict, root_name, is_multiple_models = False, 
                                    metric_name = 'time', set_type = 'test', word_dict = None,
                                    add_name = '', save_pickle = False):
    final_plot_dict       = {}
    list_score_used       = ['F1Score']
    list_score_used_valid = []
    list_features         = metrics_stat_dict[set_type].columns \
                            if is_multiple_models else metrics_stat_dict[set_type].keys()
    for score_used in list_score_used:
        for key in list_features:
            if score_used in key:
                list_score_used_valid += [key]

    eval_metric_str    = '%s_%s_' % (set_type, metric_name)
    eval_set_type    = np.array([metric_str for metric_str in metrics_stat_dict.keys() \
                                    if eval_metric_str in metric_str])
    number_in_cols   = np.array([int(col.replace(eval_metric_str, '')) for col in eval_set_type])
    final_plot_dict  = {}

    for i_score_used, score_used in enumerate(list_score_used_valid):
        final_plot_dict[score_used] = {}
        for i_set_type, set_type in enumerate(eval_set_type[number_in_cols.argsort()]):
            metrics_stat = metrics_stat_dict[set_type]
            if is_multiple_models:
                for i_row, row in enumerate(metrics_stat[[score_used]].iterrows()):
                    str_name = ' '.join(row[0]) if type(row[0]) == tuple else row[0]
                    if i_set_type == 0:
                        final_plot_dict[score_used][str_name]  = [float(row[1])]
                    else:
                        final_plot_dict[score_used][str_name] += [float(row[1])]
            else:
                if i_set_type == 0:
                    final_plot_dict[score_used]  = [float(metrics_stat[score_used])]                
                else:
                    final_plot_dict[score_used] += [float(metrics_stat[score_used])]

        # if is_multiple_models:
        #     import pdb
        #     pdb.set_trace()
        #     print("lala")
        if metric_name == 'time':
            plot_eval_time_many(number_in_cols[number_in_cols.argsort()],
                            final_plot_dict[score_used], score_used,
                            filename = '%s/%s' % (root_name, '%s_eval_%s_%s%s' % (set_type, metric_name, score_used, add_name) ),
                            word_dict = word_dict, save_pickle = save_pickle)

def plot_eval_time_many(times_to_eval, values, score_name, filename = None, word_dict = None,
                                                                             save_pickle = False):
    if type(word_dict) == dict and 'figsize_l' in word_dict.keys():
        plt.figure(figsize = (word_dict['figsize_l'], word_dict['figsize_r']))
    else:
        plt.figure(figsize = (12,8))
    fs = 16
    if type(values) == dict:
        n_colors   = len(values)
        all_color  = plt.cm.rainbow(np.linspace(0, 1, n_colors))
        if type(word_dict) == dict:
            if 'fs' in word_dict.keys():
                fs = word_dict['fs']
            for idx, key in enumerate(word_dict['models'].keys()):
                this_dict = word_dict['models'][key]
                plt.plot(np.log(times_to_eval)/np.log(2), values[key], color = this_dict['color'],
                     linewidth=4, linestyle= this_dict['lines'], label = this_dict['name'], alpha = word_dict['alpha'] \
                                                                                        if 'alpha' in word_dict.keys() else 0.8)                

        else:
            for idx, key in enumerate(values.keys()):
                plt.plot(np.log(times_to_eval)/np.log(2), values[key], color = all_color[idx],
                     linewidth=4, linestyle='--', label = key)
        plt.legend(fontsize = fs, loc = 'lower center')
    else:
        plt.plot(np.log(times_to_eval)/np.log(2), values, color = 'blue', linewidth=4, linestyle='--')
    plt.xticks(np.log(times_to_eval)/np.log(2), times_to_eval, fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlabel("Evaluated time (days after first alert)", fontsize = fs)
    if type(word_dict) == dict:
        plt.ylabel(word_dict['ylabel'], fontsize = fs)
        if 'ylim_max' in word_dict.keys():
            plt.ylim((word_dict['ylim_min'], word_dict['ylim_max']))
        if 'yticks' in word_dict.keys():
            plt.yticks([])
    else:
        plt.ylabel(score_name, fontsize = fs)
    plt.tight_layout()
    if filename is not None:
        plt.savefig('%s.jpg' % filename, transparent=True)
    if save_pickle:
        with open('%s.pickle' % filename, 'wb') as handle:
            pickle.dump(values, handle, protocol=pickle.HIGHEST_PROTOCOL)





def plot_confusion_matrix(cm, classnames, cm_std = None, filename = None,
                     apply_transformation = True, fs = 28, apply_new_transformation = ''):
    #### HARCODING!!!!!!!!!!!!!!!!!!!!!!!!!!! ########################
    # classnames = new_class_names2
    # if apply_transformation:
    #     new_id = np.array(list(idx_transform2.values()))
    #     cm = cm[new_id, :][:, new_id]
    #     if cm_std is not None:
    #         cm_std = cm_std[new_id, :][:, new_id]

    if apply_new_transformation != '':
        if apply_new_transformation == '2':
            idx_transform = idx_transform2
        if apply_new_transformation == '3':
            idx_transform = idx_transform3
        new_id = np.array(list(idx_transform.values()))
        cm = cm[new_id, :][:, new_id]
        if cm_std is not None:
            cm_std = cm_std[new_id, :][:, new_id]

    ###############################################################

    #classnames = sorted(classnames)
    #sns.set_style("whitegrid", {'axes.grid' : False})
    #cm = confusion_matrix(y, y_pred)
    plt.clf()
    plt.figure(figsize = (28,28))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax = 1)
               #vmax = np.unique(y, return_counts=True)[1].max() if cm.max()> 1 else 1)
    #plt.title(title)
    #plt.colorbar()

    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=90, fontsize = fs)
    plt.yticks(tick_marks, classnames, fontsize = fs)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        idx_i, idx_j = i, j
        #idx_i, idx_j = idx_transform[i], idx_transform[j]
        if cm_std is not None:
            plt.text(j, i,  '  {0:.2f}'.format(cm[idx_i, idx_j]) + '\n$\pm$' + '{0:.2f}'.format(cm_std[idx_i, idx_j]),
                     horizontalalignment="center",
                     color="white" if cm[idx_i, idx_j] > thresh else "black", fontsize = 14)
        else:
            plt.text(j, i,  '{0:.2f}'.format(cm[idx_i, idx_j]),  horizontalalignment="center", verticalalignment = "center",
                     color="white" if cm[idx_i, idx_j] > thresh else "black", fontsize = fs)
    #plt.tight_layout()
    plt.ylabel('True label', fontsize = fs)
    plt.xlabel('Predicted label', fontsize = fs)
    if filename is not None:
        plt.savefig('%s.jpg' % filename)

def plot_all_report(cm, classnames, cm_std = None, root = None, filename = None,
                     apply_transformation = True, fs = 28, apply_new_transformation = ''):
    #### HARCODING!!!!!!!!!!!!!!!!!!!!!!!!!!! ########################
    #classnames = new_class_names2
    # if apply_transformation:
    #     new_id = np.array(list(idx_transform2.values()))
    #     cm = cm[new_id]
    #     if cm_std is not None:
    #         cm_std = cm_std[new_id]

    mean_data = pd.DataFrame(cm, columns = ['Precision', 'Recall', 'F1-Score'], index = classnames)
    mean_data = mean_data.rename_axis(('Classnames'))
    if cm_std is not None:
        std_data = pd.DataFrame(cm_std, columns = ['Precision', 'Recall', 'F1-Score'], index = classnames)
        std_data = std_data.rename_axis(('Classnames'))
    else:
        std_data = None
    print_latex_table(mean_data, {'print_dict': {}}, root, filename,
                    pd_grouped_std = std_data)

def print_latex_table(pd_grouped_mean, dict_setting, folder_root, table_name, pd_grouped_std = None):
    import numpy as np
    def obtain_values(nn):
        all_counters  = []
        all_ncounters = []
        all_colors    = []
        all_groups    = []
        for i_column in range(nn.shape[-1]):
            values         = nn[:, i_column]
            counter_list   = []
            color_list     = []
            group_list     = []
            distinct_index = -1
            aux_str   = 'None'
            color_index  = 1
            for i, value in enumerate(values):
                if value != aux_str:
                    counter_list   += [1]
                    distinct_index  = i
                    aux_str         = value
                    color_index    += 1
                else:
                    counter_list[distinct_index]  += 1
                    counter_list                  += [-1]
                color_list     += [color_index%2]
            aux_value = 0
            ncounter_list = []
            for i, value in enumerate(counter_list):
                if value != -1:
                    if i > 0:
                        ncounter_list[-1] = aux_value
                    aux_value = value
                group_list   += [aux_value]
                ncounter_list += [-1]
            ncounter_list[-1] = aux_value
            all_counters     += [counter_list]
            all_colors       += [color_list]
            all_groups       += [group_list]
            all_ncounters    += [ncounter_list]
        return all_ncounters, all_colors, all_groups 

    def obtain_code_color(acolors, agroups):
        lcodes = ['xxx', 'xxy', 'xyx', 'xyy', 'yxx', 'yxy', 'yyx', 'yyy'][::-1]
        np_lcodes = np.array(lcodes)[::-1]
        a2    = np.array(['bb0', 'bb1'])
        arange = np.arange(acolors.shape[-1])

        ar = agroups.mean(1)[::-1]
        max_ocurrence = (ar > 1).argmax()
        lenght_copies = (ar > ar[max_ocurrence]).argmax() 
        lenght_copies = lenght_copies if lenght_copies > 0 else len(agroups)

        acodes = []
        acodes += lenght_copies * [list(np_lcodes[acolors[-1 - max_ocurrence, :]])]
        tdict = {np_lcodes[0]: np_lcodes[0], np_lcodes[1]: np_lcodes[1]}
        current_codes = [np_lcodes[0], np_lcodes[1]]

        for i in range(len(agroups) - lenght_copies):
            aux1 = np_lcodes[acolors[- 1 - max_ocurrence, :]] if i == 0 else np.array(acode)
            aux2 = a2[acolors[- 1 - i - lenght_copies, :]]
            filter_i = agroups[-i - 1 -lenght_copies] <= agroups[ - i - 0 - lenght_copies]
            aux2[arange[filter_i]] = aux1[arange[filter_i]]

            acode = []
            lcodes_copy = lcodes.copy()
            for this_code in current_codes:
                lcodes_copy.remove(this_code)
            current_keys = []
            for this_str in aux2:
                if (not (this_str in current_keys) and not (this_str in current_codes)):
                    avar = lcodes_copy.pop()
                    tdict[this_str] = avar
                    current_keys += [this_str]
                acode += [tdict[this_str]]
            current_codes = list(tdict.values())
            acodes += [acode]
        return np.array(acodes).T[:,::-1]
                
    def str_row(row_numbers, this_str, this_code):
        if row_numbers > 1:
            return '\%s \multirow{-%s}{*}{%s}' % (this_code, row_numbers, this_str)
        elif row_numbers == 1:
            return '\%s  %s' % (this_code, this_str)
        else:
            return '\%s ' % this_code
        
    def obtain_vtran(value, ponder, value_if = None):
        value_if = value_if if value_if is not None else value
        if value_if > 100:
            return '%1.3f' % (value/ponder)
        elif value_if >= 10:
            return '%1.2f' % value
        elif value_if >= 1:
            return '%1.3f' % value
        elif value_if >= 0.1:
            return '%0.4f' % value
        else:
            return '%1.3f' % (value/ponder)
        
    def get_ponder(value):
        this_var = '%2.2e' % value
        ponder = float('1%s'%this_var[-4:]) 
        if value > 100 or value < 0.1:
            add_str = f' ({this_var[-4:]})'
        else:
            add_str = None
        return ponder, add_str

    def print_score(x, i_row, i_column, other_df = None, ponder = None, value_min = None):
        str_output  = '$'
        str_output += '%3.3f' % x if ponder is None else obtain_vtran(x, ponder,
                                                                    value_if = value_min)
        if other_df is not None:
            std_value   =  other_df.values[i_row, i_column]
            str_output += '\pm '
            str_output += '%3.3f' % std_value if ponder is None else obtain_vtran(std_value,
                                                             ponder, value_if = value_min) 
        str_output += '$'
        return str_output

    def row_latex_from_pandas(df, print_dict, save_path, table_name, other_df = None):
        def pdt(aux_key,  add_str = None):
            aux_name = print_dict[aux_key] if aux_key in print_dict.keys() else aux_key
            aux_name = aux_name.replace('_', '').replace('mean1', '').replace('mean2', '')
            if add_str is not None:
                aux_name += add_str
            return aux_name
        groupby = list(df.index.names)
        scores  = df.columns
        nn      = np.asarray(df.index.to_list())
        nn      = nn if len(nn.shape) > 1 else np.expand_dims(nn, 1)
        l1, l2, l3  = obtain_values(nn)    
        all_counters, acolors, agroups = np.array(l1), np.array(l2), np.array(l3)    
        code_colors = obtain_code_color(acolors, agroups)
        # String configuration
        min_values  = df.min().to_list()
        ponder_list = min_values.copy()
        col_str  = [''] * len(ponder_list)
        for i in range(len(ponder_list)):
            if not math.isnan(ponder_list[i]):
                ponder_list[i], col_str[i] = get_ponder(ponder_list[i])
            else:
                ponder_list[i], col_str[i] = None, None

        index = 0
        with open("%s/%s.tex" % (save_path, table_name), "w") as f:
            f.write("\\begin{tabular}{" + "".join(["c"] * len(groupby)) + "|"  + "".join(["c"] * len(scores)) + "}\n")
            f.write('\\toprule \n')
            #import pdb
            #pdb.set_trace()
            f.write( " & ".join(['%s' % pdt(x) for x in groupby]) \
                + " & "  + " & ".join('%s' % pdt(x, col_str[i_col]) for i_col, x in enumerate(scores)) + " \\\\\n")
            f.write('\midrule \n')
            for i_row, row in enumerate(df.iterrows()):
                f.write( " & ".join([str_row(this_column[i_row], pdt(nn[i_row, i_column]), code_colors[i_row, i_column]) \
                        for i_column, this_column in enumerate(all_counters)  ]) \
                    + " & \%s"  % code_colors[i_row, -1] \
                    + (" & \%s" %  code_colors[i_row, -1]).\
                        join([print_score(x, i_row, i_col, other_df = other_df,
                            ponder = ponder_list[i_col], value_min = min_values[i_col]) \
                            for i_col, x in enumerate(row[1].values) ]) + " \\\\\n")
            f.write('\\bottomrule \n')
            f.write("\\end{tabular}")
    return row_latex_from_pandas(pd_grouped_mean, dict_setting['print_dict'], folder_root, table_name, pd_grouped_std)