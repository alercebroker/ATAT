from . import plot_holoview as phv
import pandas as pd
def plot_common(metrics_ext, metrics_ext_gen, config, list_scores,
                                 total_z_umap, total_zgmm_umap_gmm):
    list_target = ['y']
    if config['embedding_supervised']:
        list_target += ['y_pred']
    if config['which_train_fn'] == 'VADE':
        list_target += ['y_pred_zgmm']

    if False:     
        count = 100
        saved_name = 'rec_'
        phv.save_all_curves(metrics_ext, config['rec_holoview_root'], config,
                        saved_name = saved_name, is_obs_data = True, count = count)
        metrics_ext['z1'] = total_z_umap[:,0]
        metrics_ext['z2'] = total_z_umap[:,1]
        pd_data = pd.DataFrame()
        for key in metrics_ext.keys():
            pd_data[key] = metrics_ext[key].tolist()
        pd_data['imgs'] = './' + 'rec_holoview_root' + '/' + saved_name +  pd_data['id'].astype(str) + '.jpg'
        phv.create_scatter(pd_data[:count], list_scores, list_target, config)
    if  metrics_ext_gen is not None and (config['which_train_fn'] == 'VAE' or config['which_train_fn'] == 'VADE' and False):
        metrics_ext_gen['z1'] = total_zgmm_umap_gmm[:,0]
        metrics_ext_gen['z2'] = total_zgmm_umap_gmm[:,1]
        pd_data = pd.DataFrame()
        for key in metrics_ext.keys():
            pd_data[key] = metrics_ext[key].tolist()
        saved_name  = 'gen_'
        pd_data['imgs'] = './gen_holoview_root' + '/' + saved_name +  pd_data['id'].astype(str) + '.jpg'
        phv.save_all_curves(metrics_ext_gen, metrics_plot_gen, config['gen_holoview_root'],
                                config, saved_name = saved_name, is_obs_data = False)
        list_target_gen = ['y']
        if config['embedding_supervised'] and not config['use_extra_feat']:
            list_target_gen += ['y_pred']
        if config['which_train_fn'] == 'VADE':
            list_target_gen += ['y_pred_zgmm']
        phv.create_scatter(pd_data, ['None'], list_target_gen, config, name_scatter = 'gen')
    phv.create_plot(metrics_ext, metrics_plot, list_scores, list_target, config)