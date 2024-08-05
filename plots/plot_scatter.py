import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from . import plot_scatter    as splot
import pandas as pd
import gc
import numpy as np
FONTSIZE = 20
FONTSIZE2 = 22

def scatter(data_x, data_y, target, label_x, label_y, label_target,
                                                 path_to_save, is_colorbar = False):
    #figure
    plt.clf()
    fig, ax1 = plt.subplots()
    fig.set_size_inches(20, 15)
    #plot
    ax1.set_xlabel(label_x, fontsize = FONTSIZE)
    ax1.set_ylabel(label_y, fontsize = FONTSIZE)
    ax1.set_title(label_y + ' vs ' + label_x + ' for ' + label_target, fontsize = FONTSIZE)
    ax1.tick_params(axis='both', labelsize = FONTSIZE2)
    plt.scatter(data_x, data_y, c = target, cmap = 'viridis', s = 50., alpha =0.4)
    if is_colorbar:
        cbar = plt.colorbar()
        cbar.set_label(label_target, fontsize=12)
    plt.legend(fontsize = 18, loc='upper right')
    fig.savefig(path_to_save)
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_density(total_z_to_scatter, total_z_to_kde, total_y_to_scatter, total_y_to_kde, 
                                                name_scatter, name_kde, save_path, config, y_to_scatter_is_score = False):
    all_cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'gray_r', 'YlGn',
                 'PuRd','YlOrBr', 'autumn_r','vlag_r', 'pink','PuBu']
    all_cmaps2 = ['blue', 'green', 'red', 'purple', 'gray', 'greenyellow',
                'magenta','orange', 'yellow', 'sienna', 'black','cyan']
    d         = 2
    n_lvl     = 10
    cut       = 1. 
    bw_adjust = 1
    g = sns.JointGrid()
    import matplotlib.patches as  mpatches
    label_patches = []
    #for i in np.unique(config['n_classes']):
    num_iter = len(np.unique(total_y_to_kde))
    for i in np.unique(total_y_to_kde):
        index = total_y_to_kde == i
        ddata = {'col1': total_z_to_kde[index][:, 0], 'col2': total_z_to_kde[index][:, 1]}
        df = pd.DataFrame(data=ddata)
        sns.kdeplot(df.col1, df.col2, cmap= all_cmaps[i], shade=False, shade_lowest=False, ax=g.ax_joint,
                                      levels = n_lvl, cut  = cut, bw_adjust = bw_adjust, alpha =.4)
        color_index = -1 if i == config['n_classes'] else i
        label_patch = mpatches.Patch(color=sns.color_palette(all_cmaps[color_index])[2], 
                                        label = name_kde + config['classes_names'][i])
        label_patches.append(label_patch)
    if not y_to_scatter_is_score:
        num_iter = len(np.unique(total_y_to_scatter))
        for i in np.unique(total_y_to_scatter):
            index = total_y_to_scatter == i
            ddata = {'col1': total_z_to_scatter[index][:, 0], 'col2': total_z_to_scatter[index][:, 1]}
            df = pd.DataFrame(data=ddata)
            color_index = -1 if i == config['n_classes'] else i
            sns.scatterplot(df.col1, df.col2, ax=g.ax_joint, color = all_cmaps2[color_index], 
                      alpha = 0.4, s = 50, label = name_scatter + config['classes_names'][i])
    else:
        ddata = {'col1': total_z_to_scatter[:, 0], 'col2': total_z_to_scatter[:, 1]}
        df = pd.DataFrame(data=ddata)
        sns.scatterplot(df.col1, df.col2, hue = total_y_to_scatter, ax=g.ax_joint, palette='viridis', 
                      alpha = 0.4, s = 50)
        norm = plt.Normalize(total_y_to_scatter.min(), total_y_to_scatter.max() )
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])         
    plt.gcf().set_size_inches(12, 12)
    ax = plt.gcf().axes
    ax[0].set_xlabel(r'$\mathcal{Z}_1$', fontsize=38)
    ax[0].set_ylabel(r'$\mathcal{Z}_2$', fontsize=38)
    if not y_to_scatter_is_score:
      plt.legend()
    else:	
      ax[0].get_legend().remove()
      ax[0].figure.colorbar(sm,shrink=2, pad=.8, aspect=40).set_label(label = name_scatter,size=24)
    plt.legend(handles=label_patches, loc='upper left')
    plt.savefig(save_path, transparent=True)
    plt.clf()
    import gc;
    gc.collect()

def plot_z(total_z, total_y, config, image_filename, n_colors):
    scatter_color(total_z[:,0], total_z[:,1], 'z1', 'z2', total_y, image_filename, config['classes_names'], n_colors)

def scatter_color(total_z1, total_z2, label_1, label_2, total_y, image_filename, list_of_names, n_colors):
    plt.clf()
    all_color        = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    cmap             = plt.cm.get_cmap("viridis")
    #fig              = plt.figure(figsize = (12,12) )
    #ax               = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 18)

    ax.set_xlabel(label_1, fontsize = FONTSIZE)
    ax.set_ylabel(label_2, fontsize = FONTSIZE)
    ax.tick_params(axis='both', labelsize = FONTSIZE2)
    
    for i in range(len(list_of_names)):
        index = total_y == i
        color_index = all_color[n_colors - 1 - i]
        ax.scatter(total_z1[index], total_z2[index],
                           alpha = 0.4, c = np.tile(color_index,(len(total_z1[index]),1)), s=80, 
                           cmap = cmap, label = list_of_names[i] )
    plt.legend(fontsize = FONTSIZE)
    #plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(image_filename)
    plt.close(fig)
    plt.clf()
    import gc;
    gc.collect()


# def plot_z(total_z, total_y, config, image_filename):
#     plt.clf()
#     number_of_colors = int(np.ceil(len(config['classes_names']) / 10) * 10)
#     all_color        = plt.cm.rainbow(np.linspace(0, 1, number_of_colors))
#     cmap             = plt.cm.get_cmap("viridis")
#     fig              = plt.figure(figsize = (12,12) )
#     ax               = fig.add_subplot(1, 1, 1)

#     for i in range(len(config['classes_names'])):
#         index = total_y == i
#         if i == len(config['classes_names']) - 1 and config['is_anomaly_detection']:
#             color_index = all_color[-1]
#         else:
#             color_index = all_color[i]
#         ax.scatter(total_z[index][:,0], total_z[index][:,1],
#                            alpha = 0.4, c = np.tile(color_index,(len(total_z[index]),1)), s=40, 
#                            cmap = cmap, label = config['classes_names'][i] )
#     plt.legend(loc=2)
#     plt.gca().set_aspect('equal', adjustable='box')
#     fig.savefig(image_filename)
#     plt.close(fig)
#     plt.clf()
#     import gc;
#     gc.collect()
