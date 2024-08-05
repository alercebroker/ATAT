import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gc

def plot_bar(score, label_name, bar_name, n_classes, save_path):
    plt.close()
    fig  = plt.figure()
    plt.bar(label_name, np.eye(n_classes)[score].sum(0)) 
    plt.ylabel('Counts')
    plt.xlabel(bar_name)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_histogram(score, label_name, hist_name, save_path, normalized = False, plot_name = ''):
    plt.close()
    fig  = plt.figure()
    plt.hist(score, density=False, bins = 20, 
               histtype='step', linewidth=1, color='blue', label = label_name) 
    plt.ylabel('Counts')
    plt.xlabel(hist_name)
    if plot_name != '':
      plt.title(plot_name)
    if normalized:
      plt.xlim([-.1,1.1])
    plt.legend(loc=2)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_2histogram(score, label_name, score_2, label_name_2, hist_name, save_path, normalized = False, 
                           fixed = False, centered = -1, not_outliers = False):
    plt.close()
    fig  = plt.figure()
    bins = 20
    minlim, maxlim = -.1, 1.1
    if normalized and fixed:
      bins = np.linspace(-.1, 1.1, 20)
    minlim    = np.min(( score.min(), score_2.min()  ))
    maxlim    = np.max(( score.max(), score_2.max()  ))
    bins      = np.linspace(minlim, maxlim , 20)
    # centered should be 
    if centered != -1:
      mean   = np.mean(score) if centered == 0 else np.mean(score_2)
      std    = np.std(score)  if centered == 0 else np.std(score_2)
      maxlim = np.min((maxlim, mean + 3 * std))
      minlim = np.max((minlim, mean - 3 * std))
      bins   = np.linspace(minlim, maxlim, 20)

    if not_outliers:
      score_total = np.concatenate([score, score_2])
      mean        = np.mean(score_total) 
      std         = np.std(score_total)
      maxlim      = mean + 3 * std
      minlim      = mean - 3 * std
      fscore_max  = (score_total > maxlim)
      fscore_min  = (score_total < minlim)
      nfscore     = ~(fscore_max + fscore_min)
      max_value   = score_total[nfscore].max()
      min_value   = score_total[nfscore].min()
      score_total[fscore_max] = max_value
      score_total[fscore_min] = min_value

      bins    = np.linspace(min_value, max_value, 20)
      score   = score_total[:len(score)]
      score_2 = score_total[-len(score_2):]

    plt.hist(score, density=False, bins = bins, 
               histtype='step', linewidth=1, color='blue', label = label_name) 
    plt.hist(score_2, density=False, bins = bins, 
               histtype='step', linewidth=1, color='orange', label = label_name_2)

    plt.ylabel('Counts')
    plt.xlabel(hist_name)
    if centered != -1 or normalized:
      plt.xlim([minlim, maxlim])
    plt.legend(loc=2)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_3histogram(score, label_name, score_2, label_name_2,
                     score_3, label_name_3, hist_name, save_path, density = True):
    plt.close()
    fig  = plt.figure()
    bins = 20
    minlim    = np.min(( score.min(), score_2.min(), score_3.min() ))
    maxlim    = np.max(( score.max(), score_2.max(), score_3.min() ))
    bins      = np.linspace(minlim, maxlim , 20)
    plt.hist(score, density = density, bins = bins, 
               histtype='step', linewidth=1, color='blue', label = label_name) 
    plt.hist(score_2, density = density, bins = bins, 
               histtype='step', linewidth=1, color='green', label = label_name_2)
    plt.hist(score_3, density = density, bins = bins, 
               histtype='step', linewidth=1, color='orange', label = label_name_3)
    plt.ylabel('distribution')
    plt.xlabel(hist_name)
    plt.legend(loc=2)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()

def plot_2histogram_notoutliers(score, label_name, score_2, label_name_2, hist_name, save_path, ):
    plt.close()
    fig  = plt.figure()
    bins = np.linspace(-.1, 1.1, 20)
    # centered should be 
    score_total = np.concatenate([score, score_2])
    mean        = np.mean(score_total) 
    std         = np.std(score_total)
    maxlim      = mean + 3 * std
    minlim      = mean - 3 * std
    fscore_max  = (score_total > maxlim)
    fscore_min  = (score_total < minlim)
    nfscore     = ~(fscore_max + fscore_min)
    max_value   = score_total[nfscore].max()
    min_value   = score_total[nfscore].min()
    score_total[fscore_max] = max_value
    score_total[fscore_min] = min_value

    score_total_norm = (score_total - score_total.min() )/(score_total.max() - score_total.min())
    new_score   = score_total_norm[:len(score)]
    new_score_2 = score_total_norm[-len(score_2):]

    plt.hist(new_score, density=False, bins = bins, 
               histtype='step', linewidth=1, color='blue', label = label_name) 
    plt.hist(new_score_2, density=False, bins = bins, 
               histtype='step', linewidth=1, color='orange', label = label_name_2)
    plt.ylabel('Counts')
    plt.xlabel(hist_name)
    plt.xlim([-.1,1.1])

    plt.legend(loc=2)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()


def plot_multi_histogram(score, labels, score_2, label_name_2, hist_name, save_path, class_names, not_outliers = False):
    all_cmaps = np.array(['blue', 'green', 'red', 'purple', 'gray', 'greenyellow',
                'magenta','orange', 'yellow', 'sienna', 'black','cyan'])
    plt.close()
    fig  = plt.figure()
    #bins = 20
    min_value = np.min(( score.min(), score_2.min()  ))
    max_value = np.max(( score.max(), score_2.max()  ))
    bins = np.linspace(min_value, max_value , 20)

    if not_outliers:
      score_total = np.concatenate([score, score_2])
      mean        = np.mean(score_total) 
      std         = np.std(score_total)
      maxlim      = mean + 3 * std
      minlim      = mean - 3 * std
      fscore_max  = (score_total > maxlim)
      fscore_min  = (score_total < minlim)
      nfscore     = ~(fscore_max + fscore_min)
      max_value   = score_total[nfscore].max()
      min_value   = score_total[nfscore].min()
      score_total[fscore_max] = max_value
      score_total[fscore_min] = min_value

      #score_total_norm  = (score_total - score_total.min() )/(score_total.max() - score_total.min())
      bins = np.linspace(min_value, max_value, 20)
      score         = score_total[:len(score)]
      score_2       = score_total[-len(score_2):]
    # centered should be
    classes = np.unique(labels)
    for i in classes:
      index = labels == i
      plt.hist(score[index], density=False, bins = bins, 
               histtype='step', linewidth=1, color = all_cmaps[i], label = class_names[i]) 
    plt.hist(score_2, density=False, bins = bins, 
               histtype='step', linewidth=1, color = all_cmaps[-1], label = class_names[-1] + ' (outlier)')

    plt.xlim([min_value, max_value])
    plt.ylabel('Counts')
    plt.xlabel(hist_name)
    plt.legend(loc=2)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
    gc.collect()

# def plot_oneout(metrics_ext, keys, labels, config, plot_type = 'anomaly', folder_root = ''):
#   dict_name = {'anomaly':        {'score1': 'class %d',
#                                   'score2': 'not class %d',
#                                   'score3': 'outliers %d'},
#                'semisupervised': {'score1': 'class %d (labeled)',
#                                   'score2': 'class %d (unlabeled)',
#                                   'score3': 'not class %d (unlabeled)'} } 
#   if plot_type == 'semisupervised':
#     metrics_l, metrics_u = metrics_ext
#     labels_l,  labels_u  = labels

#   for key in keys:
#     folder_root_class  = '%s/oneclass_out_%s' % (folder_root, key)
#     if not os.path.isdir(folder_root_class):
#         os.mkdir(folder_root_class)
#     for i in range(config['n_classes']):
#       save_path = '%s/histogram_%d.jpg' % (folder_root_class, i)
#       if plot_type == 'anomaly':
#         score1             = metrics_ext[key][labels == i]
#         score2             = metrics_ext[key][(labels != i) * (labels != config['n_classes']) ]
#         score3             = metrics_ext[key][labels == config['n_classes']]
#       if plot_type == 'semisupervised':
#         score1             = metrics_l[key][labels_l == i]
#         score2             = metrics_u[key][labels_u == i]
#         score3             = metrics_u[key][labels_u != i]
#       plot_3histogram(score1, dict_name[plot_type]['score1'] % i,
#                       score2, dict_name[plot_type]['score2'] % i,
#                       score3, dict_name[plot_type]['score3'] % i,
#                       pallh.dict_name_title[key],
#                       save_path, density = True)