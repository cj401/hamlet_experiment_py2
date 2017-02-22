import collections
import glob

import matplotlib.pyplot as plt
import numpy as np

from utilities import util

__author__ = 'clayton'


def read_results_file(file_path):
    data = list()
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line_comps = [ x.strip() for x in line.split(' ') ]
            if line_comps[0] != 'iteration':
                data.append( ( int(line_comps[0]), float(line_comps[1]) ) )
    return data


def extract_subarray(data, iteration_min):
    sub = list()
    for iter, val in data:
        if iter >= iteration_min:
            sub.append(val)
    return np.array(sub)


def summarize_run_as_scalar(run_results_file_paths, results_filename, iteration_min):
    data = list()
    for run_results_file_path in run_results_file_paths:
        raw_data = read_results_file(run_results_file_path + '/' + results_filename)
        sub = extract_subarray(raw_data, iteration_min)
        data.append(np.mean(sub))
    return data


def summarize_mean_serr(data):
    mean = np.mean(data)
    serr = np.std(data) / np.sqrt(len(data))
    return mean, serr


def print_dict(d):
    for key, val in d.iteritems():
        print key, ':', val


def save_ordered_dict(filepath, d):
    with open(filepath, 'w') as fout:
        fout.write('(\n')
        for key, val in d.iteritems():
            fout.write('({0}, {1}),\n'.format(key, val))
        fout.write(')')
    print 'DONE'


def read_ordered_dict(filepath):
    """
    FEARFULLY COMPREHENSIVE DISCLAIMER: FEAR COMPREHENSIVELY
    :param filepath:
    :return:
    """
    with open(filepath, 'r') as fin:
        lines = fin.readlines()
        kv_pairs_list = eval(''.join(lines))

    print kv_pairs_list

    od = collections.OrderedDict()
    for key, val in kv_pairs_list:
        od[key] = val
    return od


def read_results(results_root='../results/cocktail/',
                 exp_glob_pattern='noise_sd*_nocs_*',
                 results_filename='F1_score.txt',
                 iteration_min=800,
                 verbose=False):

    results_root_list = glob.glob(results_root + exp_glob_pattern)

    data_dict = collections.OrderedDict()

    for exp_root_path in results_root_list:
        comps = exp_root_path.split('/')
        exp_root = comps[-1]
        data_set_name = exp_root.split('_')[-1]
        noise_level = float(exp_root.split('_')[0][1:])

        if verbose: print exp_root, noise_level

        exp_model_paths = glob.glob(exp_root_path + '/*')
        for exp_model_path in exp_model_paths:
            exp_model = exp_model_path.split('/')[-1]

            if verbose: print '  ', exp_model

            exp_lt_paths = glob.glob(exp_model_path + '/LT_*')
            data_lt_runs = summarize_run_as_scalar(exp_lt_paths,
                                                   results_filename,
                                                   iteration_min)

            exp_nolt_paths = glob.glob(exp_model_path + '/noLT_*')
            data_nolt_runs = summarize_run_as_scalar(exp_nolt_paths,
                                                     results_filename,
                                                     iteration_min)

            data_dict[(data_set_name, noise_level, exp_model, 'LT')] = summarize_mean_serr(data_lt_runs)
            data_dict[(data_set_name, noise_level, exp_model, 'noLT')] = summarize_mean_serr(data_nolt_runs)

            if verbose:
                print '    ', exp_lt_paths
                print '    ', exp_nolt_paths

    return data_dict


def read_results_fhmm(results_root='../results/cocktail_fhmm/',
                      exp_glob_pattern='noise_sd*_nocs',
                      results_filename='f1.txt',
                      iteration_min=800,
                      verbose=True):
    """
    directory structure: h0.75_nocs/cp0/hmm/

    needed structure: data_set_name, noise_level, exp_model
    example:          cp0 , 0.75 , hmm
    """

    results_root_list = glob.glob(results_root + exp_glob_pattern)

    data_dict = collections.OrderedDict()

    for exp_root_path in results_root_list:
        comps = exp_root_path.split('/')
        exp_root = comps[-1]
        noise_level = float(exp_root.split('_')[0][1:])

        data_set_path_list = glob.glob(exp_root_path + '/*')
        for data_set_path in data_set_path_list:
            data_set_name = data_set_path.split('/')[-1]

            model_path_list = glob.glob(data_set_path + '/*')
            for model_path in model_path_list:
                model_name = model_path.split('/')[-1]

                results_file_path = model_path + '/{0}'.format(results_filename)
                results_data = read_results_file(results_file_path)

                data_indices = zip(*results_data)
                data = data_indices[1]

                data_new_indices = list()
                j = 0
                for i, datum in enumerate(data):  # range(len(results_data)):
                    if i < 10:
                        j += 1
                    else:
                        j += 10
                    data_new_indices.append((j, datum))

                sub = extract_subarray(data_new_indices, iteration_min)
                # print len(sub), ':', sub

                # summarize_run_as_scalar(run_results_file_paths, results_filename, iteration_min)
                data_mean_serr = summarize_mean_serr(sub)

                data_dict[(data_set_name, noise_level, model_name)] = data_mean_serr

                if verbose: print data_set_name, noise_level, model_name, data_mean_serr

    return data_dict


# ---------------------------------------------------------------------
# plot within figures set - h
# ---------------------------------------------------------------------


def plot_noise_results_for_single_model\
                (data_dict, model, stat_name='F1', data_set_name=None):

    plt.figure()
    ax = plt.gca()

    plot_noise_results_in_ax\
        (ax, data_dict, model, stat_name=stat_name, data_set_name=data_set_name)


def plot_noise_results_for_both_models\
                (data_dict, stat_name='F1', data_set_name=None):

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)

    fig.suptitle('Data set: {0}'.format(data_set_name))

    plot_noise_results_in_ax\
        (axs[0], data_dict, 'hmm_hdp_w0',
         stat_name=stat_name, data_set_name=data_set_name,
         title_data_name=False)
    plot_noise_results_in_ax\
        (axs[1], data_dict, 'hsmm_hdp_w0',
         stat_name=stat_name, data_set_name=data_set_name,
         title_data_name=False)


def plot_noise_results_in_ax\
                (ax, data_dict, model, stat_name='F1', data_set_name=None,
                 title_data_name=False):

    noise_levels = util.OrderedSet(sorted(zip(*data_dict.keys())[1]))

    def get_plot_data(model, lt):
        data = list()
        for noise_level in noise_levels:
            stat, sterr = data_dict[ ( data_set_name, noise_level, model, lt ) ]
            data.append( ( noise_level, stat, sterr ) )
        return data

    data_lt = get_plot_data(model, 'LT')
    data_nolt = get_plot_data(model, 'noLT')

    x, y, yerr = zip(*data_lt)
    ax.errorbar(x, y, yerr=yerr, fmt='-o', label='LT')
    x, y, yerr = zip(*data_nolt)
    ax.errorbar(x, y, yerr=yerr, fmt='-o', label='noLT')

    ax.set_xlabel('Precision')
    ax.set_ylabel(stat_name)
    if title_data_name:
        title = 'Data set \'{0}\', model \'{1}\''.format(data_set_name, model)
    else:
        title = 'Model \'{0}\''.format(model)
    ax.set_title(title)
    ax.legend(loc='lower right')


# ---------------------------------------------------------------------


def plot_lt_nolt_differences(data_dict, stat_name, data_set_name):

    # data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
    noise_levels = util.OrderedSet(sorted(zip(*data_dict.keys())[1]))

    def get_plot_data(model):
        data = list()
        for noise_level in noise_levels:
            stat_LT, sterr = data_dict[ ( data_set_name, noise_level, model, 'LT' ) ]
            stat_noLT, sterr = data_dict[ ( data_set_name, noise_level, model, 'noLT' ) ]
            diff = stat_LT - stat_noLT
            data.append( ( noise_level, diff ) )
        return data

    data_hmm = get_plot_data('hmm_hdp_w0')
    data_hsmm = get_plot_data('hsmm_hdp_w0')

    fig = plt.figure()
    ax = fig.gca()

    x, y = zip(*data_hmm)
    ax.plot(x, y, label='hmm')
    x, y = zip(*data_hsmm)
    ax.plot(x, y, label='hsmm')

    ax.set_title('Data set: {0}'.format(data_set_name))
    ax.set_xlabel('precision')
    ax.set_ylabel('{0} difference: LT - noLT'.format(stat_name))
    ax.legend(loc='lower right')


# ---------------------------------------------------------------------
# plot across figures sets - h
# ---------------------------------------------------------------------


def set_xticks_precision_to_sigma_ticks(precision_noise_level, ax):
    sigma = [ 1. / np.sqrt(h) for h in precision_noise_level ]
    ax.set_xticks(precision_noise_level, sigma)


def precision_to_sigma(precision_noise_level):
    return [ 1. / np.sqrt(h) for h in precision_noise_level ]


def plot_noise_results_across_data_sets\
                (data_dict, model, stat_name='F1', ax=None):
    """

    :param data_dict:
    :param model:
    :param stat_name:
    :param title_data_name:
    :param ax:
    :return:
    """

    data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
    noise_levels = util.OrderedSet(sorted(zip(*data_dict.keys())[1]))

    def get_plot_data(model, lt):
        data = list()
        for noise_level in noise_levels:
            across_dset_data = list()
            for data_set_name in data_set_names:
                stat, sterr = data_dict[ ( data_set_name, noise_level, model, lt ) ]
                across_dset_data.append(stat)
            mean, sterr = summarize_mean_serr(across_dset_data)
            data.append( ( noise_level, mean, sterr ) )
        return data

    data_lt = get_plot_data(model, 'LT')
    data_nolt = get_plot_data(model, 'noLT')

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x, y, yerr = zip(*data_lt)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='coral', fmt='--o', label='LT')
    x, y, yerr = zip(*data_nolt)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='darkcyan', fmt='--o', label='noLT')
    # 'darkcyan', 'darkturquoise', 'indianred'

    ax.grid(True)

    # set_xticks_precision_to_sigma_ticks(x, ax)

    ax.set_xlabel(r'IID Normal noise: $\sigma$')
    ax.set_ylabel(stat_name)

    model_name = model.split('_')[0].swapcase()
    title = 'Model {0}'.format(model_name)

    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower left')


def plot_lt_nolt_diff_across_data_sets(data_dict, stat_name, ax=None):

    data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
    noise_levels = util.OrderedSet(sorted(zip(*data_dict.keys())[1]))

    def get_plot_data(model):
        data = list()
        for noise_level in noise_levels:
            diffs = list()
            for data_set_name in data_set_names:
                stat_LT, sterr = data_dict[ ( data_set_name, noise_level, model, 'LT' ) ]
                stat_noLT, sterr = data_dict[ ( data_set_name, noise_level, model, 'noLT' ) ]
                diff = stat_LT - stat_noLT
                diffs.append(diff)
            mean_diffs, sterr_diffs = summarize_mean_serr(diffs)
            data.append( ( noise_level, mean_diffs, sterr_diffs ) )
        return data

    data_hmm = get_plot_data('hmm_hdp_w0')
    data_hsmm = get_plot_data('hsmm_hdp_w0')

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x, y, yerr = zip(*data_hmm)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='blue', fmt='--o', label='HMM')
    x, y, yerr = zip(*data_hsmm)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='indianred', fmt='--o', label='HSMM')
    # 'orange'

    ax.grid(True)

    ax.set_xlim((0.2, 1.5))  # max(x)

    # set_xticks_precision_to_sigma_ticks(x, ax)

    ax.set_title('LT - noLT {0} Differences'.format(stat_name), fontsize=10)
    ax.set_xlabel(r'IID Normal noise: $\sigma$')
    ax.set_ylabel('{0} difference: LT - noLT'.format(stat_name))
    ax.legend(loc='upper right')

    return ax


def plot_all_noise_results_across_data_sets(data_dict, stat_name='F1'):

    mydpi = 96
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,
                            figsize=(1300/mydpi, 500/mydpi), dpi=mydpi)

    data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
    fig.suptitle('Comparison of LT vs noLT across emission noise levels (as precision), '
                 + '{0} figures sets at each noise level'.format(len(data_set_names)),
                 y=1.0)

    plot_noise_results_across_data_sets\
        (data_dict, 'hmm_hdp_w0', stat_name=stat_name, ax=axs[0])
    plot_noise_results_across_data_sets\
        (data_dict, 'hsmm_hdp_w0', stat_name=stat_name, ax=axs[1])
    plot_lt_nolt_diff_across_data_sets\
        (data_dict, stat_name, ax=axs[2])

    fig.subplots_adjust(wspace=0.4)

    plt.savefig('figures/F1_LTvsnoLT_over_noise.pdf', format='pdf')

    # plt.tight_layout(w_pad=0.2)  # pad=0.4, w_pad=0.5, h_pad=1.0


def plot_all_noise_results_across_data_sets_nodiff(data_dict, stat_name='F1'):

    mydpi = 96
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True,
                            figsize=(1300/mydpi, 500/mydpi), dpi=mydpi)

    data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
    fig.suptitle('Comparison of LT vs noLT across emission noise levels (as precision), '
                 + '{0} figures sets at each noise level'.format(len(data_set_names)),
                 y=1.0)

    plot_noise_results_across_data_sets\
        (data_dict, 'hmm_hdp_w0', stat_name=stat_name, ax=axs[0])
    plot_noise_results_across_data_sets\
        (data_dict, 'hsmm_hdp_w0', stat_name=stat_name, ax=axs[1])
    #plot_lt_nolt_diff_across_data_sets\
    #    (data_dict, stat_name, ax=axs[2])

    fig.subplots_adjust(wspace=0.4)

    plt.savefig('figures/F1_LTvsnoLT_over_noise_no_diff.pdf', format='pdf')

    # plt.tight_layout(w_pad=0.2)  # pad=0.4, w_pad=0.5, h_pad=1.0


# ---------------------------------------------------------------------
# factorial hmm figures
# ---------------------------------------------------------------------


def plot_fhmm_results(data_dict, stat_name='F1', ax=None, show_p=False):

    def get_plot_data(model):
        data_set_names = util.OrderedSet(sorted(zip(*data_dict.keys())[0]))
        noise_levels = util.OrderedSet(sorted(zip(*data_dict.keys())[1]))
        data = list()
        for noise_level in noise_levels:
            across_dset_data = list()
            for data_set_name in data_set_names:
                stat, sterr = data_dict[ ( data_set_name, noise_level, model ) ]
                across_dset_data.append(stat)
            mean, sterr = summarize_mean_serr(across_dset_data)
            data.append( ( noise_level, mean, sterr ) )
        return data

    data_fhmm = get_plot_data('hmm')
    data_fhsmm = get_plot_data('hsmm')

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x, y, yerr = zip(*data_fhmm)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='coral', fmt='--o', label='FHMM')
    x, y, yerr = zip(*data_fhsmm)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='darkcyan', fmt='--o', label='FHSMM')
    # 'darkcyan', 'darkturquoise', 'indianred'

    ax.grid(True)

    # set_xticks_precision_to_sigma_ticks(x, ax)

    ax.set_xlabel(r'IID Normal noise: $\sigma$')
    ax.set_ylabel(stat_name)

    title = 'Factorial HMM'

    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower left')

    if show_p:
        plt.show()


'''
data_dict = read_results_fhmm(results_root='../results/cocktail_fhmm/',
                              exp_glob_pattern='noise_sd*_nocs',
                              results_filename='f1.txt',
                              iteration_min=800,
                              verbose=False)

#for k,v in data_dict.iteritems():
#    print k, ':', v

plot_fhmm_results(data_dict, show_p=True)
'''


# ---------------------------------------------------------------------
# LT, noLT and FHMM


def plot_lt_nolt_fhmm_results(data_lt_dict, data_factorial_dict, model, stat_name='F1',
                              ax=None, show_p=False, factorial_type='hmm'):

    def get_plot_fhmm_data(model):
        data_set_names = util.OrderedSet(sorted(zip(*data_factorial_dict.keys())[0]))
        noise_levels = util.OrderedSet(sorted(zip(*data_factorial_dict.keys())[1]))
        data = list()
        for noise_level in noise_levels:
            across_dset_data = list()
            for data_set_name in data_set_names:
                stat, sterr = data_factorial_dict[ ( data_set_name, noise_level, model ) ]
                across_dset_data.append(stat)
            mean, sterr = summarize_mean_serr(across_dset_data)
            data.append( ( noise_level, mean, sterr ) )
        return data

    def get_plot_lt_data(model, lt):
        data_set_names = util.OrderedSet(sorted(zip(*data_lt_dict.keys())[0]))
        noise_levels = util.OrderedSet(sorted(zip(*data_lt_dict.keys())[1]))
        data = list()
        for noise_level in noise_levels:
            across_dset_data = list()
            for data_set_name in data_set_names:
                stat, sterr = data_lt_dict[ ( data_set_name, noise_level, model, lt ) ]
                across_dset_data.append(stat)
            mean, sterr = summarize_mean_serr(across_dset_data)
            data.append( ( noise_level, mean, sterr ) )
        return data

    data_fhmm = get_plot_fhmm_data(factorial_type)
    # data_fhsmm = get_plot_fhmm_data('hsmm')

    data_lt = get_plot_lt_data(model, 'LT')
    data_nolt = get_plot_lt_data(model, 'noLT')

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x, y, yerr = zip(*data_lt)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='coral', fmt='--o', label='LT')
    x, y, yerr = zip(*data_fhmm)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='darkcyan', fmt='--o', label='Fact{0}'.format(factorial_type.swapcase()))
    x, y, yerr = zip(*data_nolt)
    x = precision_to_sigma(x)
    ax.errorbar(x, y, yerr=yerr, color='indianred', fmt='--o', label='noLT')

    # x, y, yerr = zip(*data_fhsmm)
    # x = precision_to_sigma(x)
    # ax.errorbar(x, y, yerr=yerr, color='darkcyan', fmt='--o', label='FHSMM')
    # 'darkcyan', 'darkturquoise', 'indianred'

    ax.grid(True)

    # set_xticks_precision_to_sigma_ticks(x, ax)

    ax.set_xlabel(r'IID Normal noise: $\sigma$')
    ax.set_ylabel(stat_name)

    title = 'Factorial HMM and LT/noLT {0}'.format(model)

    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right')

    if show_p:
        plt.show()


def plot_all_noise_results_across_data_sets_with_fhmm\
                (data_lt_dict, data_factorial_dict, stat_name='F1',
                 show_diff=True,
                 show_p=True,
                 save_p=False):

    if show_diff:
        nsubplots = 3
    else:
        nsubplots = 2

    mydpi = 96
    fig, axs = plt.subplots(nrows=1, ncols=nsubplots, sharex=True,
                            figsize=(1300/mydpi, 500/mydpi), dpi=mydpi)

    data_set_names = util.OrderedSet(sorted(zip(*data_lt_dict.keys())[0]))
    fig.suptitle('Comparison of LT vs noLT across emission noise levels (as precision), '
                 + '{0} figures sets at each noise level'.format(len(data_set_names)),
                 y=1.0)

    plot_lt_nolt_fhmm_results\
        (data_lt_dict, data_factorial_dict, 'hmm_hdp_w0', factorial_type='hmm',
         stat_name=stat_name, ax=axs[0])
    plot_lt_nolt_fhmm_results\
        (data_lt_dict, data_factorial_dict, 'hsmm_hdp_w0', factorial_type='hsmm',
         stat_name=stat_name, ax=axs[1])

    if show_diff:
        plot_lt_nolt_diff_across_data_sets\
            (data_lt_dict, stat_name, ax=axs[2])

    fig.subplots_adjust(wspace=0.4)

    if save_p:
        plt.savefig('figures/F1_LT-noLT-Factorial_over_noise.pdf', format='pdf')

    if show_p:
        plt.show()


data_factorial_dict = read_results_fhmm(results_root='../results/cocktail_fhmm/',
                                        exp_glob_pattern='noise_sd*_nocs',
                                        results_filename='f1.txt',
                                        iteration_min=800,
                                        verbose=False)
data_lt_dict = read_ordered_dict('figures/results_h_F1_dict_20150712.txt')
#for k,v in data_dict.iteritems():
#    print k, ':', v

# plot_lt_nolt_fhmm_results(data_lt_dict, data_factorial_dict, 'hmm_hdp_w0', stat_name='F1', show_p=True)

plot_all_noise_results_across_data_sets_with_fhmm(data_lt_dict, data_factorial_dict, show_diff=True, save_p=True)


# ---------------------------------------------------------------------
# scripts
# ---------------------------------------------------------------------


# extract figures from venti results
'''
data_dict = read_results(results_root='../results/cocktail/',
                         exp_glob_pattern='noise_sd*_nocs_*',
                         results_filename='F1_score.txt',
                         iteration_min=800,
                         verbose=True)

save_ordered_dict('results_h_F1_dict.txt', data_dict)
'''


# ---------------

# process and plot extracted figures


# read figures
#data_dict = read_ordered_dict('figures/results_h_F1_dict_20150712.txt')
#print_dict(data_dict)


'''
# plot single model
plot_noise_results_for_single_model\
    (data_dict, 'hmm_hdp_w0', stat_name='F1', data_set_name='cp0')

plot_noise_results_for_single_model\
    (data_dict, 'hsmm_hdp_w0', stat_name='F1', data_set_name='cp0')
'''


'''
# plot paired models
plot_noise_results_for_both_models\
    (data_dict, stat_name='F1', data_set_name='cp0')
plot_noise_results_for_both_models\
    (data_dict, stat_name='F1', data_set_name='cp1')
plot_noise_results_for_both_models\
    (data_dict, stat_name='F1', data_set_name='cp2')
'''

'''
# plot LT - noLT differences, but for single "figures set"
for data_set_name in ['cp0', 'cp1', 'cp2']:
    plot_lt_nolt_differences(data_dict, 'F1', data_set_name)
'''

'''
# plot LT - noLT differences across figures sets
plot_lt_nolt_diff_across_data_sets(data_dict, 'F1')
plot_noise_results_across_data_sets(data_dict, 'hmm_hdp_w0', stat_name='F1')
plot_noise_results_across_data_sets(data_dict, 'hsmm_hdp_w0', stat_name='F1')
'''


# plot_all_noise_results_across_data_sets(data_dict, 'F1')

# plot_all_noise_results_across_data_sets_nodiff(data_dict, 'F1')

# plt.show()


