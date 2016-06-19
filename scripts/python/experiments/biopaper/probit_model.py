from __future__ import print_function

__author__ = 'clayton'

"""
This file depends on statsmodels: https://github.com/statsmodels/statsmodels/


Getting statsmodels:

Recommend using pip:
$ pip install statsmodels
"""

# http://statsmodels.sourceforge.net/devel/generated/statsmodels.discrete.discrete_model.Probit.html

import numpy as np
import statsmodels.api as sm
import glob


def read_data(filepath, verbose=False):
    """
    Read text file of binary vector time series where
    rows are one step in time, columns are binary vector elements,
    elements are separated by space.
    Designed to read files like obs.txt or states.txt
    :param filepath: path to figures file.
    :param verbose:
    :return:
    """

    def line_to_values_list(line):
        return [ int(elm) for elm in line.strip().split(' ') ]

    data = list()
    with open(filepath, 'r') as fin:
        for line in fin.readlines():
            if line[0] != '#':
                data.append(line_to_values_list(line))
    data = np.array(data)

    if verbose:
        print('{0}'.format(filepath))
        print('    figures shape: {0}'.format(data.shape))
        print('    figures[0,:] shape: {0}'.format(data[0, :].shape))
        ones = np.sum(data)
        total = np.multiply(*data.shape)
        print("    sparsity: {0} 1's out of {1} = {2}"
              .format(ones, total, float(ones) / float(total)))

    return data


# ---------------------------------------------------


# read_data('../figures/biology_papers/20150709_01/figures/p01/states.txt')
# read_data('../figures/biology_papers/20150709_01/figures/p01/obs.txt')


# ---------------------------------------------------

def train_probit_across_obs(obs, states, alpha=10.0, verbosity=0):
    """

    :param states: n x k matrix, k = num elements in latent binary vector
    :param obs: n x j matrix, j = num elements in observation/feature binary vector
    :return: k+1 x j weight matrix
    """
    states = sm.tools.add_constant(states, prepend=False)
    weight_columns = np.zeros((states.shape[1], obs.shape[1]))
    for column in range(obs.shape[1]):

        if verbosity == 1:
            print('{0} '.format(column), end="",)
            if column % 10 == 0: print()

        obs_column = obs[:, column]

        print('num 1\'s in obs_column: sum(obs_column)= {0}'.format(np.sum(obs_column)))

        if verbosity > 1: print('obs_column {0}: {1}'.format(obs_column.shape, obs_column))

        probit_model = sm.Probit(obs_column, states)
        fresult = probit_model.fit_regularized(method='l1', alpha=alpha)
        weight_columns[:, column] = fresult.params

        if verbosity > 1:
            print('fresult.params: {0}'.format(fresult.params))
            print('weight_columns:\n{0}'.format(weight_columns))

    return weight_columns


def test_train_probit_across_obs():
    states = np.array([[0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 1.],
                       [1., 1., 0.],
                       [0., 0., 0.],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 1.]])
    obs = np.array([[1., 0., 0., 1., 0., 1., 0., 0., 1., 0.],
                    [1., 1., 0., 1., 0., 1., 0., 0., 1., 0.],
                    [1., 0., 1., 1., 0., 1., 0., 0., 1., 0.]])
    obs = obs.T

    print('states {0}:\n{1}'.format(states.shape, states))
    print('obs {0}:\n{1}'.format(obs.shape, obs))

    weight_columns = train_probit_across_obs(obs, states, verbosity=2)

    print('weight_columns:\n{0}'.format(weight_columns))

# test_train_probit_across_obs()


def get_weight_matrix_for_paper(paper_path, alpha=10.0, save_path=None, verbose=True):

    if paper_path[-1] != '/':
        paper_path += '/'

    obs = read_data(paper_path + 'obs.txt', verbose=verbose)
    states = read_data(paper_path + 'states.txt', verbose=verbose)
    weights = train_probit_across_obs(obs, states, alpha=alpha, verbosity=1)

    if verbose:
        print('weights {0}'.format(weights.shape))
        for i, row in enumerate(weights):
            print('[{0}] {1}'.format(i, row))

    if save_path:
        if verbose: print('Saving weights to {0}'.format(save_path))
        with open(save_path, 'w') as fout:
            for row in weights:
                for elm in row:
                    fout.write(' {0}'.format(elm))

    if verbose: print('DONE')

    return weights


def get_weight_matrix_for_paper_all_data(data_root, data_list, save_path, alpha=15.0):
    if data_root[-1] != '/':
        data_root += '/'
    data_paths = glob.glob(data_root + 'p*')

    all_obs = None
    all_states = None
    for data_path in data_paths:
        data_dir_name = data_path.split('/')[-1]
        if data_dir_name in data_list:
            print('reading {0}'.format(data_dir_name))
            obs = read_data(data_path + '/obs.txt')
            states = read_data(data_path + '/states.txt')

            if all_obs is None:
                all_obs = obs
            else:
                all_obs = np.concatenate((all_obs, obs), axis=0)

            if all_states is None:
                all_states = states
            else:
                all_states = np.concatenate((all_states, states), axis=0)

    weights = train_probit_across_obs(all_obs, all_states, alpha=alpha, verbosity=1)

    # np.savetxt(save_path, weights, delimiter=' ')
    with open(save_path, 'w') as fout:
        for row in weights:
            for elm in row:
                fout.write('{0} '.format(elm))
            fout.write('\n')

    print('DONE.')



# get_weight_matrix_for_paper('../figures/biology_papers/20150710_02/p01/')

'''
get_weight_matrix_for_paper_all_data(data_root='../figures/biology_papers/20150710_02',
                                     data_list=['p{0:0>2}'.format(i+1) for i in range(14)],
                                     save_path='probit_weights_alpha15.txt',
                                     alpha=15.0)
'''


# ---------------------------------------------------


def test_probit():

    exog_data = np.array([[0., 1., 0.],
                          [0., 0., 1.],
                          [1., 0., 1.],
                          [1., 1., 0.],
                          [0., 0., 0.],
                          [0., 1., 0.],
                          [1., 0., 0.],
                          [0., 0., 0.],
                          [1., 0., 0.],
                          [0., 0., 1.]])
    endog_data = np.array([1., 0., 0., 1., 0., 1., 0., 0., 1., 0.])

    print('{0}'.format(exog_data))
    exog_data = sm.tools.add_constant(exog_data, prepend=False)
    print('{0}'.format(exog_data))

    print('\n'+'/'*80+'\n')

    probit_model = sm.Probit(endog_data, exog_data)

    print('probit_model.endog_names', probit_model.endog_names)
    print('probit_model.exog_names', probit_model.exog_names)

    print('\n'+'/'*80+'\n')

    print('probit_model.fit()')
    probit_result = probit_model.fit()

    print('\n'+'/'*80+'\n')

    print('Parameters: ', probit_result.params)

    print('\n'+'/'*80+'\n')

    print(probit_result.summary())

    print('\n'+'/'*80+'\n')

    print('probit_result.get_margeff()')
    margeff = probit_result.get_margeff()
    print(margeff.summary())

    print('\n'+'\\'*80+'\n')

    # --------------------

    print('probit_model.fit_regularized()')
    probit_result_l1 = probit_model.fit_regularized()

    print('\n'+'/'*80+'\n')

    print('Parameters: ', probit_result_l1.params)

    print('\n'+'/'*80+'\n')

    print(probit_result_l1.summary())

    print('\n'+'/'*80+'\n')

    print('probit_result_l1.get_margeff()')
    margeff = probit_result_l1.get_margeff()
    print(margeff.summary())

    print('\n'+'\\'*80+'\n')
