import os
import itertools

__author__ = 'clayton'

"""
Dimensions of different experiments

(1) dynamics:                 *None     HMM     HSMM
(2) transition prior:         Flat (symmetric_dirichlet)     HDP (hdp)
(3) local transition: binary_state_model b_lambda
exponential dist (big num, high peak around 0
(don't think there are local transitions - no regard for locality), lower number )    0    [>0 unbounded]    1
smaller b_lambda is, the more freedom for model to prioritize local transitions
big number says don't use it, small number says use it if you want.
categorically not use, set to zero

(4) emission:                 normal   probit
probit: the refactoring parameters
Probit_noise_model (won't have any parameters - will have weights)

new module:
IID_normal_weight_prior
one param: sigma2_0

alternative Known_weights (currently flag at :experiment)
takes filename

default: <data_file>.weights



(5) emission separability:    at least 3 levels

Separability of states in likelihood space
(how far apart is each emission distribution (given latent state))
normal: ratio of weights to noise (two parameters)
probit: just the weights

Data:
() Synthetic
() Power
() Bach
() Bio
"""

#  import ast  # for ast.literal_eval() -- only evaluates strings that turn into basic python types


class astr():

    def __init__(self, main_str, abbrev=None):
        self.str = main_str
        if not abbrev:
            self.abbrev = main_str
        else:
            self.abbrev = abbrev

    def __repr__(self):
        return '(\'{0}\', \'{1}\')'.format(self.str, self.abbrev)

    def __str__(self):
        return self.str

    def __eq__(self, other):
        return self.str == other


def read_spec(spec_file):
    """
    SECURITY WARNING: don't eval any file that you don't know/trust
    """
    s = open(spec_file, 'r').read()
    # print s
    return eval(s)


def generate_name(spec, number=None):

    names = []

    if 'spec_prefix' in spec:
        names.append(spec['spec_prefix'])

    names.append(spec['dynamics'].abbrev)
    names.append(spec['transition_prior'].abbrev)
    names.append(spec['noise'].abbrev)

    if 'IID_normal_weights:sigma2_0' in spec:
        names.append('s{0}'.format(spec['IID_normal_weights:sigma2_0']))
    if 'Binary_state_model:lambda' in spec:
        names.append('L{0}'.format(spec['Binary_state_model:lambda']))
    if 'Binary_state_model:b_lambda' in spec:
        names.append('bL{0}'.format(spec['Binary_state_model:b_lambda']))
    if 'spec_name' in spec:
        names.append(spec['spec_name'])
    if number is not None:
        names.append(number)

    names = map(lambda v: '{0}'.format(v), names)

    return '_'.join(names)


### Generate Parameters


def param_experiment(spec):
    lines = list()
    lines.append('\n// Experiment parameters')
    if 'data_file_name' in spec:
        lines.append(':experiment data_file_name {0}'.format(spec['data_file_name']))
    if 'iterations' in spec:
        lines.append(':experiment iterations {0}'.format(spec['iterations']))
    if 'random_seed' in spec:
        lines.append(':experiment random_seed {0}'.format(spec['random_seed']))
    if 'do_ground_truth_eval' in spec:
        lines.append(':experiment do_ground_truth_eval {0}'.format(spec['do_ground_truth_eval']))
    if 'do_test_set_eval' in spec:
        lines.append(':experiment do_test_set_eval {0}'.format(spec['do_test_set_eval']))
    return '\n'.join(lines)


def param_generate(spec):
    lines = list()
    if ('sequence_length' in spec) or ('observables_dimension' in spec):
        lines.append('\n// Generation parameters')
        lines.append(':generate sequence_length {0}'.format(spec['sequence_length']))
        if 'test_sequence_length' in spec:
            lines.append(':generate test_sequence_length {0}'.format(spec['test_sequence_length']))
        lines.append(':generate observables_dimension {0}'.format(spec['observables_dimension']))
    return '\n'.join(lines)


def param_module(spec):
    lines = list()
    lines.append('\n// Model configuration')
    lines.append(':MODULE TRANSITION_PRIOR {0}'.format(spec['transition_prior']))
    lines.append(':MODULE DYNAMICS {0}'.format(spec['dynamics']))
    lines.append(':MODULE STATE {0}'.format(spec['state']))
    lines.append(':MODULE EMISSION {0}'.format(spec['emission']))
    lines.append(':MODULE NOISE {0}'.format(spec['noise']))
    lines.append(':MODULE WEIGHTS_PRIOR {0}'.format(spec['weights_prior']))
    return '\n'.join(lines)


# noise = Normal
def param_normal_emission_model(spec):
    lines = list()
    lines.append('\n// Normal_emission_model')
    lines.append('Normal_noise_model a_h {0}'.format(spec['Normal_noise_model:a_h']))
    lines.append('Normal_noise_model b_h {0}'.format(spec['Normal_noise_model:b_h']))
    return '\n'.join(lines)


# weights_prior = IID_normal
def param_weights_prior(spec):
    lines = list()
    lines.append('\n// Weights model - IID_normal')
    lines.append('IID_normal_weights sigma2_0 {0}'.format(spec['IID_normal_weights:sigma2_0']))
    return '\n'.join(lines)


# weights_prior = Known
def param_weights_known(spec):
    lines = list()
    lines.append('\n// Weights model - Known')
    lines.append('Known_weights weights_file {0}'.format(spec['Known_weights:weights_file']))
    return '\n'.join(lines)


# transition_prior = Symmetric_dirichlet
def param_dirichlet_hyperprior(spec):
    lines = list()
    lines.append('\n// Dirichlet hyperprior - prior over transition matrix')

    if 'Dirichlet_hyperprior:alpha' in spec:
        lines.append('Dirichlet_hyperprior alpha {0}'.format(spec['Dirichlet_hyperprior:alpha']))
        if 'Dirichlet_hyperprior:beta_file' in spec:
            lines.append('Dirichlet_hyperprior beta_file {0}'.format(spec['Dirichlet_hyperprior:beta_file']))
    else:
        lines.append('Dirichlet_hyperprior a_alpha {0}'.format(spec['Dirichlet_hyperprior:a_alpha']))
        lines.append('Dirichlet_hyperprior b_alpha {0}'.format(spec['Dirichlet_hyperprior:b_alpha']))

    return '\n'.join(lines)


# transition_prior = HDP
def param_hdp_hyperprior(spec):
    lines = list()
    lines.append('\n// HDP_hyperprior')

    if 'HDP_hyperprior:gamma' in spec:
        lines.append('HDP_hyperprior gamma {0}'.format(spec['HDP_hyperprior:gamma']))
    else:
        lines.append('HDP_hyperprior a_gamma {0}'.format(spec['HDP_hyperprior:a_gamma']))
        lines.append('HDP_hyperprior b_gamma {0}'.format(spec['HDP_hyperprior:b_gamma']))

    if 'HDP_hyperprior:alpha' in spec:
        lines.append('HDP_hyperprior alpha {0}'.format(spec['HDP_hyperprior:alpha']))
    else:
        lines.append('HDP_hyperprior a_alpha {0}'.format(spec['HDP_hyperprior:a_alpha']))
        lines.append('HDP_hyperprior b_alpha {0}'.format(spec['HDP_hyperprior:b_alpha']))

    return '\n'.join(lines)


def param_known_transition_matrix(spec):
    lines = list()
    lines.append('\n// Known_transition_matrix')
    lines.append('Known_transition_matrix transition_matrix_file {0}'.format(spec['Known_transition_matrix:transition_matrix_file']))
    lines.append('Known_transition_matrix initial_distribution_file {0}'.format(spec['Known_transition_matrix:initial_distribution_file']))
    return '\n'.join(lines)


# dynamics = HSMM
def param_semimarkov_transition_model(spec):
    lines = list()
    lines.append('\n// Semimarkov_transition_model')
    lines.append('Semimarkov_transition_model a_omega {0}'.format(spec['Semimarkov_transition_model:a_omega']))
    lines.append('Semimarkov_transition_model b_omega {0}'.format(spec['Semimarkov_transition_model:b_omega']))
    return '\n'.join(lines)


def param_binary_state_model(spec):
    lines = list()
    lines.append('\n// Binary_state_model')
    lines.append('Binary_state_model J {0}'.format(spec['Binary_state_model:J']))
    lines.append('Binary_state_model D {0}'.format(spec['Binary_state_model:D']))

    if 'Binary_state_model:state_matrix_file' in spec:
        lines.append('Binary_state_model state_matrix_file {0}'.format(spec['Binary_state_model:state_matrix_file']))
    else:
        lines.append('Binary_state_model a_mu {0}'.format(spec['Binary_state_model:a_mu']))
        lines.append('Binary_state_model b_mu {0}'.format(spec['Binary_state_model:b_mu']))

    if 'Binary_state_model:lambda' in spec:
        lines.append('Binary_state_model lambda {0}'.format(spec['Binary_state_model:lambda']))
    else:
        lines.append('Binary_state_model b_lambda {0}'.format(spec['Binary_state_model:b_lambda']))

    return '\n'.join(lines)


def generate_parameter_file(spec, filepath):

    lines = list()
    param_file_name = generate_name(spec)

    spec['data_file_name'] = param_file_name

    lines.append(param_experiment(spec))
    lines.append(param_generate(spec))
    lines.append(param_module(spec))

    if spec['noise'] == 'Normal':
        lines.append(param_normal_emission_model(spec))

    if spec['weights_prior'] == 'IID_normal':
        lines.append(param_weights_prior(spec))
    if 'Known_weights:weights_file' in spec:
        lines.append(param_weights_known(spec))

    if spec['transition_prior'] == 'Dirichlet':
        lines.append(param_dirichlet_hyperprior(spec))
    elif spec['transition_prior'] == 'HDP':
        lines.append(param_hdp_hyperprior(spec))
    else:
        lines.append(param_known_transition_matrix(spec))

    if spec['dynamics'] == 'HSMM':
        lines.append(param_semimarkov_transition_model(spec))

    lines.append(param_binary_state_model(spec))

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(filepath + param_file_name + '.config', 'w') as fout:
        fout.write('\n'.join(lines))


def run_filepath_subdir_rules(filepath_subdir_rules, spec, filepath):
    for (param, value), subdir in filepath_subdir_rules:
        # print '>>', param, value, spec[param], spec[param] == value
        if spec[param] == value:
            return filepath + subdir


def generate_parameters(spec, filepath='', filepath_subdir_rules=None, test=True):
    """
    Given a parameter file generator spec, produces the cartesian product of
    all parameter values that are specified as tuples of values; all other
    parameter values in the spec remain constant
    :param spec: parameter file generator specification
    :param filepath: path to where generated parameter files will be saved
    :return:
    """

    if test: print 'RUNNING in TEST MODE'

    # extract parameter value lists from spec
    products = []
    for key, val in spec.iteritems():
        if isinstance(val, (list, tuple)):
            products.append((key, val))

    # iterate over cartesian product of parameter lists
    keys, valists = zip(*products)
    for valist in itertools.product(*valists):

        # make copy of general spec and set ground values
        ground_spec = spec.copy()
        for key, ground_val in zip(keys,valist):
            ground_spec[key] = ground_val

        augmented_filepath = run_filepath_subdir_rules(filepath_subdir_rules,
                                                       ground_spec,
                                                       filepath)

        if test:
            print augmented_filepath, ground_spec
        else:
            generate_parameter_file(ground_spec, augmented_filepath)

    print 'generate_parameters({0}) DONE.'.format(filepath)


### script

# spec = read_spec('specs/gen_20150521.spec')
# print 'spec:', spec



print 'SCRIPTS TURNED OFF'


# DATA

'''
generate_parameters(spec=read_spec('specs/gen_20150527.spec'),
                    filepath='parameters/figures/',
                    filepath_subdir_rules=[(('noise', 'Normal'), 'normal/'),
                                           (('noise', 'Probit'), 'probit/')],
                    test=False)
'''

# EXPERIMENT

#'''
generate_parameters(spec=read_spec('specs/exp_20150602-noise_sd.spec'),
                    filepath='parameters/exp/',
                    filepath_subdir_rules=[(('noise', 'Normal'), 'normal/'),
                                           (('noise', 'Probit'), 'probit/')],
                    test=False)
#'''
