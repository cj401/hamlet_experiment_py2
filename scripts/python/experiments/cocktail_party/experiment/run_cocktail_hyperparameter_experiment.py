import sys
import os
import multiprocessing


# ----------------------------------------------------------------------
# Ensure <hamlet>/experiment/scripts/python/ in sys.path (if possible)
# ----------------------------------------------------------------------

def seq_equal(s1, s2):
    if len(s1) != len(s2):
        return False
    for e1, e2 in zip(s1, s2):
        if e1 != e2:
            return False
    return True


def print_sys_path():
    for i, path in enumerate(sys.path):
        print i, path


def find_path_context(target_path_components):
    for path in sys.path:
        path_components = path.split('/')
        if 'hamlet' in path_components:
            if seq_equal(target_path_components,
                         path_components[-len(target_path_components):]):
                return True
    return False


def optional_add_relative_path(current, parent, relative_path, verbose=False):
    """
    If executing in current directory and parent path is not in sys.path,
    then add parent path.
    :return:
    """
    if not find_path_context(parent):
        if find_path_context(current):
            parent_path = os.path.realpath(os.path.join(os.getcwd(), relative_path))
            if verbose:
                print 'NOTICE: experiment_tools.py'
                print '    Executing from:     {0}'.format(os.getcwd())
                print '    Adding to sys.path: {0}'.format(parent_path)
            sys.path.insert(1, parent_path)


optional_add_relative_path\
    (current=('scripts', 'python', 'experiments', 'cocktail_party', 'experiment'),
     parent=('scripts', 'python'),
     relative_path='../../../',
     verbose=True)



from run import experiment_tools


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
DATA_ROOT = experiment_tools.DATA_ROOT
PARAMETERS_ROOT = experiment_tools.PARAMETERS_ROOT
RESULTS_ROOT = experiment_tools.RESULTS_ROOT


# ----------------------------------------------------------------------
# Generate Parameter Specs
# ----------------------------------------------------------------------

def float2string(f):
    charlist = list('{0}'.format(f))
    for i, c in enumerate(charlist):
        if c == '.':
            charlist[i] = 'p'
    return ''.join(charlist)


class PSpec(object):
    """
    NOTE: Not to be confused with a experiment_tools.ParameterSpec
    that bookkeeps simple info about parameter source

    This class loads a parameter spec FILE and permits changes to module/var values and saving
    Used for programmatically GENERATING parameter spec files for experiments
    (more general than hyperparameter experiments)
    """

    def __init__(self, source_path):
        self.source_path = source_path
        self.raw = dict()
        self.num_lines = 0
        self.param_dict = dict()
        self.new_lines = False
        self.read_parameter_spec_file()

    @staticmethod
    def parse_line(line):
        components = line.split(' ')
        if len(components) >= 3 and components[0][0] != '/':
            return ' '.join(components[:2]), components[2], ' '.join(components[3:])
        else:
            return False

    def read_parameter_spec_file(self, verbose=False):
        with open(self.source_path, 'r') as fin:
            lines = fin.readlines()
            for i, line in enumerate(lines):
                line = line.strip('\n')
                self.raw[i] = line
                if verbose: print i, '>', line, '<'
                parsed_line = PSpec.parse_line(line)
                if parsed_line:
                    self.param_dict[parsed_line[0]] = (i, parsed_line[1:])
                if verbose: print i, parsed_line
            self.num_lines = len(lines)

    def change_value(self, module_var, value, comment=None):
        if module_var in self.param_dict:
            line_num, (val, com) = self.param_dict[module_var]
            if comment:
                com = comment
            self.raw[line_num] = ' '.join([module_var, value, com])
            print '[NOTE] PSPec.change_value(): Changing value {0}'.format(self.raw[line_num])
            return self.raw[line_num]
        else:
            print '[WARNING] PSpec.change_value(): No module_var in self.param_dict: {0}'.format(module_var)
            if self.new_lines is False:
                self.raw[self.num_lines] = ''
                self.num_lines += 1
                self.raw[self.num_lines] = '//// New variables added'
                self.num_lines += 1
            new_line = '{0} {1}'.format(module_var, value)
            if comment:
                new_line += ' {0}'.format(comment)
            print '          Creating: {0}'.format(new_line)
            self.raw[self.num_lines] = new_line
            self.num_lines += 1

    def save(self, filename, dest_dir=None):

        if dest_dir:
            if not os.path.exists(dest_dir):
                print '[NOTE] PSpec.save(): dir does not exit, creating:'
                print '       \'{0}\''.format(dest_dir)
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, filename)
        else:
            dest_path = filename

        with open(dest_path, 'w') as fout:
            for i in range(self.num_lines):
                fout.write(self.raw[i])
                fout.write('\n')

    def show(self):
        for i in range(self.num_lines):
            print i, '>', self.raw[i], '<'


# ----------------------------------------------------------------------

def generate_parameter_spec_file(source_pspec_path,
                                 destination_dir,
                                 dest_pspec_name,
                                 parameter_changes,
                                 verbose=False):
    cwd = os.getcwd()
    os.chdir(HAMLET_ROOT)

    pspec = PSpec(source_pspec_path)
    for module_var, value, comment in parameter_changes:
        pspec.change_value(module_var, value, comment)
    if verbose: pspec.show()
    pspec.save(dest_dir=destination_dir, filename=dest_pspec_name)

    os.chdir(cwd)


def test_generate_parameter_spec_file():
    dest_path = os.path.join(PARAMETERS_ROOT, 'cocktail16_hyper_alpha')
    source_path = os.path.join(PARAMETERS_ROOT, 'cocktail16_inference_BFact_HMM_W0.config')
    parameter_changes = (('HDP_hyperprior a_alpha', 0.01, '// shape parameter for alpha_ prior varying'),
                         ('HDP_hyperprior b_alpha', 5, None))

    generate_parameter_spec_file(source_path,
                                 dest_path,
                                 'cocktail16_inference_BFact_HMM_W0_alpha.config',
                                 parameter_changes,
                                 verbose=True)

# test_generate_parameter_spec_file()


# ----------------------------------------------------------------------

def generate_parameter_spec_ab_product(source_param_dir,
                                       source_param_files,
                                       dest_param_dir,
                                       module,
                                       param_var,
                                       avals,
                                       bvals,
                                       gen_param_files_p=False,
                                       verbose=False):
    change_set = [((('{0} a_{1}'.format(module, param_var), '{0}'.format(aval), None),
                    ('{0} b_{1}'.format(module, param_var), '{0}'.format(bval), None)),
                   (aval, bval))
                  for aval in avals
                  for bval in bvals]

    parameter_spec_parameters = list()

    i = 0
    for source_param_file in source_param_files:
        source_param_file_basename = source_param_file.split('.')[0]
        for parameter_changes, (aval, bval) in change_set:
            # aval = parameter_changes[0][1]
            # bval = parameter_changes[1][1]
            a_param_name = 'a{0}{1}'.format(param_var, float2string(aval))
            b_param_name = 'b{0}{1}'.format(param_var, float2string(bval))
            model_filename_postfix = '{0}_{1}'.format(a_param_name, b_param_name)
            new_param_file_basename = source_param_file_basename + '_{0}'.format(model_filename_postfix)
            new_param_filename = new_param_file_basename + '.config'

            param_spec = experiment_tools.ParameterSpec\
                (new_param_filename, dest_param_dir, model_filename_postfix)

            if verbose:
                print '\n({0}) -------------------------------'.format(i)
                print 'source_param_dir:', source_param_dir
                print 'source_param_file:', source_param_file
                print 'source_param_file_basename:', source_param_file_basename
                print 'dest_param_dir:', dest_param_dir
                print 'a_param_name:', a_param_name
                print 'b_param_name:', b_param_name
                print 'model_filename_postfix:', model_filename_postfix
                print 'new_param_file_basename:', new_param_file_basename
                print 'new_param_filename:', new_param_filename
                print 'parameter_changes:', parameter_changes
                print 'param_spec:', param_spec

            '''
            # Example params for generate_parameter_spec_file
            source_pspec_path = experiment/parameters/cocktail16_inference_BFact_HMM_W0.config
            destination_dir = experiment/parameters/cocktail16_hyper_alpha
            dest_pspec_name
            '''

            if gen_param_files_p:
                generate_parameter_spec_file(source_pspec_path=os.path.join(source_param_dir, source_param_file),
                                             destination_dir=dest_param_dir,
                                             dest_pspec_name=new_param_filename,
                                             parameter_changes=parameter_changes,
                                             verbose=verbose)

            parameter_spec_parameters.append((new_param_filename, dest_param_dir, model_filename_postfix))

            i += 1

    return parameter_spec_parameters


def test_generate_parameter_spec_ab_product(gen_param_files_p=False):
    source_param_dir = PARAMETERS_ROOT
    source_param_files = ('cocktail16_inference_BFact_HMM_W0.config',
                          'cocktail16_inference_LT_HMM_W0-J600.config',
                          'cocktail16_inference_noLT_HMM_W0-J600.config')
    dest_param_dir = os.path.join(PARAMETERS_ROOT, 'cocktail16_hyper_alpha')
    parameter_spec_parameters\
        = generate_parameter_spec_ab_product(source_param_dir=source_param_dir,
                                             source_param_files=source_param_files,
                                             dest_param_dir=dest_param_dir,
                                             module='HDP_hyperprior', param_var='alpha',
                                             avals=(0.01, 0.1, 5),
                                             bvals=(0.01, 0.1, 5),
                                             gen_param_files_p=gen_param_files_p,
                                             verbose=True)

    print '\n-------------------------------\nparameter_spec_parameters:'
    for i, parameter_spec_parameter in enumerate(parameter_spec_parameters):
        print i, parameter_spec_parameter

# test_generate_parameter_spec_ab_product(gen_param_files_p=True)


# ----------------------------------------------------------------------

def generate_parameter_spec_ab_product_outer(module, param_var, avals, bvals,
                                             source_param_files=('cocktail16_inference_BFact_HMM_W0.config',
                                                                 'cocktail16_inference_sticky_HMM_W0-J600.config',
                                                                 'cocktail16_inference_LT_HMM_W0-J600.config',
                                                                 'cocktail16_inference_noLT_HMM_W0-J600.config',
                                                                 'cocktail16_inference_stickyLT_HMM_W0-J600.config'),
                                             dest_param_dir=None,
                                             gen_param_files_p=False,
                                             verbose=False):
    """
    Generalizes test_ version as outer-wrapper for hyperparameter experiments based on cocktail16 configs
    :param module:
    :param param_var:
    :param avals:
    :param bvals:
    :param source_param_files:
    :param dest_param_dir: Optionally specify the destination parameter directory; default to None
        If None, then will be 'cocktail16_hyper_{0}'.format(param_var)
    :param gen_param_files_p:
    :param verbose:
    :return:
    """
    source_param_dir = PARAMETERS_ROOT

    if dest_param_dir is None:
        dest_param_dir = os.path.join(PARAMETERS_ROOT, 'cocktail16_hyper_{0}'.format(param_var))

    parameter_spec_parameters \
        = generate_parameter_spec_ab_product(source_param_dir=source_param_dir,
                                             source_param_files=source_param_files,
                                             dest_param_dir=dest_param_dir,
                                             module=module, param_var=param_var,
                                             avals=avals,
                                             bvals=bvals,
                                             gen_param_files_p=gen_param_files_p,
                                             verbose=verbose)
    return parameter_spec_parameters


def generate_parameter_spec_ab_product_hyper_regression_test(param_var='alpha',
                                                             gen_param_files_p=False):
    """
    Version of parameter spec for regression testing
    a,b values only = 1.0
    Runs through models: BFact, LT, noLT, Sticky, StickyLT
    - intended for debugging
    :param param_var:
    :param gen_param_files_p:
    :return:
    """
    return generate_parameter_spec_ab_product_outer\
        (module='HDP_hyperprior', param_var=param_var, avals=(1,), bvals=(1,),
         # source_param_files=('cocktail16_inference_LT_HMM_W0-J600.config',),
         gen_param_files_p=gen_param_files_p)


def generate_parameter_spec_ab_product_hyper_alpha(gen_param_files_p=False):
    return generate_parameter_spec_ab_product_outer\
        (module='HDP_hyperprior', param_var='alpha',
         avals=(0.01, 5), bvals=(0.01, 5),
         # avals=(0.01, 0.1, 1, 5), bvals=(0.01, 0.1, 1, 5),
         gen_param_files_p=gen_param_files_p)

# generate_parameter_spec_ab_product_hyper_alpha(gen_param_files_p=True)


def generate_parameter_spec_ab_product_hyper_alpha_debug(gen_param_files_p=False):
    """
    Smaller variant that only uses LT model and smaller number of a,b values
    - used for debugging
    :param gen_param_files_p:
    :return:
    """
    return generate_parameter_spec_ab_product_outer\
        (module='HDP_hyperprior', param_var='alpha', avals=(1, 0.1), bvals=(1, 0.1),
         source_param_files=('cocktail16_inference_LT_HMM_W0-J600.config',),
         dest_param_dir=os.path.join(PARAMETERS_ROOT, 'cocktail16_hyper_alpha_debug'),
         gen_param_files_p=gen_param_files_p)


def generate_parameter_spec_ab_product_hyper_gamma(gen_param_files_p=False):
    return generate_parameter_spec_ab_product_outer\
        (module='HDP_hyperprior', param_var='gamma',
         avals=(0.01, 5), bvals=(0.01, 5),
         # avals=(0.01, 0.1, 1, 5), bvals=(0.01, 0.1, 1, 5),
         gen_param_files_p=gen_param_files_p)

# generate_parameter_spec_ab_product_hyper_gamma(gen_param_files_p=True)


def generate_parameter_spec_ab_product_hyper_h(gen_param_files_p=False):
    return generate_parameter_spec_ab_product_outer\
        (module='Normal_noise_model', param_var='h',
         avals=(0.01, 5), bvals=(0.01, 5),
         # avals=(0.01, 0.1, 1, 5), bvals=(0.01, 0.1, 1, 5),
         gen_param_files_p=gen_param_files_p)

# generate_parameter_spec_ab_product_hyper_h(gen_param_files_p=True)


# ----------------------------------------------------------------------
# Parameter spec list
# ----------------------------------------------------------------------

# CTM 20170110: The following is deprecated -- can eventually remove.
def collect_parameter_spec_list_cocktail16_w0_hyperab(parameters_path, hyperparam):
    """
    cp **NO** weight learning (w0), 1500 iterations, D=16, and J=600 for hmm
    works with: cocktail_s16_m12
    :return:
    """
    bfact = [experiment_tools.ParameterSpec
             ('cocktail16_inference_BFact_HMM_W0_a{0}{1}_b{0}{2}.config' \
              .format(hyperparam, float2string(aalpha), float2string(balpha)),
              parameters_path,
              'a{0}{1}_b{0}{2}' \
              .format(hyperparam, float2string(aalpha), float2string(balpha)))
             for aalpha in (0.01, 0.1, 5)
             for balpha in (0.01, 0.1, 5)]

    lt = [experiment_tools.ParameterSpec
          ('cocktail16_inference_LT_HMM_W0-J600_a{0}{1}_b{0}{2}.config' \
           .format(hyperparam, float2string(aalpha), float2string(balpha)),
           parameters_path,
           'a{0}{1}_b{0}{2}' \
           .format(hyperparam, float2string(aalpha), float2string(balpha)))
          for aalpha in (0.01, 0.1, 5)
          for balpha in (0.01, 0.1, 5)]

    nolt = [experiment_tools.ParameterSpec
            ('cocktail16_inference_noLT_HMM_W0-J600_a{0}{1}_b{0}{2}.config' \
             .format(hyperparam, float2string(aalpha), float2string(balpha)),
             parameters_path,
             'a{0}{1}_b{0}{2}' \
             .format(hyperparam, float2string(aalpha), float2string(balpha)))
            for aalpha in (0.01, 0.1, 5)
            for balpha in (0.01, 0.1, 5)]

    return bfact + lt + nolt


def test_collect_parameter_spec_list_cocktail16_w0_hyper_alpha():
    pspecs = collect_parameter_spec_list_cocktail16_w0_hyperab(PARAMETERS_ROOT + '/cocktail16_hyper_alpha', 'alpha')
    for pspec in pspecs:
        print pspec

# test_collect_parameter_spec_list_cocktail16_w0_hyper_alpha()


# ----------------------------------------------------------------------
# The following work: will just collect the ParameterSpecs


def collect_parameter_spec_list_cocktail16_w0_hyper_regression(param_var='alpha'):
    """
    REGRESSION TEST Version of get parameter_spec_list for hyper
    sets a/b_<hyper_param_var>
    :param param_var:
    :return:
    """
    spec_list = generate_parameter_spec_ab_product_hyper_regression_test(param_var=param_var)
    return spec_list


# ----------------------------------------------------------------------


def collect_parameter_spec_list_cocktail16_w0_hyper_alpha():
    spec_list = generate_parameter_spec_ab_product_hyper_alpha(gen_param_files_p=False)
    return [experiment_tools.ParameterSpec(parameters_file, parameters_dir, model_filename_postfix)
            for parameters_file, parameters_dir, model_filename_postfix in spec_list]


def collect_parameter_spec_list_cocktail16_w0_hyper_alpha_lambda1p6():
    """
    Add the lambda1p6 (i.e., lambda fixed to value 1.6)
    :return:
    """
    spec_list = generate_parameter_spec_ab_product_hyper_alpha_debug(gen_param_files_p=False)
    pspec_list = [experiment_tools.ParameterSpec(parameters_file, parameters_dir, model_filename_postfix)
                  for parameters_file, parameters_dir, model_filename_postfix in spec_list]
    pspec_list += \
        [experiment_tools.ParameterSpec(
            parameters_file='cocktail16_inference_LT_HMM_W0-J600_aalpha1_balpha1_lambda1p6.config',
            parameters_dir='experiment/parameters/cocktail16_hyper_alpha_debug',
            model_filename_postfix='aalpha1_balpha1_lambda1p6'),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_LT_HMM_W0-J600_aalpha1_balpha0p1_lambda1p6.config',
             parameters_dir='experiment/parameters/cocktail16_hyper_alpha_debug',
             model_filename_postfix='aalpha1_balpha0p1_lambda1p6'),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_LT_HMM_W0-J600_aalpha0p1_balpha1_lambda1p6.config',
             parameters_dir='experiment/parameters/cocktail16_hyper_alpha_debug',
             model_filename_postfix='aalpha0p1_balpha1_lambda1p6'),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_LT_HMM_W0-J600_aalpha0p1_balpha0p1_lambda1p6.config',
             parameters_dir='experiment/parameters/cocktail16_hyper_alpha_debug',
             model_filename_postfix='aalpha0p1_balpha0p1_lambda1p6')
         ]
    return pspec_list


def collect_parameter_spec_list_cocktail16_w0_hyper_gamma():
    spec_list = generate_parameter_spec_ab_product_hyper_gamma(gen_param_files_p=False)
    return [experiment_tools.ParameterSpec(parameters_file, parameters_dir, model_filename_postfix)
            for parameters_file, parameters_dir, model_filename_postfix in spec_list]


def collect_parameter_spec_list_cocktail16_w0_hyper_h():
    spec_list = generate_parameter_spec_ab_product_hyper_h(gen_param_files_p=False)
    return [experiment_tools.ParameterSpec(parameters_file, parameters_dir, model_filename_postfix)
            for parameters_file, parameters_dir, model_filename_postfix in spec_list]


# ----------------------------------------------------------------------
# Scripts
# ----------------------------------------------------------------------


match_select_cp16 = {0: ['h{0}_nocs'.format(h) for h in [10.0]],
                     1: ['cp{0}'.format(i) for i in range(1)]}

# ----------------------------------------------------------------------

"""
hyper_alpha: [0.01, 0.1, 1.0, 5]
HDP_hyperprior a_alpha
HDP_hyperprior b_alpha

hyper_beta: [0.01, 0.1, 1.0, 5]
HDP_hyperprior a_gamma
HDP_hyperprior b_gamma

hyper_blambda: [0.01, 0.1, 5]
Isotropic_exponential_similarity blambda

hyper_h: [0.01, 0.1, 5]
Normal_noise_model a_h
Normal_noise_model b_h
"""


# ----------------------------------------------------------------------

# TODO: generate parameter files that run only 5 iterations

def exp_hyper_regression(test=True, param_var='alpha'):
    """
    REGRESSION TEST
    Using hyperparameter experiment as base, but a/b=1.0
    Using config cocktail16_inference_{BFact,LT,no_LT}
    2000 iterations, J=600,
    {a,b}_h=0.1 (prior over precision of noise)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'cocktail_s16_m12/'),
         results_dir=os.path.join(RESULTS_ROOT, 'cocktail_s16_m12/hyper_{0}_REGRESSION'.format(param_var)),
         replications=1,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_cocktail16_w0_hyper_regression(param_var=param_var),
         match_dict={0: ['h{0}_nocs'.format(h) for h in [10.0]],
                     1: ['cp{0}'.format(i) for i in range(1)]},
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=test,
         select_subdirs_verbose=False)

# exp_hyper_regression(test=True)


# ----------------------------------------------------------------------

def exp_hyper_alpha(test=True):
    """
    Experiment varying hyperparameters: a_alpha, b_alpha
    Using config cocktail16_inference_{BFact,LT,no_LT}
    2000 iterations, J=600,
    {a,b}_h=0.1 (prior over precision of noise)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'cocktail_s16_m12/'),
         results_dir=os.path.join(RESULTS_ROOT, 'cocktail_s16_m12/hyper_alpha'),
         replications=5,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_cocktail16_w0_hyper_alpha(),
         match_dict=match_select_cp16,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=test,
         select_subdirs_verbose=False)

# GENERATE parameter spec files
# generate_parameter_spec_ab_product_hyper_alpha(gen_param_files_p=True)

# RUN EXPRIMENT
# exp_hyper_alpha(test=True)


# ----------------------------------------------------------------------

def exp_hyper_alpha_plus_lambda1p6(test=True):
    """
    Experiment varying hyperparameters: a_alpha, b_alpha
    {a,b}_alpha \in (1.0, 0.1)
    Using config cocktail16_inference_LT only
    2000 iterations, J=600,
    {a,b}_h=0.1 (prior over precision of noise)
    :return:
    """
    spec_list = experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'cocktail_s16_m12/'),
         results_dir=os.path.join(RESULTS_ROOT, 'cocktail_s16_m12/hyper_alpha'),
         replications=5,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_cocktail16_w0_hyper_alpha_lambda1p6(),
         match_dict=match_select_cp16,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=test,
         select_subdirs_verbose=False)

    print spec_list

# NOTE: This only generates the four {a,b}_alpha variations;
# the lambda1p6 need to be generated manually...
# collect_parameter_spec_list_cocktail16_w0_hyper_alpha_lambda1p6(gen_param_files_p=True)

# Run hyper_alpha debug experiment
# exp_hyper_alpha_plus_lambda1p6(test=True)


# ----------------------------------------------------------------------

def exp_hyper_gamma(test=True):
    """
    Experiment varying hyperparameters: a_gamma, b_gamma
    Using config cocktail16_inference_{BFact,LT,no_LT}
    2000 iterations, J=600,
    {a,b}_h=0.1 (prior over precision of noise)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'cocktail_s16_m12/'),
         results_dir=os.path.join(RESULTS_ROOT, 'cocktail_s16_m12/hyper_gamma'),
         replications=5,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_cocktail16_w0_hyper_gamma(),
         match_dict=match_select_cp16,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=test,
         select_subdirs_verbose=False)

# GENERATE parameter spec files
# generate_parameter_spec_ab_product_hyper_gamma(gen_param_files_p=True)

# RUN EXPRIMENT
# exp_hyper_gamma(test=False)
