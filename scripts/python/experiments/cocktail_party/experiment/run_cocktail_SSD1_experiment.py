import sys
import os
import multiprocessing


# ----------------------------------------------------------------------
# Ensure <hamlet>/experiment/scripts/python/ in sys.path  (if possible)
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

def collect_parameter_spec_list_cocktail16_w0_all_models():
    return \
        (experiment_tools.ParameterSpec(
            parameters_file='cocktail16_inference_BFact_HMM_W0.config',
            parameters_dir='experiment/parameters',
            model_filename_postfix=''),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_LT_HMM_W0-J200.config',
             parameters_dir='experiment/parameters',
             model_filename_postfix=''),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_noLT_HMM_W0-J200.config',
             parameters_dir='experiment/parameters',
             model_filename_postfix=''),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_sticky_HMM_W0-J200.config',
             parameters_dir='experiment/parameters',
             model_filename_postfix=''),
         experiment_tools.ParameterSpec(
             parameters_file='cocktail16_inference_stickyLT_HMM_W0-J200.config',
             parameters_dir='experiment/parameters',
             model_filename_postfix=''))


# ----------------------------------------------------------------------
# Scripts
# ----------------------------------------------------------------------

match_dict = {0: ['n0.3_500'],
              1: ['cp{0}'.format(i) for i in range(3)]}


def exp_hyper_regression(test=True):
    """
    First cocktail16_SSD1 'RW' experiment
    Using config cocktail16_inference_{BFact,LT,no_LT,sticky,stickyLT}
    2000 iterations
    J=200
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'cocktail_SSC1_s16_m12/'),
         results_dir=os.path.join(RESULTS_ROOT, 'cocktail_SSC1_s16_m12'),
         replications=5,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_cocktail16_w0_all_models(),
         match_dict=match_dict,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=test,
         select_subdirs_verbose=False)

exp_hyper_regression(test=True)
