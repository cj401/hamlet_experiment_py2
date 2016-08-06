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
    (current=('scripts', 'python', 'experiments', 'REDD', 'experiment'),
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
# Parameter spec list
# ----------------------------------------------------------------------

def collect_parameter_spec_list_redd(parameters_path):
    """
    REDD data parameter spec list
    :return:
    """
    return [ experiment_tools.ParameterSpec('cocktail16_inference_BFact_HMM_W0.config', parameters_path),
             experiment_tools.ParameterSpec('cocktail16_inference_LT_HMM_W0-J600.config', parameters_path),
             experiment_tools.ParameterSpec('cocktail16_inference_noLT_HMM_W0-J600.config', parameters_path)]


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------

'''
print os.path.join(HAMLET_ROOT, DATA_ROOT), 'REDD/jw2013_downsampled_intervals/'
print os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT),
                   'REDD/jw2013_downsampled_intervals/')
print os.listdir(os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT),
                              'REDD/jw2013_downsampled_intervals/'))
'''

# TODO: create parameter spec files for REDD experiment
# TODO: update collect_parameter_spec_list_redd for the spec files


match_select_cp16 = {0: ['house_1_1200_6800',
                         'house_1_22600_27800',
                         'house_1_27300_32300',
                         'house_1_55300_59100',
                         'house_1_80000_83900',
                         'house_1_112600_116800',
                         'house_2_700_6700',
                         'house_2_15000_23200',
                         'house_2_36400_41800',
                         'house_3_3000_8000',
                         'house_3_9000_14000',
                         'house_3_19000_24000',
                         'house_3_37000_42000',
                         'house_3_42000_47000',
                         'house_6_15000_20000',
                         'house_6_29000_36000',
                         'house_6_36000_41000',
                         'house_6_46000_51000',
                         'house_6_51000_56000'], }

experiment_tools.run_experiment_script \
    (main_path=HAMLET_ROOT,
     data_dir=os.path.join(DATA_ROOT, 'REDD/jw2013_downsampled_intervals/'),
     results_dir=os.path.join(RESULTS_ROOT, 'redd'),
     replications=1,
     offset=0,
     parameter_spec_list=collect_parameter_spec_list_redd(PARAMETERS_ROOT),
     match_dict=match_select_cp16,
     multiproc=True,
     processor_pool_size=multiprocessing.cpu_count(),
     rerun=False,
     test=True,
     select_subdirs_verbose=False)
