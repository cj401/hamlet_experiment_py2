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
    (current=('scripts', 'python', 'experiments', 'latent_continue_syn', 'experiment'),
     parent=('scripts', 'python'),
     relative_path='../../../',
     verbose=True)


from run import experiment_tools


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'

print os.listdir(HAMLET_ROOT)

DATA_ROOT = experiment_tools.DATA_ROOT
PARAMETERS_ROOT = experiment_tools.PARAMETERS_ROOT
RESULTS_ROOT = experiment_tools.RESULTS_ROOT


# ----------------------------------------------------------------------
# Parameter spec list
# ----------------------------------------------------------------------

def collect_parameter_spec_list_latent_continue_syn_block_diag12(parameters_path):
    """
    Latent continuous state synthetic experiment parameters
    :return:
    """
    return [ experiment_tools.ParameterSpec('block_diag12_LT.config', parameters_path),
             experiment_tools.ParameterSpec('block_diag12_noLT.config', parameters_path) ]


def collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr(parameters_path):
    """
    Latent continuous state synthetic experiment parameters
    :return:
    """
    return [ experiment_tools.ParameterSpec('block_diag12_LT_10000itr.config', parameters_path),
             experiment_tools.ParameterSpec('block_diag12_noLT_10000itr.config', parameters_path) ]


def collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc(parameters_path):
    """
    Latent continuous state synthetic experiment parameters
    :return:
    """
    return [ experiment_tools.ParameterSpec('block_diag12_LT_10000itr_hmc.config', parameters_path),
             experiment_tools.ParameterSpec('block_diag12_noLT_10000itr_hmc.config', parameters_path) ]


def collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc_h01(parameters_path):
    """
    Latent continuous state synthetic experiment parameters
    :return:
    """
    return [ experiment_tools.ParameterSpec('block_diag12_LT_10000itr_hmc_h0.1.config', parameters_path),
             experiment_tools.ParameterSpec('block_diag12_noLT_10000itr_hmc_h0.1.config', parameters_path) ]


def collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc_h20(parameters_path):
    """
    Latent continuous state synthetic experiment parameters
    :return:
    """
    return [ experiment_tools.ParameterSpec('block_diag12_LT_10000itr_hmc_h2.0.config', parameters_path),
             experiment_tools.ParameterSpec('block_diag12_noLT_10000itr_hmc_h2.0.config', parameters_path) ]


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------

# DONE: define parameter specs (in <hamlet_root>/experiment/parameters/), for LT, noLT and BFact models (others?)
# DONE: fill in parameter spec file names in collect_parameter_spec_list_latent_continue_syn()
# DONE: generate synthetic data and place in <hamlet_root>/data/data/latent_continue_syn/
# DONE: update match_select_latent_continue_syn to match directory structure of latent_continue_syn data
# TODO: test experiment generation; ensure all paths correct

# DONE: The following needs to be modified to match the directory structure of the latent_continue_syn data
match_select_latent_continue_syn_block_diag12 \
    = { 0: ['block_diag12_s{0}'.format(h) for h in [2.5, 4, 10]] }


def exp1():
    """
    Using block_diag12_s{2.5, 4, 10} dataset
    Using syn_block_diag12, 2000 iterations, J=24
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag12,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)


# ----------------------------------------------------------------------

match_select_latent_continue_syn_block_diag12_10000itr \
    = { 0: ['block_diag12_s{0}'.format(h) for h in [2.5, 4, 10]] }


def exp2():
    """
    Using block_diag12_s{2.5, 4, 10} dataset
    Using syn_block_diag12_10000itr, 10,000 iterations, J=24
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag12,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# exp2()


# ----------------------------------------------------------------------

match_select_latent_continue_syn_block_diag12_10000itr_hmc \
    = { 0: ['block_diag12_s{0}'.format(h) for h in [2.5, 4, 10]] }


def exp3():
    """
    Using block_diag12_s{2.5, 4, 10} dataset
    Using syn_block_diag12_10000itr_hmc, 10,000 iterations, J=100
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr_hmc'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag12_10000itr_hmc,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# exp3()


# ----------------------------------------------------------------------

match_select_latent_continue_syn_block_diag40_10000itr_hmc \
    = { 0: ['block_diag40_s{0}'.format(h) for h in [2]] }


def exp4():
    """
    Using block_diag40_s2 dataset (10 blocks of 4 states, for a total of 40 states)
    Using syn_block_diag12_10000itr_hmc, 10,000 iterations, J=100
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr_hmc_diag40'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag40_10000itr_hmc,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# exp4()


# ----------------------------------------------------------------------

# match_select_latent_continue_syn_block_diag40_10000itr_hmc \
#     = { 0: ['block_diag40_s{0}'.format(h) for h in [2]] }


def exp5():
    """
    Using block_diag40_s2 dataset (10 blocks of 4 states, for a total of 40 states)
    Using syn_block_diag12_10000itr_hmc, 10,000 iterations, J=100,
    h=0.1 (precision of prior on latent locations; weaker prior)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr_hmc_diag40_h0.1'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc_h01(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag40_10000itr_hmc,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# exp5()


# ----------------------------------------------------------------------

# match_select_latent_continue_syn_block_diag40_10000itr_hmc \
#     = { 0: ['block_diag40_s{0}'.format(h) for h in [2]] }


def exp6():
    """
    Using block_diag40_s2 dataset (10 blocks of 4 states, for a total of 40 states)
    Using syn_block_diag12_10000itr_hmc, 10,000 iterations, J=100,
    h=2.0 (precision of prior on latent locations; stronger prior)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr_hmc_diag40_h2.0'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc_h20(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag40_10000itr_hmc,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run me!
# exp6()


# ----------------------------------------------------------------------


match_select_latent_continue_syn_block_diag4x10_10000itr_hmc \
    = { 0: ['block_diag4x10_s{0}'.format(h) for h in [2]] }


def exp7():
    """
    Using block_diag4x10_s2 dataset (4 blocks of 10 states, for a total of 40 states)
    Using syn_block_diag12_10000itr_hmc, 10,000 iterations, J=100,
    h=1.0 (precision of prior on latent locations)
    :return:
    """
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'continuous_latent_syn/'),
         results_dir=os.path.join(RESULTS_ROOT, 'continuous_latent_syn_10000itr_hmc_diag4x10_h1.0'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_latent_continue_syn_block_diag12_10000itr_hmc(PARAMETERS_ROOT),
         match_dict=match_select_latent_continue_syn_block_diag4x10_10000itr_hmc,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run me!
exp7()


# ----------------------------------------------------------------------

# print os.listdir(os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT), 'latent_continue_syn'))
