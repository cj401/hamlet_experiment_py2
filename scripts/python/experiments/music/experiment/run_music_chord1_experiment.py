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
    (current=('scripts', 'python', 'experiments', 'music', 'experiment'),
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

def collect_parameter_spec_list_music_chord1(parameters_path):
    """
    Music chord 1 **NO** weight learning (w0), 5000 iterations
    works with:
    :return:
    """
    return [ experiment_tools.ParameterSpec('music_chord1_LT.config', parameters_path),
             experiment_tools.ParameterSpec('music_chord1_noLT.config', parameters_path)
    ]


def collect_parameter_spec_list_music_chord1_J2(parameters_path):
    """
    Music chord 1 **NO** weight learning (w0), 10000 iterations, J=2
    works with:
    :return:
    """
    return [ experiment_tools.ParameterSpec('music_chord1_LT_J2.config', parameters_path),
             experiment_tools.ParameterSpec('music_chord1_noLT_J2.config', parameters_path)
    ]


def collect_parameter_spec_list_music_chord1_J4(parameters_path):
    """
    Music chord 1 **NO** weight learning (w0), 10000 iterations, J=4
    works with:
    :return:
    """
    return [ experiment_tools.ParameterSpec('music_chord1_LT_J4.config', parameters_path),
             experiment_tools.ParameterSpec('music_chord1_noLT_J4.config', parameters_path)
    ]


def collect_parameter_spec_list_music_chord1_J10(parameters_path):
    """
    Music chord 1 **NO** weight learning (w0), 10000 iterations, J=10
    works with:
    :return:
    """
    return [ experiment_tools.ParameterSpec('music_chord1_LT_J10.config', parameters_path),
             experiment_tools.ParameterSpec('music_chord1_noLT_J10.config', parameters_path)
    ]


def collect_parameter_spec_list_music_simple_melody1(parameters_path):
    """
    Music simple melody 1 **NO** weight learning (w0), 100,000 iterations
    works with:
    :return:
    """
    return [ experiment_tools.ParameterSpec('music_simple_melody1_LT.config', parameters_path),
             experiment_tools.ParameterSpec('music_simple_melody1_chord1_noLT.config', parameters_path)
    ]


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------


match_select_music_chord1 = {0: ['music_chord1'.format(h) for h in [10.0]] }


def run_exp1():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_chord1/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1(PARAMETERS_ROOT),
         match_dict=match_select_music_chord1,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# print os.listdir(os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT), 'cocktail_s16_m12'))


# ----------------------------------------------------------------------

match_select_music_chord1_Js = {0: ['music_chord1'] }


def run_exp_J2():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_chord1/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1_J2(PARAMETERS_ROOT),
         match_dict=match_select_music_chord1_Js,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_J2()


def run_exp_J4():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_chord1/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1_J4(PARAMETERS_ROOT),
         match_dict=match_select_music_chord1_Js,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_J4()


def run_exp_J10():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_chord1/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=10,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1_J10(PARAMETERS_ROOT),
         match_dict=match_select_music_chord1_Js,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_J10()


# ----------------------------------------------------------------------

match_select_music_tsd_v1_CS = {0: ['music_tsd_v1_CS'] }


def run_exp_tsd_v1_CS():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_tsd/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=3,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1(PARAMETERS_ROOT),
         match_dict=match_select_music_tsd_v1_CS,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_tsd_v1_CS()

# ----------------------------------------------------------------------

match_select_music_tsd_v1_triads = {0: ['music_tsd_v1_triads'] }


def run_exp_tsd_v1_triads():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/kulitta_tsd/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=3,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_chord1(PARAMETERS_ROOT),
         match_dict=match_select_music_tsd_v1_triads,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_tsd_v1_triads()


# ----------------------------------------------------------------------

match_select_music_simple_melody = {0: ['music_simple_melody1'] }


def run_exp_simple_melody1():
    experiment_tools.run_experiment_script \
        (main_path=HAMLET_ROOT,
         data_dir=os.path.join(DATA_ROOT, 'music/simple_melodies/'),
         results_dir=os.path.join(RESULTS_ROOT, 'music'),
         replications=3,
         offset=0,
         parameter_spec_list=collect_parameter_spec_list_music_simple_melody1(PARAMETERS_ROOT),
         match_dict=match_select_music_simple_melody,
         multiproc=True,
         processor_pool_size=multiprocessing.cpu_count(),
         rerun=False,
         test=True,
         select_subdirs_verbose=False)

# run_exp_simple_melody1()
