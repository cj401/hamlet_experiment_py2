import datetime
import glob
import multiprocessing
import os
import sys
from subprocess import call


# ----------------------------------------------------------------------
# Attempt to ensure parent to <hamlet>/experiment/scripts/python/ is available

def hello():
    print 'Hello from run.experiment_tools.hello() !'


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
    (current=('scripts', 'python', 'run'),
     parent=('scripts', 'python'),
     relative_path='..',
     verbose=True)


from utilities import util


# ----------------------------------------------------------------------


__author__ = 'clayton'


# TODO: Still in mid refactor -- new directory structure breaks things


# ----------------------------------------------------------------------
# Hamlet root paths
# ----------------------------------------------------------------------

# NOTE: The following is valid from <hamlet>/experiment/scripts/python
# Each domain experiment specified within <hamlet>/experiment/scripts/python/experiments
# should have its own local HAMLET_ROOT definition
HAMLET_ROOT = '../../../'

# print os.listdir(HAMLET_ROOT)

# The following assume a standard HAMLET_ROOT configuration
# and can be used by any experiment definition that assumes
# the <hamlet> directory structure, assuming a locally
# specified HAMLET_ROOT
DATA_ROOT = 'data/data'
PARAMETERS_ROOT = 'experiment/parameters'
RESULTS_ROOT = 'experiment/results'


# ----------------------------------------------------------------------
# Global LOCK
# ----------------------------------------------------------------------

lock = None  # Global definition of lock


# ----------------------------------------------------------------------


class ExperimentSpec:
    def __init__(self,

                 # possible wholly-formed command, typically specified by process_failures
                 command=None,

                 # specified by process_failures when rerunning experiment
                 rerun_exp_num=None,

                 # specified by collect_experiment_specs
                 parameters_file=None,   # -p
                 parameters_dir=None,    # --parameters_dir

                 data_subdir=None,       # --data_subdir
                 data_dir=None,          # --data_dir     # figures root
                 data_base_name=None,    #

                 results_subdir=None,    # -r
                 results_postfix=None,   # --results_timestamp
                 results_dir=None,       # --results_dir  # results root

                 weights_file=None,      # --weights_file

                 # specified by collect_experiment_specs after above determined
                 exp_num=None,
                 total_exps=None,

                 # specified by run_experiment_wrapper
                 main_path=None,
                 log_file=None,
                 test=None
                 ):

        self.command = command
        self.rerun_exp_num = rerun_exp_num

        self.parameters_file = parameters_file
        self.parameters_dir = parameters_dir

        self.data_dir = data_dir
        self.data_subdir = data_subdir
        self.data_base_name = data_base_name

        self.results_subdir = results_subdir
        self.results_postfix = results_postfix
        self.results_dir = results_dir

        self.weights_file = weights_file

        self.exp_num = exp_num
        self.total_exps = total_exps

        self.main_path = main_path

        self.log_file = log_file
        self.test = test

    def __str__(self):
        s = ''
        if self.command:
            s += 'command={0}, '.format(self.command)
        if self.rerun_exp_num:
            s += 'rerun_exp_num={0}'.format(self.rerun_exp_num)
        if self.parameters_file:
            s += 'parameters_file={0}, '.format(self.parameters_file)
        if self.parameters_dir:
            s += 'parameters_dir={0}, '.format(self.parameters_dir)
        if self.data_dir:
            s += 'data_dir={0}, '.format(self.data_dir)
        if self.data_subdir:
            s += 'data_subdir={0}, '.format(self.data_subdir)
        if self.data_base_name:
            s += 'data_base_name={0}, '.format(self.data_base_name)
        if self.results_dir:
            s += 'results_dir={0}, '.format(self.results_dir)
        if self.results_subdir:
            s += 'results_subdir={0}, '.format(self.results_subdir)
        if self.results_postfix:
            s += 'results_postfix={0}, '.format(self.results_postfix)
        if self.weights_file:
            s += 'weights_file={0}, '.format(self.weights_file)
        if self.exp_num:
            s += 'exp_num={0}, '.format(self.exp_num)
        if self.total_exps:
            s += 'total_exps={0}, '.format(self.total_exps)
        if self.main_path:
            s += 'main_path={0}, '.format(self.main_path)
        if self.log_file:
            s += 'log_file={0}, '.format(self.log_file)
        if self.test:
            s += 'test={0}'.format(self.test)
        return s


# ----------------------------------------------------------------------


# -p parameters_filename
# --parameters_dir parameters_dir

# --weights_file weights_file

# --figures data_file_name
# --data_subdir data_subdir

# -r results_subdir_name
# --results_dir results_dir  # if different from default


def run_experiment(spec):

    global lock

    if spec.command:
        command = spec.command
    else:
        command = './main -p {0}'.format(spec.parameters_file)
        if spec.parameters_dir:
            command += ' --parameters_dir={0}'.format(spec.parameters_dir)
        if spec.data_subdir:
            command += ' --data_subdir={0}'.format(spec.data_subdir)
        if spec.data_dir:
            command += ' --data_dir={0}'.format(spec.data_dir)
        if spec.results_subdir:
            command += ' -r {0}'.format(spec.results_subdir)
        if spec.results_postfix:
            command += ' --results_timestamp={0}'.format(spec.results_postfix)
        if spec.results_dir:
            command += ' --results_dir={0}'.format(spec.results_dir)
        if spec.weights_file:
            command += ' --weights_file={0}'.format(spec.weights_file)

    log_message = "('command', {0}, {1}, '{2}'" \
        .format(spec.exp_num, spec.total_exps, command)

    if spec.rerun_exp_num:
        log_message += ', {0}'.format(spec.rerun_exp_num)

    if spec.test:

        log_message += ')'
        lock.acquire()
        print log_message
        with open(spec.log_file, 'a') as logf:
            logf.write(log_message + '\n')
        lock.release()

    owd = os.getcwd()
    os.chdir(spec.main_path)

    print '>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'spec.main_path', spec.main_path
    print 'os.getcwd()', os.getcwd()
    # sys.exit()

    ret = -1  # indicates test

    start = datetime.datetime.now()

    # check if target results subdir already exists
    # if so, bail...
    results_path = spec.results_subdir
    if spec.results_dir:
        results_path = spec.results_dir + results_path
    if os.path.isdir(results_path):
        log_message = "('ERROR', 'results_dir_exists', " + log_message + ')'
        if not spec.test:
            log_message += ' )'

        print log_message

        os.chdir(owd)

        lock.acquire()
        with open(spec.log_file, 'a') as logf:
            logf.write(log_message + '\n')
        lock.release()

        return spec.exp_num, -2

    elif not spec.test:

        ret = call(command, shell=True)

    end = datetime.datetime.now()

    os.chdir(owd)

    if not spec.test:
        lock.acquire()
        log_message += ", {0}, '{1}')\n".format(ret, end-start)
        with open(spec.log_file, 'a') as logf:
            logf.write(log_message)
        lock.release()

    return spec.exp_num, ret


# ----------------------------------------------------------------------


def run_experiment_batch(parameter_spec_list,
                         log_file,
                         multiproc=False,
                         processor_pool_size=8,
                         test=True):

    global lock

    results = ''
    start = datetime.datetime.now()

    log_message = "('total_experiments', {0})".format(len(parameter_spec_list))

    if multiproc:

        log_message += "\n('multiprocessing', {0})".format(processor_pool_size)
        print log_message
        lock.acquire()
        with open(log_file, 'a') as logf:
            if test: logf.write('RUNNING in TEST MODE\n')
            logf.write(log_message + '\n')
        lock.release()

        p = multiprocessing.Pool(processor_pool_size)
        results = p.map(run_experiment, parameter_spec_list)

    else:

        log_message += "\n('single_process')"
        print log_message
        lock.acquire()
        with open(log_file, 'a') as logf:
            if test: logf.write('RUNNING in TEST MODE')
            logf.write(log_message + '\n')
        lock.release()

        for spec in parameter_spec_list:
            run_experiment(spec)

    end = datetime.datetime.now()
    total_time = end - start

    lock.acquire()
    with open(log_file, 'a') as logf:
        if results:
            failures_list = get_failures(results)
            logf.write("('results', {0})\n".format(results))
            if failures_list > 0:
                logf.write("('failures', {0}, {1})\n".format(len(failures_list), failures_list))
        logf.write("('total_time', '{0}')\n".format(total_time))
    lock.release()

    if results: print results
    print '\nTotal time: {0}'.format(total_time)


# ----------------------------------------------------------------------


def read_log(filepath):
    command_dict = dict()
    results_list = list()
    with open(filepath, 'r') as fp:
        for line in fp.readlines():
            if line[2:9] == 'command':
                t = eval(line)
                command_dict[t[1]] = t[3]
            if line[2:9] == 'results':
                results_list = eval(line)[1]
    return command_dict, results_list


def test_read_log():
    command_dict, results_list = read_log('test_data/test.log')
    assert command_dict[7] == './main -p cocktail_inference_no_LT_HMM.config ' \
                              + '--data_subdir=cp_1/ -r cp_1/no_LT ' \
                              + '--results_timestamp=01'
    assert command_dict[15] == './main -p cocktail_inference_no_LT_HMM.config ' \
                               + '--data_subdir=cp_1/ -r cp_1/no_LT ' \
                               + '--results_timestamp=02'
    assert results_list == [(1, 0), (2, 0), (3, 0)]
    print 'PASS test_read_log()'

# TODO: test is broken b/c relying on paths that aren't valid if not executing in same dir
# test_read_log()


def collect_results_from_log(filepath):
    results = []
    with open(filepath, 'r') as fp:
        for line in fp.readlines():
            if line[2:9] == 'command':
                cline = eval(line)
                results.append((cline[1], cline[4]))
    failures = [ (exp_num, res) for exp_num, res in results if res == 1 ]
    print "('results', {0})".format(results)
    print "('failures, {0}, {1})".format(len(failures), failures)
    return results, failures

#collect_results('exp_run_20150609_223203867358.log')


def get_params_from_command_string(command):
    parameters_file = None
    parameters_dir = None
    results_subdir = None
    results_dir = None
    results_timestamp = None

    parameters_file_idx = command.find(' -p ')
    if parameters_file_idx > -1:
        parameters_file = command[parameters_file_idx:].split(' ')[2]

    parameters_dir_idx = command.find('--parameters_dir=')
    if parameters_dir_idx > -1:
        parameters_dir = command[parameters_dir_idx:].split(' ')[0][17:]

    results_subdir_idx = command.find(' -r ')
    if results_subdir_idx > -1:
        results_subdir = command[results_subdir_idx:].split(' ')[2]

    results_dir_idx = command.find('--results_dir=')
    if results_dir_idx > -1:
        results_dir = command[results_dir_idx:].split(' ')[0][14:]

    results_timestamp_idx = command.find('--results_timestamp=')
    if results_timestamp_idx > -1:
        results_timestamp = command[results_timestamp_idx:].split(' ')[0][20:]

    return parameters_file, parameters_dir, results_dir, results_subdir, results_timestamp


def test_get_params_from_command_string():
    t1 = './main -p cocktail_inference_LT_HMM.config --parameters_dir=parameters/ ' \
         + '--weights_file=figures/cocktail/noise_a2b3/cp_2/weights.txt ' \
         + '--data_subdir=cp_2 --data_dir=figures/cocktail/noise_a2b3/ -r LT_01 ' \
         + '--results_dir=results/cocktail/hmm/a2b3/cp2/'
    parameters_file, parameters_dir, results_dir, results_subdir, results_timestamp = \
        get_params_from_command_string(t1)
    # print parameters_file, parameters_dir, results_dir, results_subdir, results_timestamp
    assert parameters_file == 'cocktail_inference_LT_HMM.config'
    assert parameters_dir == 'parameters/'
    assert results_dir == 'results/cocktail/hmm/a2b3/cp2/'
    assert results_subdir == 'LT_01'
    assert results_timestamp is None

    t2 = './main -p cocktail_inference_noLT_HMM.config --data_subdir=cp0/ ' \
         + '-r cp0/noLT --results_timestamp=08'
    parameters_file, parameters_dir, results_dir, results_subdir, results_timestamp = \
        get_params_from_command_string(t2)
    # print parameters_file, parameters_dir, results_dir, results_subdir, results_timestamp
    assert parameters_file == 'cocktail_inference_noLT_HMM.config'
    assert parameters_dir is None
    assert results_dir is None
    assert results_subdir == 'cp0/noLT'
    assert results_timestamp == '08'

    print 'PASS test_get_params_from_command_string()'

test_get_params_from_command_string()


def get_failures(results_list):
    return [ (exp_num, res) for exp_num, res in results_list if res != 0 ]


def failures_p(results_list):
    if len(get_failures(results_list)) > 0:
        return True
    return False


def separate_last_dir_from_path(path):
    if path[-1] == '/':
        path = path[:-1]
    pieces = path.split('/')
    if len(pieces) == 1:
        return None, pieces[0]
    else:
        lead = pieces[:-1]
        lead = '/'.join(lead) + '/'
        end = pieces[-1]
        return lead, end


def move_results_subdir(spec, test=True):

    global lock

    ret = -2

    owd_outer = os.getcwd()
    os.chdir(spec.main_path)

    # find results path
    # results_path = <results_dir> + <results_subdir> + <results_dir>

    # (1) extract any params specified in command args
    # 'results_subdir',   # -r
    # 'results_postfix',  # --results_timestamp
    # 'results_dir',      # --results_dir  # results root

    parameters_file, parameters_dir, \
        results_dir, results_subdir, results_postfix = \
        get_params_from_command_string(spec.command)

    if not parameters_dir:
        parameters_dir = 'parameters/'  # default parameters dir

    # (2) using path to params file, extract parameters

    params_dict = util.read_parameter_file_as_dict\
        (parameters_dir, parameters_file)

    # (3) for each results directory param: if not already set,
    #     check if exists in param file, if so use that value
    #     else, specify default or None

    if not results_dir:
        if ':environment:results_dir' in params_dict:
            results_dir = params_dict[':environment:results_dir']
        else:
            results_dir = 'results/'  # default results dir

    if not results_subdir:
        if ':experiment:results_subdir' in params_dict:
            results_subdir = params_dict[':experiment:results_subdir']

    if not results_postfix:
        if ':experiment:results_timestamp' in params_dict:
            results_postfix = params_dict[':experiment:results_timestamp']

    # 'results_dir' is optional in either cli or params file, defaults as above
    # 'results_subdir' must either be in cli or params file
    # 'results_postfix' is optional in either cli or params file
    #    if not specified, then timestamp was created and must glob for it

    results_path = results_dir
    if results_subdir:
        results_path += results_subdir

    if results_postfix:
        results_path += results_postfix
    else:
        # timestamp was generated, so search for single timestamped directory
        globbed_results_path = results_path
        if globbed_results_path[-1] == '/':
            globbed_results_path = globbed_results_path[:-1]
        globbed_results_path += '*'
        candidate_results_paths = glob.glob(globbed_results_path)
        if len(candidate_results_paths) != 1:

            log_message = "('ERROR', 'move_results_subdir()', 'Expected 1 " \
                          + "directory for results_path', '{0}', "\
                              .format(globbed_results_path)\
                          + '{0}, '.format(candidate_results_paths)\
                          + "'Need to resolve by hand')"
            lock.acquire()
            print log_message
            lwd = os.getcwd()
            os.chdir(owd_outer)
            with open(spec.log_file, 'a') as logf:
                logf.write(log_message + '\n')
            os.chdir(lwd)
            lock.release()

            if not test:
                sys.exit(ret)
        else:
            results_path = candidate_results_paths[0]

    if not os.path.isdir(results_path):

        log_message = "('ERROR', 'move_results_subdir()', 'Cannot find results_path', " \
                      + "'{0}')".format(results_path)
        lock.acquire()
        print log_message
        lwd = os.getcwd()
        os.chdir(owd_outer)
        with open(spec.log_file, 'a') as logf:
            logf.write(log_message + '\n')
        os.chdir(lwd)
        lock.release()

        if not test:
            sys.exit(ret)

    else:

        path_lead, path_end = separate_last_dir_from_path(results_path)

        new_path_end = 'dead_{0}_{1}'.format(path_end, util.get_timestamp())
        cmd = 'mv {0} {1}'.format(path_end, new_path_end)

        test_note = ''
        if test: test_note = "'TEST', "
        log_message = "({0}'MOVE', '{1}', ('at', '{2}')".format(test_note, cmd, path_lead)

        if test: log_message += ')'

        if not test:

            owd = os.getcwd()
            os.chdir(path_lead)  # Assumes were at ./main

            ret = call(cmd, shell=True)

            os.chdir(owd)

            log_message += ', {0})'.format(ret)

        lock.acquire()
        print log_message
        lwd = os.getcwd()
        os.chdir(owd_outer)
        with open(spec.log_file, 'a') as logf:
            logf.write(log_message + '\n')
        os.chdir(lwd)
        lock.release()

    os.chdir(owd_outer)

    return spec.exp_num, ret


def process_failures(command_dict,
                     results_list,
                     log_file='exp_rerun',
                     main_path='../',
                     multiproc=False,
                     processor_pool_size=multiprocessing.cpu_count(),
                     test=True):
    """
    Processes previously failed experiment runs
    :param command_dict:
    :param results_list:
    :param log_file:
    :param main_path:
    :param multiproc:
    :param processor_pool_size:
    :param test:
    :return: log_file
    """

    global lock

    lock = multiprocessing.Lock()

    if test: print 'RUNNING in TEST MODE'

    log_file += '_' + util.get_timestamp() + '.log'

    if test: log_file = 'test_' + log_file

    rerun_exp_specs_list = [ ExperimentSpec(command=command_dict[key],
                                            rerun_exp_num=key,
                                            main_path=main_path,
                                            log_file=log_file,
                                            test=test)
                             for key, res in results_list if res !=0 ]

    total_exps = len(rerun_exp_specs_list)
    for exp_num, spec in zip(range(1, total_exps + 1), rerun_exp_specs_list):
        spec.exp_num = exp_num
        spec.total_exps = total_exps

    log_message = "('rerun_total_experiments', {0})".format(total_exps)

    lock.acquire()
    print log_message
    with open(log_file, 'a') as logf:
        logf.write(log_message + '\n')
    lock.release()

    # safely mv dead experiment directories
    move_cmd_results = []
    for spec in rerun_exp_specs_list:
        move_cmd_results.append(move_results_subdir(spec, test))

    log_message = "('move_cmd_results', {0})".format(move_cmd_results)

    lock.acquire()
    print log_message
    with open(log_file, 'a') as logf:
        logf.write(log_message + '\n')
    lock.release()

    move_failures = [ (exp_num, res)
                      for exp_num, res in move_cmd_results
                      if res != 0 ]
    if move_failures:

        log_message = "('ERROR', 'process_failures()', 'move_failures', {0})"\
            .format(move_failures)

        lock.acquire()
        print log_message
        with open(log_file, 'a') as logf:
            logf.write(log_message + '\n')
        lock.release()

        if not test:
            sys.exit(-1)

    if not test:

        # rerun the experiments!
        run_experiment_batch(rerun_exp_specs_list,
                             log_file,
                             multiproc=multiproc,
                             processor_pool_size=processor_pool_size,
                             test=test)

    return log_file


def rerun_experiment(target_log_file,
                     rerun_count_max=10,
                     main_path='../',
                     multiproc=False,
                     processor_pool_size=multiprocessing.cpu_count(),
                     test=True):

    command_dict, results_list = read_log(target_log_file)
    rerun_p = failures_p(results_list)
    rerun_count = 0
    while rerun_p and rerun_count < rerun_count_max:
        print '>>>>>>>>>>>>>>>>>>>>>> RERUN {0} <<<<<<<<<<<<<<<<<<<<<<<<<<'\
            .format(rerun_count)
        new_log_file = process_failures\
            (command_dict, results_list,
             main_path=main_path,
             multiproc=multiproc,
             processor_pool_size=processor_pool_size,
             test=test)
        command_dict, results_list = read_log(new_log_file)
        rerun_p = failures_p(results_list)
        rerun_count += 1
        if test: rerun_p = False


# ----------------------------------------------------------------------


def run_experiment_wrapper(experiment_spec_list,

                           multiproc=False,
                           processor_pool_size=8,

                           rerun_count_max=10,

                           main_path='../',

                           log_file='exp_run',
                           test=True,

                           rerun=True):

    global lock

    lock = multiprocessing.Lock()

    if test: print 'RUNNING in TEST MODE'

    log_file += '_' + util.get_timestamp() + '.log'

    print "log file name: '{0}'".format(log_file)

    for spec in experiment_spec_list:
        spec.main_path = main_path
        spec.log_file = log_file
        spec.test = test

    run_experiment_batch(experiment_spec_list,
                         log_file,
                         multiproc=multiproc,
                         processor_pool_size=processor_pool_size,
                         test=test)

    if rerun and not test:

        # Rerun failed experiments, looping up to rerun_count_max times

        command_dict, results_list = read_log(log_file)
        rerun_p = failures_p(results_list)
        rerun_count = 0
        while rerun_p and rerun_count < rerun_count_max:
            print '>>>>>>>>>>>>>>>>>>>>>> RERUN {0} <<<<<<<<<<<<<<<<<<<<<<<<<<'\
                .format(rerun_count)
            new_log_file = process_failures\
                (command_dict, results_list,
                 main_path=main_path,
                 multiproc=multiproc,
                 processor_pool_size=processor_pool_size,
                 test=test)
            command_dict, results_list = read_log(new_log_file)
            rerun_p = failures_p(results_list)
            rerun_count += 1

    print 'DONE.'


# ----------------------------------------------------------------------
# scripts
# ----------------------------------------------------------------------


# changes:
# (1) policy proposal for results structure:
#     <results_root>/<data_subdir>/<model_subdir>_<postfix>
#     underscores are significant, interpreted like directory path
# (2) run_experiment now checks if target path already exists, if it does, bails
# (3) rationalized log format, labeled tuple-based, easier to read automatically
# (4) rationalized experiment re-running so now integrated with normal run exp

# figures root: cocktail
# figures:        [ {noise_a1b1} x {cp_0, cp_1} ] x
# model:       [ {HMM, HSMM} x {LT, no_LT} ] x
# repetitions: 10

# <hamlet>/results/<results_root_name>/<data_subdir>/<model_subdir>/<postfix>

# <results_root_name> -- typically referring to the figures set
#   := cocktail | cocktail_narrow_4 | normal | probit ...
# <data_subdir>    := <cocktail>/noise_a1b1_cp_2/ | <normal>/HMM_Dir_s1.0_L0.0_01
# <model_subdir>   := hmm_hdp_LT_05 | hsmm_dir_noLT_03
# <postfix>        := num and/or timestamp

# examples:
# <hamlet>/results/cocktail/noise_a1b1_cp0/hmm_hdp_LT_04/
# <hamlet>/results/cocktail_narrow_4/noise_a1b1_cp0/hmm/hmm_hdp_noLT_05/
# <hamlet>/results/normal/HMM_Dir_s1.0_L0.0_01/hmm_hdp_LT_04/


# -------------------------------


ParameterSpec = util.namedtuple_with_defaults\
    ('ParameterSpec', ['parameters_file', 'parameters_dir'])


def collect_parameter_spec_list_cp_W0(parameters_path):
    """
    Parameters with Known weights
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HSMM_W0.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HSMM_W0.config', parameters_path) ]


def collect_parameter_spec_list_cp_BFact_only_W0(parameters_path):
    """
    Parameters with Known weights
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W0.config', parameters_path) ]


def collect_parameter_spec_list_cp_W1(parameters_path):
    """
    Parameters for weight learning
    :return:
    """
    return [ # ParameterSpec('cocktail_inference_BFact_HMM_W1.config'),
             ParameterSpec('cocktail_inference_LT_HMM_W1.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HSMM_W1.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HMM_W1.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HSMM_W1.config', parameters_path) ]


def collect_parameter_spec_list_cp_BFact_only_W1(parameters_path):
    """
    Parameters with Known weights
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W1.config', parameters_path) ]


def collect_parameter_spec_list_cp_W1_4000(parameters_path):
    """
    cp, weight learning (w1), 4000 iterations
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W1_4000.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HMM_W1_4000.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HSMM_W1_4000.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HMM_W1_4000.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HSMM_W1_4000.config', parameters_path) ]


def collect_parameter_spec_list_cp_W1_4000_D16(parameters_path):
    """
    cp weight learning (w1), 4000 iterations, D=16
    works with: cocktail_s16_m12
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W1_4000_D16.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HMM_W1_4000_D16.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HMM_W1_4000_D16.config', parameters_path)
             ]


def collect_parameter_spec_list_cp_W0_1500_D16(parameters_path):
    """
    cp **NO** weight learning (w0), 1500 iterations, D=16
    works with: cocktail_s16_m12
    :return:
    """
    return [ ParameterSpec('cocktail16_inference_BFact_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail16_inference_LT_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail16_inference_noLT_HMM_W0.config', parameters_path)]


def collect_parameter_spec_list_cp_W0_2000_D16_hmmJ600(parameters_path):
    """
    cp **NO** weight learning (w0), 1500 iterations, D=16, and J=600 for hmm
    works with: cocktail_s16_m12
    :return:
    """
    return [ ParameterSpec('cocktail16_inference_BFact_HMM_W0.config', parameters_path),
             ParameterSpec('cocktail16_inference_LT_HMM_W0-J600.config', parameters_path),
             ParameterSpec('cocktail16_inference_noLT_HMM_W0-J600.config', parameters_path)]


def collect_parameter_spec_list_cp_W1_4000_D14_J250(parameters_path):
    """
    cp weight learning (w1), 4000 iterations, D=14, J=250
    works with: cocktail_s14_m12/
    :return:
    """
    return [ ParameterSpec('cocktail_inference_BFact_HMM_W1_4000_D14_J250.config', parameters_path),
             ParameterSpec('cocktail_inference_LT_HMM_W1_4000_D14_J250.config', parameters_path),
             ParameterSpec('cocktail_inference_noLT_HMM_W1_4000_D14_J250.config', parameters_path)]


def collect_parameter_spec_list_synth(parameters_path):
    """
    Synthetic model figures experiment, no weight learning
    works with: synth/
    :return:
    """
    return [ ParameterSpec('BFact_inference.config', parameters_path),
             ParameterSpec('LT_inference.config', parameters_path),
             ParameterSpec('noLT_inference.config', parameters_path)]


def collect_parameter_spec_list_synth16(parameters_path):
    """
    Synthetic model figures experiment, 16 states, no weight learning
    works with: synth16/
    :return:
    """
    return [ ParameterSpec('BFact16_inference.config', parameters_path),
             ParameterSpec('LT16_inference.config', parameters_path),
             ParameterSpec('noLT16_inference.config', parameters_path)]


'''
def collect_parameter_spec_list_cp_W1_1500():
    """
    Parameters for weight learning
    :return:
    """
    return [ ParameterSpec('cocktail_inference_LT_HMM_W1_1500.config'),
             ParameterSpec('cocktail_inference_LT_HSMM_W1_1500.config'),
             ParameterSpec('cocktail_inference_noLT_HMM_W1_1500.config'),
             ParameterSpec('cocktail_inference_noLT_HSMM_W1_1500.config') ]

def collect_parameter_spec_list_cp_W1_1500_Nemh01():
    """
    Parameters for weight learning, but ensuring Normal_emission_model a_h=b_h=0.1
    :return:
    """
    return [ ParameterSpec('cocktail_inference_LT_HMM_W1_1500_Nemh0.1.config'),
             ParameterSpec('cocktail_inference_LT_HSMM_W1_1500_Nemh0.1.config'),
             ParameterSpec('cocktail_inference_noLT_HMM_W1_1500_Nemh0.1.config'),
             ParameterSpec('cocktail_inference_noLT_HSMM_W1_1500_Nemh0.1.config') ]


def factorial_parameter_spec_list_W0():
    return [ ParameterSpec('binary_factorial_fixed_mean.config') ]


def factorial_parameter_spec_list_W1():
    return [ ParameterSpec('binary_factorial_fixed_mean_W1_1500.config') ]


def HMM_LT_parameter_spec_list():
    return [ ParameterSpec('cocktail_inference_LT_HMM.config'), ]


def HMM_noLT_parameter_spec_list():
    return [ ParameterSpec('cocktail_inference_noLT_HMM.config'), ]
'''


# -------------------------------


DataSpec = util.namedtuple_with_defaults\
    ('DataSpec', ['data_dir', 'data_subdir', 'weights_file'])


'''
def select_subdirs(dir_branches, match_dict=None, verbose=False):
    """

    HACK!!: hard coding to ignore emissions/ subdir in figures dir.

    Select subset of dir_branches according to match_dict
    match_dict has this format:
        { index_num : [ <list of strings> ], }
    For each branch in dir_branches:
        If a match_dict is defined, iterate through each index_num
        only accept branch if each branch[index_num] matches a string in <list of strings>
    :param dir_branches:
    :param match_dict:
    :return:
    """

    if verbose:
        print 'select_subdirs(): match_dict={0}'.format(match_dict)

    # return dir_branches if no match_dict provided
    if not match_dict:
        return dir_branches

    branches = []
    for branch in dir_branches:

        if verbose:
            print '    checking branch: {0}'.format(branch)

        branch_pieces = branch
        if branch_pieces[-1] == '/':
            branch_pieces = branch_pieces[:-1]
        branch_pieces = branch_pieces.split('/')
        accept = True
        for match_idx in match_dict.keys():

            if verbose:
                print '        testing match_idx={0}'.format(match_idx)

            match_set = match_dict[match_idx]

            if verbose:
                print '            branch_pieces[match_idx]: {0}'.format(branch_pieces[match_idx])
                print '            match_set:                {0}'.format(match_set)
                print '            branch_pieces[match_idx] not in match_set : {0}'\
                    .format(branch_pieces[match_idx] not in match_set)

            if branch_pieces[match_idx] not in match_set:
                accept = False

            if verbose:
                print '            ACCEPT: {0}'.format(accept)

        if accept:
            # TODO CTM 20150702: at some point, make this more general

            if verbose:
                print '        ACCEPTING: {0} ; {1}'.format(branch, branch_pieces)

            branch_accept = branch

            # TODO: fix this SUPER HACK !!!!!
            if branch_pieces[-1] in ['emissions', 'A', 'pi', 'pi0', 'theta', 'thetastar', 'W'] \
                    + ['{0}'.format(i) for i in range(50)]:
                branch_accept = '/'.join(branch_pieces[:-1])  # + '/'

            if branch_accept not in branches:
                branches.append(branch_accept)
    return branches
'''


def select_subdirs(dir_branches, match_dict=None, verbose=False):
    """
    """

    if verbose:
        print 'select_subdirs(): match_dict={0}'.format(match_dict)

    # return dir_branches if no match_dict provided
    if not match_dict:
        return dir_branches

    def get_matched_branch(branch):
        branch_components = branch.strip('/').split('/')
        # matched_branch = list()
        end = 0
        for k in sorted(match_dict.keys()):
            # print k, branch, branch_components, k, match_dict[k], k > len(branch_components),\
            #     branch_components[k] not in match_dict[k]
            if k > len(branch_components):
                return None
            if branch_components[k] not in match_dict[k]:
                return None
            end = k
        return '/'.join(branch_components[0:end+1])

    branches = []
    for branch in dir_branches:
        matched_branch = get_matched_branch(branch)
        if matched_branch is not None and matched_branch not in branches:
            branches.append(matched_branch)

    return branches


def test_select_subdirs():

    dir_branches = ['a1b1/cp0/', 'a1b1/cp1/', 'a1b1/cp2/',
                    'a1b1/cp3/', 'a1b1/cp4/', 'a1b1/cp5/',
                    'a1b1/cp6/', 'a1b1/cp7/', 'a1b1/cp8/', 'a1b1/cp9/',
                    'a3b2/cp0/', 'a3b2/cp1/', 'a3b2/cp2/',
                    'a3b2/cp3/', 'a3b2/cp4/', 'a3b2/cp5/',
                    'a3b2/cp6/', 'a3b2/cp7/', 'a3b2/cp8/', 'a3b2/cp9/',
                    'a3b6/cp0/', 'a3b6/cp1/', 'a3b6/cp2/',
                    'a3b6/cp3/', 'a3b6/cp4/', 'a3b6/cp5/',
                    'a3b6/cp6/', 'a3b6/cp7/', 'a3b6/cp8/', 'a3b6/cp9/']

    match_dict1 = {
        1: [ 'cp{0}'.format(i) for i in range(3) ]
    }

    selected_dirs = select_subdirs(dir_branches, match_dict1)
    print selected_dirs
    assert ['a1b1/cp0/', 'a1b1/cp1/', 'a1b1/cp2/',
            'a3b2/cp0/', 'a3b2/cp1/', 'a3b2/cp2/',
            'a3b6/cp0/', 'a3b6/cp1/', 'a3b6/cp2/'] == selected_dirs

    match_dict2 = {
        0: [ 'a3b2' ],
        1: [ 'cp{0}'.format(i) for i in range(3) ]
    }

    selected_dirs = select_subdirs(dir_branches, match_dict2)
    assert ['a3b2/cp0/', 'a3b2/cp1/', 'a3b2/cp2/'] == selected_dirs

    print 'PASS test_select_subdirs()'

# TODO Reinstate
# test_select_subdirs()


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def get_dir_branches(root_dir, main_path=HAMLET_ROOT,
                     remove_root_p=True, match_dict=None,
                     select_subdirs_verbose=False):
    """
    Uses os.walk to walk directory tree under root_dir and collects
    subdir branches
    :param root_dir:
    :param remove_root_p:
    :return: list of subdir branches below root_dir
    """

    # select_subdirs_verbose = True

    dir_branches = []

    owd = os.getcwd()
    os.chdir(main_path)

    for dirName, subdirList, fileList in os.walk(root_dir):
        if not subdirList:
            if remove_root_p:
                # print 'dirName:', dirName, 'subdirList:', subdirList, 'fileList:', fileList
                without_root = dirName.replace(root_dir, '')  # + '/'
                # print 'without_root', without_root
                dir_branches.append(without_root)
            else:
                dir_branches.append(dirName)  #  + '/')

    os.chdir(owd)

    if select_subdirs_verbose:
        print 'dir_branches before selection:'
        for dir_branch in dir_branches:
            print '  {0}'.format(dir_branch)

    # select subset according to match_dict pattern
    # dir_branches = select_subdirs(dir_branches, match_dict=match_dict, verbose=select_subdirs_verbose)
    dir_branches = select_subdirs(dir_branches, match_dict=match_dict, verbose=True)

    # print '-----'
    # for db in dir_branches:
    #     print db

    if select_subdirs_verbose:
        print 'dir_branches after selection:'
        for dir_branch in dir_branches:
            print '  {0}'.format(dir_branch)

    return dir_branches


def test_get_dir_branches():
    match_select_h_cp0to2 = { 0: ( 'h0.5', 'h0.75' ),
                              1: ( 'cp0', 'cp1' ) }
    dir_branches = get_dir_branches('figures/cocktail/',
                                    remove_root_p=True,
                                    match_dict=match_select_h_cp0to2,
                                    main_path='../')

    print 'test_get_dir_branches(): ',
    for branch in dir_branches:
        print branch,

    print 'DONE test_get_dir_branches()'

# TODO reinstate
# test_get_dir_branches()


def collect_data_spec_list(main_path=HAMLET_ROOT,
                           data_dir=None,
                           weights_p=False, match_dict=None,
                           select_subdirs_verbose=False):
    data_spec_list = []

    data_directories = get_dir_branches(data_dir,
                                        main_path=main_path,
                                        remove_root_p=True,
                                        match_dict=match_dict,
                                        select_subdirs_verbose=select_subdirs_verbose)

    for data_subdir in data_directories:
        weights_file = None
        if weights_p:
            weights_file = data_dir + data_subdir + 'weights.txt'
        data_spec_list.append(DataSpec(data_dir, data_subdir, weights_file))

    return data_spec_list


def test_collect_data_spec_list_synth():
    match_select_synth = {0: ['BFact', 'LT', 'noLT'],
                          1: ['s{0}'.format(i) for i in range(3)]}

    # print 'data_spec_list:'
    # print collect_data_spec_list('figures/synth', match_dict=match_select_synth)

    print 'get_dir_branches'
    print get_dir_branches('figures/synth',
                           main_path=HAMLET_ROOT,
                           match_dict=match_select_synth)

# test_collect_data_spec_list_synth()


# -------------------------------


ResultsSpec = util.namedtuple_with_defaults\
    ('ResultsSpec', ['results_subdir', 'results_postfix', 'results_dir'])

result_name_abbreviation = \
    {'HMM': 'hmm', 'HSMM': 'hsmm',
     'HDP': 'hdp', 'Dirichlet': 'dir',
     'Known': 'w0', 'IID_normal': 'w1',
     'Binary_factorial': 'BFact'}


def lt_p(params):
    if 'Isotropic_exponential_similarity:lambda' in params \
            and params['Isotropic_exponential_similarity:lambda'] == '0.0':
        return False
    return True


def results_spec_fn(results_dir, pspec, dspec, replication_postfix,
                    main_path=HAMLET_ROOT):
    """
    Given ParameterSpec and DataSpec, constructs ResultsSpec
    specifying: results_subdir, results_postfix, results_dir
    <results_dir>/<results_subdir>
    <results_subdir> := <results_root_name>/<data_subdir>/<model_subdir>/<postfix>

    :param pspec: ParameterSpec
    :param dspec: DataSpec
    :param replication_postfix: integer representing replication num
    :return: ResultsSpec
    """

    results_root_name = results_dir  # dspec.data_dir[0:-1].split('/')[-1]

    if dspec.data_subdir[-1] == '/':
        data_subdir = dspec.data_subdir[0:-1].replace('/', '_')
    else:
        data_subdir = dspec.data_subdir.replace('/', '_')

    pdir = 'parameters/'
    if pspec.parameters_dir:
        # print 'pspec.parameters_dir', pspec.parameters_dir
        pdir = pspec.parameters_dir

    owd = os.getcwd()
    os.chdir(main_path)

    params = util.read_parameter_file_as_dict(pdir, pspec.parameters_file)

    os.chdir(owd)

    dynamics = result_name_abbreviation[params[':MODULE:DYNAMICS']]
    trans_prior = result_name_abbreviation[params[':MODULE:TRANSITION_PRIOR']]
    weights_learned = result_name_abbreviation[params[':MODULE:WEIGHTS_PRIOR']]

    model_subdir = list()

    factorial_p = False
    '''
    if params[':MODULE:TRANSITION_PRIOR'] == 'Binary_factorial':
        factorial_p = True
    '''

    if ':MODEL:MODEL_TYPE' in params and params[':MODEL:MODEL_TYPE'] == 'FACTORIAL':
        factorial_p = True
        trans_prior = 'BFact'

    if factorial_p:
        pass
    elif lt_p(params):
        model_subdir.append('LT')
    else:
        model_subdir.append('noLT')

    model_subdir += [trans_prior, dynamics]
    if weights_learned:
        model_subdir += [ weights_learned ]
    model_subdir = '_'.join(model_subdir)

    # factorial_p = False
    # # add extra annotation for factorial model
    # if params[':MODULE:TRANSITION_PRIOR'] == 'Binary_factorial':
    #     factorial_p = True
    #     fm = '1'
    #     if 'Binary_factorial:fixed_mean' in params:
    #         fm = params['Binary_factorial:fixed_mean']
    #     if fm == '1':
    #         if 'Binary_factorial:p_mean' in params:
    #             p_mean = params['Binary_factorial:p_mean']
    #         else:
    #             print 'ERROR: was expecting Binary_factorial:p_mean, not found'
    #             sys.exit(-1)
    #         model_subdir += '_fm{0}'.format(p_mean)
    #     else:
    #         model_subdir += '_fmHP'  # indicating using HDP hyperprior params

    # if 'Normal_noise_model:a_h' in params:
    #     if params['Normal_noise_model:a_h'] == '0.1':
    #         model_subdir += '_Nemh01'

    results_subdir = '/'.join([results_root_name, data_subdir, model_subdir])

    results_postfix = '{0:0>2}'.format(replication_postfix)

    return ResultsSpec(results_subdir=results_subdir, results_postfix=results_postfix)


# TODO: 20160619 - refactoring broke this test, fix...
def test_results_spec_fn():
    pspec = ParameterSpec('cocktail_inference_LT_HMM_W0.config', parameters_root)
    dspec = DataSpec(data_dir=os.path.join(data_root, 'cocktail'),
                     data_subdir='a3b6/cp2/',
                     weights_file=os.path.join(data_root, 'cocktail/a3b6/cp2/weights.txt'))
    rspec = results_spec_fn(results_root, pspec, dspec, 5)

    # print rspec.results_dir, rspec.results_subdir, rspec.results_postfix

    assert rspec.results_dir is None

    print 'rspec.results_subdir', rspec.results_subdir  ################
    assert rspec.results_subdir == 'cocktail/a3b6_cp2/LT_hdp_hmm_w0'
    assert rspec.results_postfix == '05'

    pspec = ParameterSpec('cocktail_inference_noLT_HSMM_W0.config', parameters_root)
    dspec = DataSpec(data_dir=os.path.join(data_root, 'cocktail'),
                     data_subdir='a3b6/cp2/',
                     weights_file=os.path.join(data_root, 'cocktail/a3b6/cp2/weights.txt'))
    rspec = results_spec_fn(pspec, dspec, 2)

    # print rspec.results_dir, rspec.results_subdir, rspec.results_postfix

    assert rspec.results_dir is None
    # print 'rspec.results_subdir', rspec.results_subdir
    assert rspec.results_subdir == 'cocktail/a3b6_cp2/noLT_hdp_hsmm_w0'
    assert rspec.results_postfix == '02'

    print 'PASS test_results_spec_fn()'

# test_results_spec_fn()

# -------------------------------


def collect_experiment_spec_list(parameter_spec_list,
                                 data_spec_list,
                                 results_dir,
                                 results_spec_fn=results_spec_fn,
                                 replications=1,
                                 offset=0,
                                 main_path=HAMLET_ROOT):

    experiment_spec_list = []
    for r in range(1, replications + 1):
        for dspec in data_spec_list:
            for pspec in parameter_spec_list:

                rspec = results_spec_fn(results_dir, pspec, dspec, r + offset,
                                        main_path)

                experiment_spec_list.append\
                    (ExperimentSpec
                     (parameters_file=pspec.parameters_file,  # -p
                      parameters_dir=pspec.parameters_dir,    # --parameters_dir

                      data_subdir=dspec.data_subdir,          # --data_subdir
                      data_dir=dspec.data_dir,                # --data_dir
                      weights_file=dspec.weights_file,        # --weights_file

                      results_subdir=rspec.results_subdir,    # -r
                      results_postfix=rspec.results_postfix,  # --results_timestamp
                      results_dir=rspec.results_dir           # --results_dir
                      ) )

    total_exps = len(experiment_spec_list)

    for spec, exp_num in zip(experiment_spec_list, range(1, total_exps + 1)):
        spec.exp_num = exp_num
        spec.total_exps = total_exps

    return experiment_spec_list


# -------------------------------


def run_experiment_script(main_path=HAMLET_ROOT,
                          data_dir=None,
                          results_dir=None,
                          replications=1,
                          offset=0,
                          parameter_spec_list=None,
                          # parameter_spec_collector=collect_parameter_spec_list_cp_W0,
                          match_dict=None,
                          multiproc=True,
                          processor_pool_size=multiprocessing.cpu_count(),
                          rerun=True,
                          test=True,
                          select_subdirs_verbose=False):

    # parameter_spec_list = parameter_spec_collector()

    if test:
        print 'parameter_spec_list:'
        for i, spec in enumerate(parameter_spec_list):
            print '    [{0}] : {1}'.format(i, spec)

    data_spec_list = collect_data_spec_list(main_path=main_path,
                                            data_dir=data_dir,
                                            match_dict=match_dict,
                                            select_subdirs_verbose=select_subdirs_verbose)

    if test:
        print 'data_spec_list:'
        for i, spec in enumerate(data_spec_list):
            print '    [{0}] : {1}'.format(i, spec)

    spec_list = collect_experiment_spec_list \
        (parameter_spec_list=parameter_spec_list,
         data_spec_list=data_spec_list,
         results_dir=results_dir,
         results_spec_fn=results_spec_fn,
         replications=replications,
         offset=offset,
         main_path=main_path)

    if test:

        print 'TOTAL EXPERIMENTS: {0}'.format(len(spec_list))

        print 'experiment_spec_list:'
        for i, ps in enumerate(spec_list):
            print '    [{0}] : {1}'.format(i, ps)

        #for spec in spec_list:
        #    print '{0}'.format(spec)

    run_experiment_wrapper \
        (spec_list,
         main_path=main_path,
         multiproc=multiproc,
         processor_pool_size=processor_pool_size,
         log_file='exp_run',
         rerun=rerun,
         test=test)


'''
match_select_cp0to2 = { 0: ( 'a6b10', 'a20b76' ),
                        1: [ 'cp{0}'.format(i) for i in range(3) ] }
# match_select_cp3to9 = { 1: [ 'cp{0}'.format(i) for i in range(3,10) ] }

run_experiment_script('figures/cocktail/', # 'figures/cocktailNarrow7/',  # 'figures/cocktailNarrow4/',  # 'figures/cocktail/'
       replications=5,
       offset=0,
       match_dict=match_select_cp0to2, # match_select_cp3to9
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True)
'''


'''
match_select_h_cp0to2 = { 0: ( 'a1b1' ),
                          1: [ 'cp{0}'.format(i) for i in range(10) ] }

# match_select_cp3to9 = { 1: [ 'cp{0}'.format(i) for i in range(3,10) ] }

run_experiment_script('figures/cocktail_rw/',  # 'figures/cocktailNarrow7/',  # 'figures/cocktailNarrow4/',  # 'figures/cocktail/'
       replications=5,
       offset=0,
       match_dict=match_select_h_cp0to2,  # match_select_cp3to9
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True)
'''

'''
match_select_h_cp0to2 = { 0: ( 'h0.5', 'h0.75', 'h1.0', 'h1.5', 'h2.0', 'h3.0', 'h5.0', 'h10.0' ),
                          1: [ 'cp{0}'.format(i) for i in range(3) ] }

# match_select_cp3to9 = { 1: [ 'cp{0}'.format(i) for i in range(3,10) ] }

run_experiment_script('figures/cocktail/',  # 'figures/cocktailNarrow7/',  # 'figures/cocktailNarrow4/',  # 'figures/cocktail/'
       replications=5,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W0,
       match_dict=match_select_h_cp0to2,  # match_select_cp3to9
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True)
'''


'''
match_select_a1b1new_cp0to2 = { 0: ( 'a1b1_nocs' ),
                                1: [ 'cp{0}'.format(i) for i in range(3) ] }

run_experiment_script('figures/cocktail/',
       replications=1,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W0,
       match_dict=match_select_a1b1new_cp0to2,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


'''
# h experiment

# >>>>>>>> NOTE THE RANGE IN cp{0} <<<<<<<<<<<<<
# //// >>>>>>>> generating 03..06 (4x)  <<<<<<<<<<<<<
# >>>>>>>> generating 07..09 (3x)  <<<<<<<<<<<<<

match_select_hnocs_cp0to2 = {0: ['h{0}_nocs'.format(h)
                                 for h in [0.5]],  # [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                             1: ['cp{0}'.format(i) for i in range(1)]}  # >>>NOTE range<<<

run_experiment_script('figures/cocktail/',
       replications=1,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W0,
       match_dict=match_select_hnocs_cp0to2,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


'''
# normal, varying J experiment

match_select_normalJ_cp0to2 = {0: ['HMM_HDP_N_J{0}_s1.0_L0.0'.format(J)
                                   for J in [3, 5, 7, 12]],
                               1: ['n{0}'.format(i) for i in range(3)]}

run_experiment_script('figures/normal/',
       replications=5,
       offset=0,
       match_dict=match_select_normalJ_cp0to2,
       parameter_spec_collector=HMM_noLT_parameter_spec_list,  # HMM_LT_parameter_spec_list,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=False,
       select_subdirs_verbose=False)
'''


'''
# Learn weights
# [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
match_select_hnocs_cp0to2 = {0: ['h{0}_nocs'.format(h)
                                 for h in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                             1: ['cp{0}'.format(i) for i in range(0, 10)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=10,
       offset=10,  # ADDITIONAL runs
       parameter_spec_collector=collect_parameter_spec_list_cp_W1_1500,  # learn weights
       match_dict=match_select_hnocs_cp0to2,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


'''
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Learn weights with Normal_emission_model a_h=b_h=0.1
match_select_hnocs_cp0to2 = {0: ['h{0}_nocs'.format(h)
                                 for h in [10.0]],  # [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                             1: ['cp{0}'.format(i) for i in range(1)]}  # range(0, 10)

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=1,  # replications = 10
       offset=0,  # offset = 10 # for _additional_ runs
       parameter_spec_collector=collect_parameter_spec_list_cp_W1_1500_Nemh01,  # learn weights, Nem a_h=b_h=0.1
       match_dict=match_select_hnocs_cp0to2,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


# factorial
'''
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [0.5]],  # , 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                      1: ['cp{0}'.format(i) for i in range(10)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=5,
       offset=0,
       parameter_spec_collector=factorial_parameter_spec_list_W0,  # learn weights
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''

# factorial w1
'''
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                      1: ['cp{0}'.format(i) for i in range(10)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=10,
       offset=0,
       parameter_spec_collector=factorial_parameter_spec_list_W1,  # learn weights
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''

'''
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],
                      1: ['cp{0}'.format(i) for i in range(10)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=10,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W1,
       #collect_parameter_spec_list_cp_BFact_only_W1,
       # collect_parameter_spec_list_cp_W1,
       # collect_parameter_spec_list_cp_BFact_only_W0,
       # collect_parameter_spec_list_cp_W0,
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''

'''
# w1 long run: 4000 iterations
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [2.0, 3.0, 5.0, 10.0]],
                      1: ['cp{0}'.format(i) for i in range(3)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail'),
       replications=10,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W1_4000,
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''

'''
# 14-speaker 5,5,4 cocktail party, w1 long run: 4000 iterations, h2.0 only
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [2.0]],
                      1: ['cp{0}'.format(i) for i in range(3)]}

run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail_s14_m12'),
       replications=10,
       offset=0,
       parameter_spec_collector=collect_parameter_spec_list_cp_W1_4000_D14_J250,
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


'''
# 16-speaker 4x4 cocktail party, w1 long run: 4000 iterations, h2.0 only
match_select_hnocs = {0: ['h{0}_nocs'.format(h)
                          for h in [3.0, 10.0]],
                      1: ['cp{0}'.format(i) for i in range(3)]}

# {BFact, LT, noLT} x {h3.0, h10.0} x {cp0..cp3} x {3 reps} = 3 x 2 x 3 x 3 = 54
run_experiment_script(os.path.join(HAMLET_ROOT, 'cocktail_s16_m12',
       replications=3,
       offset=2,
       # collect_parameter_spec_list_cp_W1_4000_D16,
       # collect_parameter_spec_list_cp_W0_1500_D16,
       parameter_spec_collector=collect_parameter_spec_list_cp_W0_2000_D16_hmmJ600,
       match_dict=match_select_hnocs,
       multiproc=True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


'''
#
match_select_synth = {0: ['BFact', 'LT', 'noLT'],
                      1: ['s{0}'.format(i) for i in range(3)]}

# Synthetic experiment
run_experiment_script(os.path.join(HAMLET_ROOT, 'synth',
       replications=20,
       offset=0,
       # collect_parameter_spec_list_cp_W1_4000_D16,
       # collect_parameter_spec_list_cp_W0_1500_D16,
       parameter_spec_collector=collect_parameter_spec_list_synth,
       match_dict=match_select_synth,
       multiproc=False,  # True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''

'''
#
match_select_synth = {0: ['BFact', 'LT', 'noLT'],
                      1: ['s{0}'.format(i) for i in range(1)]}

# Synthetic 16 experiment
run_experiment_script(os.path.join(data_root, 'synth16'),
       results_dir=results_root,
       main_path=HAMLET_ROOT,
       replications=4,
       offset=0,
       # collect_parameter_spec_list_cp_W1_4000_D16,
       # collect_parameter_spec_list_cp_W0_1500_D16,
       parameter_spec_list=collect_parameter_spec_list_synth16(parameters_root),
       match_dict=match_select_synth,
       multiproc=False,  # True,
       processor_pool_size=multiprocessing.cpu_count(),
       rerun=False,
       test=True,
       select_subdirs_verbose=False)
'''


# ----------------------------------------------------------------------

# rerun_experiment('exp_run_20150609_223203867358.log', test=True)


if __name__ == '__main__':
    # test to make sure can access fns in utilities.util
    print 'Running __main__ in experiment_tools.py'
    print 'Executing from:', os.getcwd()
    hello()       # self
    util.hello()  # relative to utilities
    print 'DONE'
