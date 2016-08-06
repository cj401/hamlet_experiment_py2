import datetime
import glob
import itertools
import multiprocessing
import os
import shutil
import sys
from subprocess import call

from utilities.util import get_timestamp, read_parameter_file_as_dict

__author__ = 'clayton'


lock = None  # Global definition of lock


class DataGenerationSpec:
    def __init__(self,
                 parameter_file_name=None,
                 parameters_dir=None,
                 data_dir=None,
                 data_file_base_name=None,
                 timestamp=None,

                 main_path=None,

                 log_file=None,
                 gen_num=None,
                 total_gens=None,
                 test=True):
        self.parameter_file_name = parameter_file_name
        self.parameters_dir = parameters_dir
        self.data_dir = data_dir
        self.data_file_base_name = data_file_base_name
        self.timestamp = timestamp

        self.main_path = main_path

        self.log_file = log_file
        self.gen_num = gen_num
        self.total_gens = total_gens
        self.test = test

    def pprint(self):
        print 'DataGenerationSpec(parameter_file=\'{0}\', parameters_dir=\'{1}\', data_dir=\'{2}\','\
            .format(self.parameter_file_name, self.parameters_dir, self.data_dir)
        print '                   data_file_base_name=\'{0}\', timestamp=\'{1}\', main_path=\'{2}\','\
            .format(self.data_file_base_name, self.timestamp, self.main_path)
        print '                   log_file=\'{0}\', gen_num=\'{1}\', total_gens=\'{2}\', test=\'{3}\''\
            .format(self.log_file, self.gen_num, self.total_gens, self.test)


def execute_main_generate(spec):

    global lock

    # unpack args
    # parameter_file, parameters_dir, data_dir, timestamp, main_path, \
    #     log_file, gen_num, total_gens, test = params

    command = './main -p {0} --parameters_dir={1} -g'\
        .format(spec.parameter_file_name, spec.parameters_dir)
    if spec.data_file_base_name:
        command += " -d {0}".format(spec.data_file_base_name)
    if spec.data_dir:
        command += " --data_dir={0}".format(spec.data_dir)
    if spec.timestamp:
        command += " --timestamp={0}".format(spec.timestamp)
    else:
        command += " --timestamp=0"

    log_message = '[{0} of {1}] \'{2}\' ... '.format(spec.gen_num, spec.total_gens, command)

    if spec.test:

        lock.acquire()
        print log_message
        spec.pprint()
        lock.release()

    else:

        owd = os.getcwd()
        os.chdir(spec.main_path)

        start = datetime.datetime.now()
        ret = call(command, shell=True)
        end = datetime.datetime.now()

        os.chdir(owd)

        lock.acquire()
        log_message += '{0} {1}\n'.format(ret, end-start)
        with open(spec.log_file,'a') as logf:
            logf.write(log_message)
        lock.release()

        return spec.gen_num, ret


def get_parameters_list_from_directory(parameters_dir='parameters/', main_path='../'):

    # print 'parameters_dir', parameters_dir

    owd = os.getcwd()
    os.chdir(main_path)
    files = glob.glob(parameters_dir + '*.config')

    # print 'files', files

    os.chdir(owd)
    files = map(lambda v: v.split('/')[-1], files)  # extract just the config file

    # print 'files', files

    return files


# parameter_file, augmented_parameters_dir, timestamp, augmented_data_dir


def get_parameter_spec_list(parameters_dir, parameter_file, data_dir, repetitions,
                            offset=0, timestamp_p=False, main_path='../'):

    parameter_spec_list = list()

    if parameters_dir[-1] != '/':
        parameters_dir += '/'
    if data_dir[-1] != '/':
        data_dir += '/'

    if parameter_file:
        parameter_files = [ parameter_file + '.config' ]
    else:
        parameter_files = get_parameters_list_from_directory(parameters_dir, main_path)

    for rep in range(offset, repetitions + offset):

        for parameter_file_name in parameter_files:

            param_dict = read_parameter_file_as_dict(parameters_dir, parameter_file_name, main_path=main_path)

            data_dir_subdir = data_dir

            if ':experiment:data_file_name' in param_dict:
                data_dir_subdir += param_dict[':experiment:data_file_name']
                if data_dir_subdir[-1] != '/':
                    data_dir_subdir += '/'

            data_file_base_name = ''

            # print param_dict.keys()

            if ':MODULE:NOISE' in param_dict:
                if param_dict[':MODULE:NOISE'] == 'Normal':
                    data_file_base_name = 'n'
                elif param_dict[':MODULE:NOISE'] == 'Probit':
                    data_file_base_name = 'p'
            data_file_base_name += '{0}'.format(rep)

            parameter_spec_list.append(DataGenerationSpec(parameter_file_name=parameter_file_name,
                                                          parameters_dir=parameters_dir,
                                                          data_dir=data_dir_subdir,
                                                          data_file_base_name=data_file_base_name))

    # figures/<data_file_name>/<data_file_base_name>#
    # parameter_file_name, parameters_dir, data_dir, data_file_base_name, timestamp

    return parameter_spec_list


def generate_data(parameters_dir='parameters/generate_normal/',
                  parameter_file=None,  # specify particular param file, else iterate over all in parameters_dir...
                  data_dir='figures/normal/',

                  repetitions=1,
                  offset=0,

                  timestamp_p=False,

                  log_file='data_generation',

                  main_path='../',
                  test=True,
                  multiproc=True,
                  proc_pool_size=8):

    global lock

    lock = multiprocessing.Lock()

    log_file += '_' + get_timestamp() + '.log'

    if test: print 'RUNNING in TEST MODE'

    parameter_spec_list = get_parameter_spec_list(parameters_dir, parameter_file, data_dir,
                                                  repetitions, offset,
                                                  timestamp_p, main_path)

    total_gens = len(parameter_spec_list)

    for gen_num, spec in enumerate(parameter_spec_list):
        spec.gen_num = gen_num
        spec.total_gens = total_gens
        spec.log_file = log_file
        spec.main_path = main_path
        spec.test = test

    results = ''
    start = datetime.datetime.now()
    if multiproc:

        print 'Multiprocessing'
        lock.acquire()
        with open(log_file, 'a') as logf:
            logf.write('Multiprocessing ON\n')
        lock.release()

        p = multiprocessing.Pool(proc_pool_size)
        results = p.map(execute_main_generate, parameter_spec_list)

    else:

        print 'Single process'
        lock.acquire()
        with open(log_file, 'a') as logf:
            logf.write('Single process\n')
        lock.release()

        for spec in parameter_spec_list:
            execute_main_generate(spec)

    end = datetime.datetime.now()

    '''
    for parameter_file in parameter_files:
        repeat_main_generate(parameter_file, repetitions, main_path=main_path,
                             parameters_dir=main_relative_parameters_dir, pause=pause,
                             log_file=log_file)
    '''

    lock.acquire()
    with open(log_file, 'a') as logf:
        if results:
            logf.write('Results: {0}\n'.format(results))
        logf.write('Total time: {0}\n'.format(end-start))
    lock.release()

    if results: print results
    print "Total time: {0}".format(end-start)
    print "DONE."


### script

# figures/<noise_precision>_nocs/cp0

'''
generate_data(parameters_dir='parameters/generate_normal/HMM/',

              # specify particular .config file (but without extension),
              # else, if None, iterate over all .config files in parameters_dir
              parameter_file=None,  # 'HSMM_HDP_N_J5_s1.0_L0.0',  # None,

              data_dir='figures/normal/',

              repetitions=3,
              offset=0,  # num to start repetitions with; e.g., if want to generate starting at 3, offset=3

              timestamp_p=False,

              log_file='data_generation',

              main_path='../',
              test=False,

              multiproc=False,
              proc_pool_size=8
              )
'''


# results/cocktail/<noise_precision>_<no_center_scale>_cp#/<hmm>_<hdp>_w0/<LT>_<inference-#>/
# results/normal/


# ----------------------------------------------------------------------
# Old version --- DEPRICATE

'''
def get_parameters_and_stamps(parameters_dir,
                              repetitions=1,
                              data_dir=None,
                              subdir_postfix=None,
                              timestamp_p=True,
                              main_path='../'):

    augmented_parameters_dir = parameters_dir
    augmented_data_dir = data_dir
    if subdir_postfix:
        augmented_parameters_dir += subdir_postfix
        augmented_data_dir += subdir_postfix

    # print 'augmented_parameters_dir', augmented_parameters_dir

    parameter_files = get_parameters_list_from_directory(augmented_parameters_dir, main_path)

    # print 'parameter_files:', parameter_files

    if timestamp_p:
        timestamps = [get_timestamp()]
        if repetitions > 1:
            timestamps = ['r{0}_{1}'.format(i, get_timestamp()) for i in range(repetitions)]
    else:
        timestamps = ['1']
        if repetitions > 1:
            timestamps = ['{0}'.format(i) for i in range(repetitions)]

    param_stamp_data_set = [(parameter_file, augmented_parameters_dir, timestamp, augmented_data_dir)
                            for parameter_file, timestamp
                            in list(itertools.product(parameter_files, timestamps)) ]

    return param_stamp_data_set


def generate_data_old(parameters_dir,
                      data_dir=None,
                      subdir_postfix_list=None,
                      repetitions=1,
                      timestamp_p=True,
                      main_path='../',
                      log_file='data_generation',
                      test=True,
                      multiproc=False,
                      proc_pool_size=8):

    global lock

    lock = multiprocessing.Lock()

    log_file += '_' + get_timestamp() + '.log'

    if test: print 'RUNNING in TEST MODE'

    param_stamp_data_set = []
    if subdir_postfix_list:
        for subdir_postfix in subdir_postfix_list:
            param_stamp_data_set += get_parameters_and_stamps(parameters_dir,
                                                              repetitions,
                                                              data_dir,
                                                              subdir_postfix,
                                                              timestamp_p,
                                                              main_path)
    else:
        param_stamp_data_set = get_parameters_and_stamps(parameters_dir,
                                                         repetitions,
                                                         data_dir,
                                                         '',
                                                         timestamp_p,
                                                         main_path)

    # print param_stamp_data_set

    total_gens = len(param_stamp_data_set)
    gen_nums = range(1,total_gens+1)

    args = [(parameter_file, parameters_dir, augmented_data_dir, timestamp, main_path,
             log_file, gen_num, total_gens, test)
            for (parameter_file, parameters_dir, timestamp, augmented_data_dir), gen_num
            in zip(param_stamp_data_set, gen_nums) ]

    results = ''
    start = datetime.datetime.now()
    if multiproc:

        print 'Multiprocessing'
        lock.acquire()
        with open(log_file, 'a') as logf:
            logf.write('Multiprocessing ON\n')
        lock.release()

        p = multiprocessing.Pool(proc_pool_size)
        results = p.map(execute_main_generate, args)

    else:

        print 'Single process'
        lock.acquire()
        with open(log_file, 'a') as logf:
            logf.write('Single process\n')
        lock.release()

        for arg in args:
            execute_main_generate(arg)

    end = datetime.datetime.now()

    ''
    for parameter_file in parameter_files:
        repeat_main_generate(parameter_file, repetitions, main_path=main_path,
                             parameters_dir=main_relative_parameters_dir, pause=pause,
                             log_file=log_file)
    ''

    lock.acquire()
    with open(log_file, 'a') as logf:
        if results:
            logf.write('Results: {0}\n'.format(results))
        logf.write('Total time: {0}\n'.format(end-start))
    lock.release()

    if results: print results
    print "Total time: {0}".format(end-start)
    print "DONE."

'''

### Script


# print repeat_main_generate('HMM_HDP_N_spec20150521.config', 10, parameters_dir='experiment/parameters/')


'''
generate_data_old(parameters_dir='parameters/generate_normal/',   # 'experiment/parameters/figures/',
              data_dir='figures/normal/',

              # generally for specifying emission type,
              # but could be used as an arbitrary list of subdirs
              # shared across parameters sources and figures output
              # subdir_postfix_list=['generate_normal/', ],  # 'probit/'],

              repetitions=1,

              timestamp_p=False,

              log_file='data_generation',
              main_path='../',
              test=True,
              multiproc=True,
              proc_pool_size=8
              )
'''

'''
generate_data_old(parameters_dir='experiment/parameters/figures/',
              data_dir='figures/20150602/',

              # generally for specifying emission type,
              # but could be used as an arbitrary list of subdirs
              # shared across parameters sources and figures output
              subdir_postfix_list=['normal/',], # 'probit/'],

              repetitions=1,

              log_file='data_generation',

              # path python current working directory (cwd) will be changed
              # to for running run_experiment_script
              main_path='../',

              test=False,
              multiproc=True,
              proc_pool_size=40
              )
'''

# ---------------------------------------------------------------------
# Copy narrow


def generate_paths(root, noise):
    # 'noise_a2b3', 'noise_a2b6'

    cpdir = [ 'cp{0}'.format(n) for n in range(10) ]

    paths = [ '/'.join( [ root, n, c ] ) + '/'
              for n, c
              in list(itertools.product(noise, cpdir)) ]

    return paths


def copy_data_narrow(source_dir, destination_dir, indices, data_root='../figures/', test=True):

    def copy_file_narrow(filename):
        with open(destination_dir + filename, 'w') as fout:
            with open(source_dir + filename, 'r') as fin:
                for line in fin.readlines():
                    vals = line.split()
                    for idx in indices:
                        fout.write(' {0}'.format(vals[idx]))
                    fout.write('\n')

    '''
    def copy_file_narrow(filename):
        with open(destination_dir + filename, 'w') as fout:
            with open(source_dir + filename, 'r') as fin:
                for line in fin.readlines():
                    vals = line.split()
                    for v, i in zip(vals, range(width)):
                        fout.write(' {0}'.format(v))
                    fout.write('\n')
    '''

    if test:
        print 'copy_data_narrow() - TEST'
        print '    data_root:      ', data_root
        print '    source_dir:     ', source_dir
        print '    destination_dir:', destination_dir

    owd = os.getcwd()
    os.chdir(data_root)

    print 'CWD: {0}'.format(os.getcwd())

    # ensure directories exist
    if not os.path.exists(source_dir):
        print 'ERROR: source dir does not exist: {0}'.format(source_dir)
        sys.exit(1)

    if test:
        if not os.path.exists(destination_dir):
            print 'TEST Would create dir: {0}'.format(destination_dir)
    else:
        if not os.path.exists(destination_dir):
            print 'Creating directory: {0}'.format(destination_dir)
            os.makedirs(destination_dir)

        # copy states.txt
        shutil.copyfile(source_dir + 'states.txt', destination_dir + 'states.txt')

        copy_file_narrow('obs.txt')
        copy_file_narrow('test_obs.txt')
        copy_file_narrow('weights.txt')

    os.chdir(owd)


def copy_cocktail_narrow(data_root='../figures/', indices=(0,1,2,3), noise=None, test=True):
    source_dirs = generate_paths('cocktail', noise=noise)
    destination_dirs = generate_paths('cocktailNarrow{0}'.format(len(indices)), noise=noise)

    for source_dir, destination_dir in zip(source_dirs, destination_dirs):
        copy_data_narrow(source_dir, destination_dir, indices, data_root, test)



# noise = ( 'a1b1', 'a3b2', 'a3b6' )
# copy_cocktail_narrow(noise=('a6b10', 'a20b76'), test=True)

# copy_cocktail_narrow(indices=range(4), noise=('a6b10', 'a20b76'), test=False)
# copy_cocktail_narrow(indices=range(7), noise=('a6b10', 'a20b76'), test=False)
# copy_cocktail_narrow(indices=range(10), noise=('a6b10', 'a20b76'), test=False)

'''
copy_data_narrow(source_dir='cocktail/a1b1/cp1/',
                 destination_dir='cocktailNarrow_0_6_8_10/a1b1/cp1/',
                 indices=(0, 6, 8, 10),
                 data_root='../figures/',
                 test=False)
'''
