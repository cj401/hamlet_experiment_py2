__author__ = 'clayton'


"""
NOTE (CTM:20160618): This facility needs major reworking


Provides functionality to generate figures using the same parameters as
the provided existing figures, but uses seed none.
Will add 'r#' to name (increments based on how many of the base name exists)

This code makes a lot of assumptions WITHOUT TESTING, esp re. the form of the source
figures directory name:  <base-name>_{<r#>_}<date-timestamp>_<time-timestamp>
( {...} means optional: zero or one)

"""

import glob
import os
import sys
from subprocess import call

from utilities.util import get_timestamp


def get_file_from_path(path):
    return path.split('/')[-1]


def get_data_file_name(data_target, data_path):
    second_idx = data_target.rfind('_', 0, data_target.rfind('_'))
    data_target_basename = data_target[0:second_idx]

    # print 'data_target_basename:', data_target_basename
    # print ':', data_path + data_target_basename + '*'
    # print 'glob:', glob.glob(data_path + data_target_basename + '*')

    repetitions = len(glob.glob(data_path + data_target_basename + '*'))
    if repetitions >= 1:
        data_target_basename += '_r{0}'.format(repetitions-1)

    return data_target_basename


def regenerate_data(data_source, data_dir='figures/', main_path='../'):

    print 'data_path:        ', data_dir
    print 'data_source:      ', data_source

    owd = os.getcwd()
    os.chdir(main_path)
    target_data_path = data_dir + data_source + '/'
    if not os.path.isdir(target_data_path):
        print "ERROR: Could not find target figures at path:"
        print '    \'{0}\''.format(target_data_path)
        sys.exit()

    data_file_name = get_data_file_name(data_source, data_dir)

    parameters_file = glob.glob(target_data_path + '*.config')
    if not parameters_file:
        print "ERROR: Could not find *.config parameters file at path:"
        print '    \'{0}\''.format(target_data_path)
        sys.exit()
    if len(parameters_file) > 1:
        print "ERROR: Found {0} *.config parameters files at path:".format(len(parameters_file))
        print '    \'{0}\''.format(target_data_path)
        i = 0
        for pfile in parameters_file:
            print '({0}) {1}'.format(i, pfile)
            ++i
        sys.exit()

    parameters_file = get_file_from_path(parameters_file[0])
    data_timestamp = get_timestamp()

    print 'parameters_file:  ', parameters_file
    print 'parameters_dir:   ', target_data_path
    print 'figures (file_name): ', data_file_name
    print 'data_timestamp:   ', data_timestamp

    command = './main -p {parameters_file} -g --parameters_dir={target_data_path}' \
        .format(parameters_file=parameters_file, target_data_path=target_data_path)
    command += ' --data_dir={data_dir}'.format(data_dir=data_dir)
    command += ' --figures={data_file_name} --data_timestamp={data_timestamp}' \
        .format(data_file_name=data_file_name, data_timestamp=data_timestamp)
    command += ' --seed=none'

    print 'calling:\n', command

    print '-----'
    ret = call(command, shell=True)
    print '-----'

    os.chdir(owd)
    print 'return value: {0}'.format(ret)

    print 'cwd', os.getcwd()
    print 'DONE'


print 'NEEDS REVISITING since changes to generation'

# regenerate_data('HMM_HDP_P_s0.1_spec20150521_20150522_225215285764')