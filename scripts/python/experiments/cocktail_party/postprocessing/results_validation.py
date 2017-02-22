import errno
import multiprocessing
import os
import shutil
import sys

from run import run_experiment
from utilities import util

__author__ = 'clayton'

"""

Parameter sources:

(*) T:
        obs.txt number of rows
(*) output vector size:
        obs.txt number of columns

(*) D:
        Binary_state_model D

(*) J:
        BFact Dirichlet_hyperprior J
        HDP   HDP_hyperprior J

(*) iteration == 0..10, then by 10 up to 1000
    "{0:0>5}".format(index)


Data to process:

(*) F1_score.txt, accuracy.txt, precision.txt, recall.txt,  (BFact, HDP_HMM, HDP_HSMM)
        test_log_likelihood, train_log_likelihood.txt,      (BFact, HDP_HMM, HSP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-1> <float or NaN>

(*) alpha.txt, gamma.txt, lambda.txt                        (HDP_HMM, HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <float or NaN>

(*) beta.txt                                                (HDP_HMM, HDP_HSMM)
        omega.txt                                           (HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of floats: J >

(*) u.txt                                                   (HDP_HMM, HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of floats: J+1 >

(*) n_dot.txt,                                              (HDP_HMM, HDP_HSMM)
        dtotals.txt                                         (HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of ints: length= J >

(*) noise_sd.txt                                                   (BFact, HDP_HMM, HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of floats: length= (output vector size), infer from number of obs.txt columns >

(*) W directory                                             (BFact, HDP-HMM, HDP_HSMM)
    if BFact, single file 'W.txt'
        else: one file per iteration-0
    each file contains <list of floats: cols= (output vector size), infer from number of obs.txt columns
                                        rows= D + 1 (latent state length + 1 bias)>

(*) mu.txt                                                  (HDP_HMM, HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of floats: length= D (latent vector size) >

(*) z.txt  (looks like it is tab-delimited)                 (BFact, HDP_HMM, HDP_HSMM)
    line 1   : iteration value
    lines >1 : <iteration-0> <list of integers: T + 1 -- has to be inferred from number of obs.txt rows >

(*) theta directory                                         (BFact, HDP_HMM, HDP_HSMM)
    one file per iteration-0
    each file contains only int 0 or 1 values -- if it is floats, perhaps then write as binary values (to save space)
        columns = D (latent vector size)
        rows = J (number of latent states)

    TODO: Currently turned off for Binary_Factorial:
        Proper approach: IF Binary_Factorial model, then by default combinatorial_theta = 1
            (even if it that doesn't show up in the config file)
           Binary_state_model combinatorial_theta 1
            or
           Binary_Factorial
        If combinatorial_theta is true,
            then 1 file called 'ground_truth.txt' with
                columns = D
                rows = 2^D

(*) thetastar directory                                     (BFact, HDP_HMM, HDP_HSMM)
    one file per iteration-0
    each file contains only int 0 or 1 values -- *** if it is floats, perhaps then write as binary values (to save space)
        columns = D (latent vector size)
        rows = T (number of observations in obs.txt)

(*) A directory,                                            (HDP_HMM, HDP_HSMM)
    one file per iteration-0
    each file contains list of floats - JxJ

(*) pi0 directory                                           (BFact, HDP_HMM, HDP_HSMM)
    one file per iteration-0
    each file contains list of floats - 1xJ

    NOTE:
        From Colin 10/22:
        pi0 is the initial distribution, so expected dimensions are 1 x J.
        There is a setting to use a fixed transition matrix and initial distribution,
        which is specified by the transition prior module being set to "Known_transition_matrix".

        However, in the BinaryFactorial case, transitions are handled by the "similarity model"
        instead (since, essentially, the factorial model is all "local transitions" and no "(H)DP"),
        and so Known_transistion_matrix is used as a placeholder for the transition module.
        So pi0 will not record real figures in any BinaryFactorial run.

        Ignore Binary_Factorial

() pi directory                                             (BFact, HDP_HMM, HDP_HSMM)
    EMPTY ??

"""


# ---------------------------------------------------------------


def get_iteration_strings(iterations, iter_start=0):
    iteration_strings_list = list()
    i = iter_start
    while (i < iterations) and (i < 10):
        iteration_strings_list.append('{0:0>5}'.format(i))
        i += 1
    while i <= iterations:
        iteration_strings_list.append('{0:0>5}'.format(i))
        i += 10
    return iteration_strings_list


# ---------------------------------------------------------------


def limit_string(s, limit=None):
    if limit and len(s) >= limit + 3:
        return s[:limit] + '...'
    else:
        return s


def test_limit_string():
    s = '12345678901234567890'
    assert limit_string(s) == '12345678901234567890'
    assert limit_string(s, 100) == '12345678901234567890'
    assert limit_string(s, 5) == '12345...'
    assert limit_string(s, 20) == '12345678901234567890'
    assert limit_string(s, 19) == '12345678901234567890'
    assert limit_string(s, 18) == '12345678901234567890'
    assert limit_string(s, 17) == '12345678901234567...'
    assert limit_string(s, 16) == '1234567890123456...'
    print 'TEST limit_string PASSED'


test_limit_string()


def limit_list(l, list_len_limit, list_elm_limit=None):
    if list_len_limit < len(l):
        gap = len(l) - list_len_limit + 1
        new_list = l[:list_len_limit - 2] + ['<{0}>'.format(gap)] + [l[-1]]
    else:
        new_list = l
    if list_elm_limit:
        return [elm[:list_elm_limit] + '...'
                if isinstance(elm, basestring) and len(elm) > list_elm_limit and elm != '...'
                else elm
                for elm in new_list]
    else:
        return new_list


def test_limit_list():
    list1 = ['a12345678901234567890', 'b12345678901234567890', 'c12345678901234567890',
             'd12345678901234567890', 'e12345678901234567890', 'f12345678901234567890',
             'g12345678901234567890', 'h12345678901234567890', 'i12345678901234567890',
             'j12345678901234567890', 'k12345678901234567890', 'l12345678901234567890']
    assert limit_list(list1, 100) == ['a12345678901234567890', 'b12345678901234567890',
                                      'c12345678901234567890', 'd12345678901234567890',
                                      'e12345678901234567890', 'f12345678901234567890',
                                      'g12345678901234567890', 'h12345678901234567890',
                                      'i12345678901234567890', 'j12345678901234567890',
                                      'k12345678901234567890', 'l12345678901234567890']
    assert limit_list(list1, 4) == ['a12345678901234567890', 'b12345678901234567890',
                                    '<9>', 'l12345678901234567890']
    assert limit_list(list1, 4, 50) == ['a12345678901234567890', 'b12345678901234567890',
                                        '<9>', 'l12345678901234567890']
    assert limit_list(list1, 100, 5) == ['a1234...', 'b1234...', 'c1234...', 'd1234...',
                                         'e1234...', 'f1234...', 'g1234...', 'h1234...',
                                         'i1234...', 'j1234...', 'k1234...', 'l1234...']
    assert limit_list(list1, 4, 5) == ['a1234...', 'b1234...', '<9>', 'l1234...']
    print 'TEST limit_list PASSED'


test_limit_list()


# ---------------------------------------------------------------


'''
import re
def is_float_or_nan(string):
    if (string != 'NaN') and (re.match('^\d+?\.\d+?$', string) is None):
        return False
    else:
        return True
'''


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def test_is_int():
    assert '1', is_int('1') is True
    assert '0', is_int('0') is True
    assert '1.0', is_int('1.0') is False
    assert '0.0', is_int('0.0') is False
    assert is_int('1.000000000000000e+00') is False
    assert is_int('0.000000000000000e+00') is False
    print 'TEST is_int() PASSED'


test_is_int()

'''
def ensure_int(string):
    try:
        i = int(string)
        return i
    except ValueError:
        try:
            if not is_int(string):
                f = float(string)
                return int(f)
        except ValueError, err:
            print ValueError, err
            print 'String: \'{0}\''.format(string)
            sys.exit()
'''


def ensure_int(string):
    try:
        i = int(string)
        return i
    except ValueError:
        if not is_int(string):
            f = float(string)
            return int(f)


def test_ensure_int():
    assert ensure_int('1') == 1
    assert ensure_int('0') == 0
    assert ensure_int('1.0') == 1
    assert ensure_int('0.0') == 0
    assert ensure_int('1.000000000000000e+00') == 1
    assert ensure_int('0.000000000000000e+00') == 0
    try:
        ensure_int('0.00blee')
    except ValueError:
        pass

    print 'TEST ensure_int PASSED'


test_ensure_int()


def is_float_or_nan(string):
    if string == 'NaN':
        return True
    try:
        float(string)
        return True
    except ValueError:
        return False


def test_is_float_or_nan():
    assert is_float_or_nan('0.372429') == True
    assert is_float_or_nan('-37.7209') == True
    assert is_float_or_nan('NaN') == True
    assert is_float_or_nan('00001') == True
    assert is_float_or_nan('iteration') == False
    print 'TEST is_float_or_nan PASSED'


test_is_float_or_nan()


# ---------------------------------------------------------------

def validate_iter_float(base_dir, filename,
                        iterations, iter_start=0,
                        value_list_length=1,
                        value_type='float',
                        error_limit=10,
                        list_len_limit=5,
                        list_elm_limit=50,
                        verbosity=1):
    """
    Validates files with <iteration> <value>, where value can be a list, and is either float or int
    :param base_dir:
    :param filename:
    :param iterations: total number of iterations
    :param iter_start: (default 0); start iteration count from this value
    :param value_list_length: (default 1); how many values per iteration
    :param value_type: (default 'float'); type of value to test; 'float' or 'int'
    :param error_limit: (default 10); total number of errors to collect before stopping
    :param list_len_limit: (default 5); the limit on the number of elements of a line to display
    :param list_elm_limit: (default 50); the limit on the string length of elements in line to display
    :param verbosity: (default 1); when >1, prints message to screen
    :return:
    """

    if verbosity > 1:
        print '    validate_iter_float(): val_list_len={0}, vtype={1}, iters={2}, iter_start={3}, {4}' \
            .format(value_list_length, value_type, iterations, iter_start, filename)

    if base_dir[-1] != '/':
        base_dir += '/'

    error_info = list()

    iteration_strings = get_iteration_strings(iterations, iter_start=iter_start)
    iteration_out_of_order = False

    expected_line_length = 1 + value_list_length

    filepath = base_dir + filename

    if not os.path.isfile(filepath):
        error_info.append(('ERROR', 'file not found', filepath))
    else:
        with open(filepath, 'r') as fin:
            i = 0
            for line in fin.readlines():
                line = line.strip('\n').split()
                # print '{0}: {1}'.format(i, line)

                if i == 0:
                    if (line[0] != 'iteration') or (line[1] != 'value'):
                        error_info.append(
                            (i, 'not iteration and value', limit_list(line, list_len_limit, list_elm_limit)))
                else:

                    if len(line) != expected_line_length:
                        error_info.append((i, 'expected {0} values, got {1}'.format(expected_line_length, len(line)),
                                           limit_list(line, list_len_limit, list_elm_limit)))

                    if line[0] not in iteration_strings:
                        error_info.append((i, 'unexpected iteration {0}'.format(line[0]),
                                           limit_list(line, list_len_limit, list_elm_limit)))
                    elif not iteration_out_of_order and line[0] != iteration_strings[0]:
                        error_info.append((i, 'iteration {0} out of order'.format(line[0]),
                                           limit_list(line, list_len_limit, list_elm_limit)))
                        iteration_out_of_order = True
                        iteration_strings.pop(iteration_strings.index(line[0]))
                    else:
                        iteration_strings.pop(iteration_strings.index(line[0]))

                    j = 0
                    for value in line[1:]:
                        if value_type == 'float':
                            if not is_float_or_nan(value):
                                error_info.append(
                                    (i, 'value {0}, "{1}", not float'.format(j, limit_string(value, list_elm_limit)),
                                     limit_list(line, list_len_limit, list_elm_limit)))
                                j += 1
                        elif value_type == 'int':
                            if not is_int(value):
                                error_info.append(
                                    (i, 'value {0}, "{1}", not int'.format(j, limit_string(value, list_elm_limit)),
                                     limit_list(line, list_len_limit, list_elm_limit)))
                                j += 1
                        else:
                            print 'ERROR validate_iter_float(): Unsupported value type: \'{0}\''.format(value_type)
                            sys.exit(-1)
                        if len(error_info) > error_limit:
                            break

                if len(error_info) > error_limit:
                    error_info.append((i, 'exceeded error_limit', error_limit))
                    break
                i += 1

        if len(iteration_strings) > 0:
            error_info.append((None, 'missing iterations {0}'.format(len(iteration_strings)),
                               limit_list(iteration_strings, list_len_limit, list_elm_limit)))

    if len(error_info) == 0:
        return None
    else:
        return error_info


def test_validate_iter_float():
    error_info = validate_iter_float(base_dir='figures',
                                     filename='test_data_raw/validate_iter_float_test_fail.txt',
                                     iterations=20,
                                     iter_start=1,
                                     value_type='float')
    assert error_info == [(0, 'not iteration and value', ['iteraton', 'value']),
                          (1, 'expected 2 values, got 3', ['00001', '0.372429', '023']),
                          (3, 'iteration 00004 out of order', ['00004', 'NaN']),
                          (5, 'value 0, "0.490971f", not float', ['00005', '0.490971f']),
                          (None, 'missing iterations 5', ['00007', '00008', '00009', '00010', '00020'])]

    error_info = validate_iter_float(base_dir='figures',
                                     filename='test_data_raw/validate_iter_float_test_fail.txt',
                                     iterations=20,
                                     iter_start=1,
                                     value_type='float',
                                     error_limit=3)
    assert error_info == [(0, 'not iteration and value', ['iteraton', 'value']),
                          (1, 'expected 2 values, got 3', ['00001', '0.372429', '023']),
                          (3, 'iteration 00004 out of order', ['00004', 'NaN']),
                          (5, 'value 0, "0.490971f", not float', ['00005', '0.490971f']),
                          (5, 'exceeded error_limit', 3),
                          (None, 'missing iterations 6', ['00006', '00007', '00008', '<2>', '00020'])]

    error_info = validate_iter_float(base_dir='figures',
                                     filename='test_data_raw/validate_iter_float_test_succeed.txt',
                                     iterations=1000,
                                     iter_start=1,
                                     value_type='float')
    assert error_info is None
    print 'TEST validate_iter_float PASSED'


test_validate_iter_float()


# ---------------------------------------------------------------

def validate_subdir(base_dir, subdir_name,
                    iterations=0, iter_start=0,
                    filename=None,
                    rows=1,
                    columns=1,
                    value_type='float',
                    allow_float_binary=False,
                    error_limit=10,
                    list_len_limit=5,
                    list_elm_limit=50,
                    verbosity=1):
    if verbosity > 1:
        print '    validate_subdir(): rows={0}, columns={1} vtype={2}, iters={3}, iter_start={4}, {5}, {6}, {7}' \
            .format(rows, columns, value_type, iterations, iter_start, filename, subdir_name, base_dir)

    if base_dir[-1] != '/':
        base_dir += '/'
    if subdir_name[-1] != '/':
        subdir_name += '/'

    subdir_path = base_dir + subdir_name

    error_info = list()
    warn_info = list()
    iteration_strings = list()  # just to be safe...

    if not os.path.isdir(subdir_path):
        error_info.append(('ERROR', 'directory not found', subdir_path))
        return error_info, warn_info

    _, _, files = next(os.walk(subdir_path), (None, None, []))

    if filename:
        if filename in files and len(files) == 1:
            pass
        else:
            warn_info.append(('WARNING', 'expected file \'{0}\''.format(filename),
                              limit_list(files, list_len_limit, list_elm_limit)))
            iteration_strings = get_iteration_strings(iterations, iter_start=iter_start)
    else:
        iteration_strings = get_iteration_strings(iterations, iter_start=iter_start)

    """
    if filename:
        files = [filename]
    else:
        iteration_strings = get_iteration_strings(iterations, iter_start=iter_start)

        # Read dir contents
        _, _, files = next(os.walk(subdir_path), (None, None, []))
    """

    f = 1
    if verbosity > 2:
        print '        ',

    files_with_binary_float = list()

    for fname in files:
        path = subdir_path + fname

        warn_float_binary = False

        if verbosity > 2:
            print '{0}'.format(f),
            if f % 20 == 0:
                print
                print '        ',

        if filename is None:
            # check fname against iteration_strings
            fname_base = fname.split('.')[0]
            if fname_base not in iteration_strings:
                error_info.append((fname, None, 'non-iteration filename', fname_base))
            else:
                iteration_strings.pop(iteration_strings.index(fname_base))

        i = 0
        with open(path, 'r') as fin:
            for line in fin.readlines():
                line = line.strip('/n').split()

                if len(line) > 0:  # don't count blank lines

                    if len(line) != columns:
                        error_info.append((fname, i, 'expected {0} columns, got {1}'.format(columns, len(line)),
                                           limit_list(line, list_len_limit, list_elm_limit)))

                    j = 0
                    for value in line:
                        if value_type == 'float':
                            if not is_float_or_nan(value):
                                error_info.append((fname, i, 'value {0}, "{1}", not float'.format(j, limit_string(value,
                                                                                                                  list_elm_limit)),
                                                   limit_list(line, list_len_limit, list_elm_limit)))
                                # j += 1
                        elif value_type == 'int':
                            if not is_int(value):
                                if allow_float_binary:
                                    try:
                                        f2b_value = ensure_int(value)
                                        if f2b_value == 0 or f2b_value == 1:
                                            if not warn_float_binary:
                                                warn_float_binary = True
                                                files_with_binary_float.append(fname)
                                    except ValueError:
                                        error_info.append((fname, i,
                                                           'value {0}, "{1}", neither int OR binary float'.format(j,
                                                                                                                  limit_string(
                                                                                                                      value,
                                                                                                                      list_elm_limit)),
                                                           limit_list(line, list_len_limit, list_elm_limit)))
                                        # j += 1
                                else:
                                    error_info.append((fname, i, 'value {0}, "{1}", not int'.format(j,
                                                                                                    limit_string(value,
                                                                                                                 list_elm_limit)),
                                                       limit_list(line, list_len_limit, list_elm_limit)))
                                    # j += 1
                        else:
                            print 'ERROR validate_iter_float(): Unsupported value type: \'{0}\''.format(value_type)
                            sys.exit(-1)
                        j += 1
                        if len(error_info) > error_limit:
                            break
                    if len(error_info) > error_limit:
                        break
                    i += 1

        if len(error_info) > error_limit:
            error_info.append((fname, i, 'exceeded error_limit', error_limit))
            break

        if i != rows:
            error_info.append((fname, None, 'expected {0} rows, got {1}'.format(rows, i), i))

        f += 1

    if verbosity > 2:
        print

    if filename is None:
        if len(iteration_strings) > 0:
            error_info.append((None, 'missing {0} files per iteration'.format(len(iteration_strings)),
                               limit_list(iteration_strings, list_len_limit, list_elm_limit)))

    if len(files_with_binary_float) > 0:
        warn_info.append(
            ('WARNING', 'files with binary float', limit_list(files_with_binary_float, list_len_limit, list_elm_limit)))

    return error_info, warn_info


def test_validate_subdir():
    for base_dir, iterations, filename in \
            [('../results/cocktail/h0.5_nocs_cp0/hmm_BFact_w0_fm0.3/F_01', 0, 'W.txt'),
             ('../results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0/LT_01', 1000, None),
             ('../results/cocktail/h0.5_nocs_cp0/hsmm_hdp_w0/LT_01', 1000, None)]:
        error_info, warn_info = validate_subdir(base_dir, 'W',
                                                iterations=iterations,
                                                filename=filename,
                                                rows=8, columns=12,
                                                verbosity=3)

        i = 0
        if error_info:
            for ei in error_info:
                print '    {0} : {1}'.format(i, ei)
                i += 1


# test_validate_subdir()


# ---------------------------------------------------------------

def write_to_log_file(filepath, message, log_dir=None):
    if log_dir:
        owd = os.getcwd()
        os.chdir(log_dir)
    with open(filepath, 'a') as logf:
        logf.write(message)
    if log_dir:
        os.chdir(owd)


# ---------------------------------------------------------------

def validate_results_directory(base_dir, log_file_summary, log_file_detail, log_file_error, log_dir=None,
                               data_root='../', execution_dir=None, dir_num=None, verbosity=1):
    """

    :param base_dir:
    :param log_file_summary:
    :param log_file_detail:
    :param log_file_error:
    :param log_dir:
    :param data_root: Specifies root directory for all references to figures;
    typically this is <hamlet:main> (i.e., '../' if executing from the experiemnt directory)
    :param execution_dir: Keeps track of original directory from which the scrip
    was executed.  Allows for specification of data_root relative to execution_dir
    while permitting the global working directory relative to results root.
    :param verbosity:
    :return:
    """

    if base_dir[-1] != '/':
        base_dir += '/'

    timestamp = util.get_timestamp()

    first_log_message = '({0}, \'{1}\', {2})'.format(dir_num, base_dir, timestamp)
    write_to_log_file(log_file_summary, first_log_message, log_dir)
    write_to_log_file(log_file_detail, '( ' + first_log_message, log_dir)

    global errors_p, error_file_count, warnings_p, warning_file_count
    errors_p = False
    error_file_count = 0
    warnings_p = False
    warning_file_count = 0

    def log_error_info(filename, error_info, warn_info=None):
        global errors_p, error_file_count, warnings_p, warning_file_count
        if error_info:
            errors_p = True
            error_file_count += 1

            log_message = '\n    (' + filename
            for ei in error_info:
                log_message += '\n     {0}'.format(ei)
            log_message += ' )'
            write_to_log_file(log_file_detail, log_message, log_dir)
        if warn_info:
            warnings_p = True
            warning_file_count += 1

            log_message = '\n    (' + filename
            for wi in warn_info:
                log_message += '\n     {0}'.format(wi)
            log_message += ' )'
            write_to_log_file(log_file_detail, log_message, log_dir)

    have_params = False
    try:
        if verbosity > 2:
            print 'CWD:     ', os.getcwd()
            print 'base_dir:', base_dir
        params = util.read_parameter_file_as_dict(base_dir, 'parameters.config')
        have_params = True
    except StandardError:
        errors_p = True
        error_file_count += 1
        log_message = '\n    (parameters.config, \'cannot read\', \'{0}\')' \
            .format(base_dir + 'parameters.config')
        log_message += '\n    (CWD: {0})\n'.format(os.getcwd())
        write_to_log_file(log_file_detail, log_message, log_dir)

    '''
    for key, value in params.iteritems():
        print '\'{0}\' : \'{1}\''.format(key, value)
    sys.exit()
    '''

    if have_params:

        # ---------------------
        # Determine model type from parameters

        model_prior_hdp, model_prior_dir, model_dynamics_hsmm, model_bfact, model_weights_prior_known \
            = False, False, False, False, False
        if params[':MODULE:TRANSITION_PRIOR'] == 'HDP':
            model_prior_hdp = True
        if (params[':MODULE:TRANSITION_PRIOR'] == 'Dirichlet_hyperprior') \
                or (params[':MODULE:TRANSITION_PRIOR'] == 'Binary_factorial'):
            model_prior_dir = True
        if params[':MODULE:DYNAMICS'] == 'HSMM':
            model_dynamics_hsmm = True
        if params[':MODULE:TRANSITION_PRIOR'] == 'Binary_factorial':
            model_bfact = True
        if params[':MODULE:WEIGHTS_PRIOR'] == 'Known':
            model_weights_prior_known = True

        # ---------------------
        # Extract parameters

        # attempt to extract T and output_vec_length from source figures obs.txt
        T = None
        output_vec_length = None
        if ':experiment:data_path' in params:
            experiment_data_path = params[':experiment:data_path']
            if experiment_data_path[-1] != '/':
                experiment_data_path += '/'
            data_obs_file_path = experiment_data_path + 'obs.txt'
            owd = os.getcwd()
            os.chdir(execution_dir)
            os.chdir(data_root)  # this be relative to execution_dir: e.g., '../'
            with open(data_obs_file_path, 'r') as obsf:
                i = 0
                for line in obsf.readlines():
                    if i == 0:
                        output_vec_length = len(line.strip('\n').split())
                    i += 1
                T = i
            os.chdir(owd)

        # D
        D = None
        if 'Binary_state_model:D' in params:
            D = int(params['Binary_state_model:D'])

        # J
        J = None
        if model_prior_hdp:
            if 'HDP_hyperprior:J' in params:
                J = int(params['HDP_hyperprior:J'])
        if model_prior_dir:
            if 'Dirichlet_hyperprior:J' in params:
                J = int(params['Dirichlet_hyperprior:J'])

        # iterations
        experiment_iterations = None
        if ':experiment:iterations' in params:
            experiment_iterations = int(params[':experiment:iterations'])

        print '    T={0}, output_vec_length={1}, D={2}, J={3}'.format(T, output_vec_length, D, J)

        # ---------------------

        files = ['F1_score.txt', 'precision.txt', 'recall.txt', 'accuracy.txt']
        for filename in files:
            error_info = validate_iter_float(base_dir, filename,
                                             experiment_iterations,
                                             iter_start=1,
                                             value_list_length=1,
                                             value_type='float',
                                             verbosity=verbosity)
            log_error_info(filename, error_info)

        files = ['test_log_likelihood.txt', 'train_log_likelihood.txt']
        if model_prior_hdp:
            # I think lambda.txt should only be generated for LT, but it is showing up in HDP noLT...?
            files += ['alpha.txt', 'gamma.txt', 'lambda.txt']

        for filename in files:
            error_info = validate_iter_float(base_dir, filename,
                                             experiment_iterations,
                                             iter_start=0,
                                             value_list_length=1,
                                             value_type='float',
                                             verbosity=verbosity)
            log_error_info(filename, error_info)

        if model_prior_hdp:
            files = ['beta.txt']
            if model_dynamics_hsmm:
                files += ['omega.txt']
            for filename in files:
                error_info = validate_iter_float(base_dir, filename,
                                                 experiment_iterations,
                                                 iter_start=0,
                                                 value_list_length=J,
                                                 value_type='float',
                                                 verbosity=verbosity)
                log_error_info(filename, error_info)

            filename = 'u.txt'
            error_info = validate_iter_float(base_dir, filename,
                                             experiment_iterations,
                                             iter_start=0,
                                             value_list_length=J + 1,
                                             value_type='float',
                                             verbosity=verbosity)
            log_error_info(filename, error_info)

            files = ['n_dot.txt']
            if model_dynamics_hsmm:
                files += ['dtotals.txt']
            for filename in files:
                error_info = validate_iter_float(base_dir, filename,
                                                 experiment_iterations,
                                                 iter_start=0,
                                                 value_list_length=J,
                                                 value_type='int',
                                                 verbosity=verbosity)
                log_error_info(filename, error_info)

        # h.txt
        filename = 'noise_sd.txt'
        error_info = validate_iter_float(base_dir, filename,
                                         experiment_iterations,
                                         iter_start=0,
                                         value_list_length=output_vec_length,
                                         value_type='float',
                                         verbosity=verbosity)
        log_error_info(filename, error_info)

        # W directory
        filename = 'W_directory'
        subdir_filename = None
        if model_weights_prior_known:
            subdir_filename = 'W.txt'
        error_info, warn_info \
            = validate_subdir(base_dir, 'W',
                              iterations=experiment_iterations,
                              iter_start=0,
                              filename=subdir_filename,
                              rows=D + 1,
                              columns=output_vec_length,
                              value_type='float',
                              verbosity=verbosity)
        log_error_info(filename, error_info, warn_info)

        # mu.txt
        filename = 'mu.txt'
        if model_prior_hdp:
            error_info = validate_iter_float(base_dir, filename,
                                             experiment_iterations,
                                             iter_start=0,
                                             value_list_length=D,
                                             value_type='float',
                                             verbosity=verbosity)
            log_error_info(filename, error_info)

        # z.txt
        filename = 'z.txt'
        error_info = validate_iter_float(base_dir, filename,
                                         experiment_iterations,
                                         iter_start=0,
                                         value_list_length=T + 1,
                                         value_type='int',
                                         verbosity=verbosity)
        log_error_info(filename, error_info)

        # theta directory
        filename = 'theta_directory'
        """
        # BROKEN: BFactorial has J=16, but actual output has 128 rows.  What's up?
        if model_bfact:
            error_info = validate_subdir(base_dir, 'theta',
                                         iterations=experiment_iterations,
                                         iter_start=0,
                                         filename='theta.txt',
                                         rows=J,
                                         columns=D,
                                         value_type='float',
                                         verbosity=verbosity)
            log_error_info(filename, error_info)
        """
        if model_prior_hdp:
            error_info, warn_info \
                = validate_subdir(base_dir, 'theta',
                                  iterations=experiment_iterations,
                                  iter_start=0,
                                  filename=None,
                                  rows=J,
                                  columns=D,
                                  value_type='int',
                                  verbosity=verbosity)
            log_error_info(filename, error_info, warn_info)

        # thetastar directory
        filename = 'thetastar_directory'
        error_info, warn_info \
            = validate_subdir(base_dir, 'thetastar',
                              iterations=experiment_iterations,
                              iter_start=0,
                              filename=None,
                              rows=T,
                              columns=D,
                              value_type='int',
                              allow_float_binary=True,
                              verbosity=verbosity)
        log_error_info(filename, error_info, warn_info)

        # A directory
        if model_prior_hdp:
            filename = 'A_directory'
            error_info, warn_info \
                = validate_subdir(base_dir, 'A',
                                  iterations=experiment_iterations,
                                  iter_start=0,
                                  filename=None,
                                  rows=J,
                                  columns=J,
                                  value_type='float',
                                  verbosity=verbosity)
            log_error_info(filename, error_info, warn_info)

        # pi0 directory
        if model_prior_hdp:
            filename = 'pi0_directory'
            error_info, warn_info \
                = validate_subdir(base_dir, 'pi0',
                                  iterations=experiment_iterations,
                                  iter_start=0,
                                  filename=None,
                                  rows=1,
                                  columns=J,
                                  value_type='float',
                                  verbosity=verbosity)
            log_error_info(filename, error_info, warn_info)

    timestamp = util.get_timestamp()
    if not errors_p and not warnings_p:
        log_message = ' (\'success\', {0})'.format(timestamp)
    else:
        if warnings_p:
            log_message = ' ((\'warnings\' {0}), {1})'.format(warning_file_count, timestamp)
        if errors_p:
            log_message = ' ((\'errors\' {0}), {1})'.format(error_file_count, timestamp)
            write_to_log_file(log_file_error, first_log_message + ' ' + log_message + ' )\n', log_dir)
    write_to_log_file(log_file_summary, log_message + '\n', log_dir)
    write_to_log_file(log_file_detail, log_message + ' )\n', log_dir)

    print 'DONE.'


def extract_and_log_experiment_directory_branches(results_root_dir, results_root='../', log_dir=None):
    print 'extract_and_log_experiment_directory_branches() START'
    dir_branches = util.get_directory_branches(root_dir=results_root_dir, main_path=results_root)
    dir_branches = sorted(dir_branches)
    execution_dir = os.getcwd()
    if log_dir is None:
        log_dir = execution_dir
    log_file_branches = 'validate_{0}_branches.log'.format(util.get_timestamp())
    i = 1
    log_message = list()
    log_message.append('{0}'.format(len(dir_branches)))
    for d in dir_branches:
        log_message.append('{0} {1}'.format(i, d))
        i += 1
    log_message = '\n'.join(log_message)
    write_to_log_file(log_file_branches, log_message, log_dir)
    print 'extract_and_log_experiment_directory_branches() DONE'
    return log_file_branches


'''
# laplace
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='../')
'''


'''
# venti - figures
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='../../../figures/')
'''

'''
# venti - results
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='../')
'''


def validate_results(results_root_dir='results/cocktail',
                     results_root='../',
                     data_root='../',
                     log_dir=None,
                     dir_branches=None,
                     verbosity=1):
    execution_dir = os.getcwd()

    if log_dir is None:
        # assign original execution directory as log_dir if none provided
        log_dir = execution_dir

    if dir_branches is None:
        dir_branches = util.get_directory_branches(root_dir=results_root_dir, main_path=results_root)

    dir_branches = sorted(dir_branches)

    timestamp = util.get_timestamp()
    log_file_branches = 'validate_' + timestamp + '_branches.log'
    log_file_summary = 'validate_' + timestamp + '_summary.log'
    log_file_detail = 'validate_' + timestamp + '_detail.log'
    log_file_error = 'validate_' + timestamp + '_error.log'

    log_message = list()
    log_message.append('{0}'.format(len(dir_branches)))
    i = 1
    for d in dir_branches:
        log_message.append('{0} {1}'.format(i, d))
        i += 1
    log_message = '\n'.join(log_message)
    write_to_log_file(log_file_branches, log_message, log_dir)

    if results_root:
        owd = os.getcwd()
        os.chdir(results_root)

    total_dirs = len(dir_branches)
    i = 1
    for results_dir in dir_branches:
        if verbosity > 0:
            print '[{0} : {1}] {2}'.format(i, total_dirs, results_dir)
        validate_results_directory(results_dir, log_file_summary, log_file_detail, log_file_error,
                                   log_dir=log_dir,
                                   data_root=data_root,
                                   execution_dir=execution_dir,
                                   dir_num=i,
                                   verbosity=verbosity)
        i += 1

    if results_root:
        os.chdir(owd)


# ---------------------------------------------------------------
# Validate Scripts
# ---------------------------------------------------------------

# laplace <hamlet>/experiment/ context
'''
validate_results(results_root_dir='results/cocktail/h10.0_nocs_cp1',
                 results_root='../',
                 data_root='../',
                 verbosity=3)
'''

# venti /projects/hamlet/figures/ context
'''
validate_results(results_root_dir='results/cocktail',
                 results_root='../../../figures/',
                 data_root='../',
                 verbosity=3)
'''

# venti /projects/hamlet/src/hdp_hmm_lt/experiment/ context
'''
validate_results(results_root_dir='results/cocktail',
                 results_root='../',
                 data_root='../',
                 verbosity=3)
'''

'''
validate_results(results_root_dir='results/cocktail',
                 results_root='../',
                 data_root='../',
                 dir_branches=['results/cocktail/h1.5_nocs_cp2/hsmm_hdp_w0/noLT_10'
                               #'results/cocktail/h0.5_nocs_cp0/hmm_BFact_w0_fm0.3/F_01',
                               #'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0/LT_01',
                               #'results/cocktail/h0.5_nocs_cp0/hsmm_hdp_w0/LT_01'
                               ],
                 verbosity=3)
'''

'''
dir_branches = util.get_directory_branches(root_dir='../results/cocktail/', main_path=None)
for d in dir_branches:
    print d
'''

'''
owd = None
if main_path:
    owd = os.getcwd()
    os.chdir(main_path)

if main_path:
    os.chdir(owd)
'''


# ---------------------------------------------------------------
# Process post-validation errors
# USE: Re-runs experiments that failed validation
# ---------------------------------------------------------------

def test_experiment_spec(multiproc=True,
                         processor_pool_size=multiprocessing.cpu_count(),
                         main_path='../',
                         test=True):
    spec_list = [run_experiment.ExperimentSpec
                 (parameters_file='binary_factorial_fixed_mean.config',
                  data_dir='figures/cocktail/',
                  data_subdir='h0.5_nocs/cp0',
                  results_subdir='cocktail/h0.5_nocs_cp0/hmm_BFact_w0_fm0.3/F',
                  results_postfix='01',
                  exp_num=1,
                  total_exps=1)]
    run_experiment.run_experiment_script \
        (spec_list,
         main_path=main_path,
         multiproc=multiproc,
         processor_pool_size=processor_pool_size,
         log_file='exp_rerun',
         rerun=False,
         test=test)


# test_experiment_spec()

'''
'./main -p binary_factorial_fixed_mean.config --data_subdir=h0.5_nocs/cp0 --data_dir=figures/cocktail/ -r cocktail/h0.5_nocs_cp0/hmm_BFact_w0_fm0.3/F --results_timestamp=01'
'./main -p binary_factorial_fixed_mean.config --data_subdir=h0.5_nocs/cp0 --data_dir=figures/cocktail/ -r cocktail/h0.5_nocs_cp0/hmm_BFact_w0_fm0.3/F --results_timestamp=01'
'''

# ---------------------------------------------------------------


param_file_map = {('hmm_BFact_w0_fm0.3', 'F'): 'binary_factorial_fixed_mean.config',
                  ('hmm_hdp_w0', 'LT'): 'cocktail_inference_LT_HMM.config',
                  ('hmm_hdp_w0', 'noLT'): 'cocktail_inference_noLT_HMM.config',
                  ('hsmm_hdp_w0', 'LT'): 'cocktail_inference_LT_HSMM.config',
                  ('hsmm_hdp_w0', 'noLT'): 'cocktail_inference_noLT_HSMM.config',
                  ('hmm_hdp_w1', 'LT'): 'cocktail_inference_LT_HMM_W1.config',
                  ('hmm_hdp_w1', 'noLT'): 'cocktail_inference_noLT_HMM_W1.config',
                  ('hsmm_hdp_w1', 'LT'): 'cocktail_inference_LT_HSMM_W1.config',
                  ('hsmm_hdp_w1', 'noLT'): 'cocktail_inference_noLT_HSMM_W1.config'}


def process_validation_errors(validation_errors_log_file,
                              multiproc=True,
                              processor_pool_size=multiprocessing.cpu_count(),
                              main_path='../',
                              test=True):
    spec_list = list()
    bad_branches = list()

    with open(validation_errors_log_file, 'r') as fin:
        for line in fin.readlines():
            branch_list = line.split()[1].strip(',').strip('\'').split('/')[2:-1]

            try:
                key = (branch_list[1], branch_list[2].split('_')[0])
                parameters_file = param_file_map[key]
            except KeyError:
                bad_branches.append(line)
            else:
                data_subdir = branch_list[0]
                data_subdir = data_subdir.split('_')
                data_subdir = data_subdir[0] + '_' + data_subdir[1] + '/' + data_subdir[2]

                results_postfix = branch_list[2].split('_')[1]

                results_subdir = 'cocktail/' + branch_list[0] + '/' + branch_list[1] + '/' + branch_list[2].split('_')[
                    0]

                spec_list.append(run_experiment.ExperimentSpec \
                                     (parameters_file=parameters_file,
                                      data_dir='figures/cocktail/',
                                      data_subdir=data_subdir,
                                      results_subdir=results_subdir,
                                      results_postfix=results_postfix))

                print branch_list, parameters_file, data_subdir, results_subdir, results_postfix

    print '\nBAD BRANCHES:'
    print bad_branches
    print

    total_exps = len(spec_list)
    for i, spec in enumerate(spec_list):
        spec.exp_num = i
        spec.total_exps = total_exps

    run_experiment.run_experiment_script \
        (spec_list,
         main_path=main_path,
         multiproc=multiproc,
         processor_pool_size=processor_pool_size,
         log_file='exp_rerun',
         rerun=False,
         test=test)

    print 'DONE'


'''
process_validation_errors('validate_20151022_215204766407_error.log',
                          test=True)
'''

'''
process_validation_errors('validate_20151025_202819667765_error.log',
                          test=True)
'''


# ---------------------------------------------------------------
# Move branches
# USE:
#    Generally used to move 'BestMatchF1', 'BestMatchHamming'
#    Use move_specified_branches()
# ---------------------------------------------------------------

def move_branches(branches,
                  results_root='../',
                  results_destination=None,
                  log_file=None,
                  test=True):
    if log_file is None:
        log_file = 'validate_move_' + util.get_timestamp() + '.log'

    if test:
        log_message = 'Running in TEST mode'
        print log_message
        write_to_log_file(log_file, log_message + '\n')

    if not os.path.isdir(results_root):
        print 'ERROR: results_root not found: {0}' \
            .format(results_root)
        sys.exit()
    if not os.path.isdir(results_destination):
        print 'ERROR: results_destination not found: {0}' \
            .format(results_destination)
        sys.exit()

    log_message = 'Planning to move {0} branches'.format(len(branches))
    write_to_log_file(log_file, log_message + '\n')
    print log_message

    branches_not_found = list()

    i = 0
    for branch in branches:
        src = os.path.join(results_root, branch)
        dst = os.path.join(results_destination, branch)

        parent_branch = '/'.join(branch.split('/')[:-1])
        src_parent = os.path.join(results_root, parent_branch)

        if not os.path.isdir(src):
            branches_not_found.append(branch)
        else:
            if test:
                log_message = 'would move {0}\n        to {1}'.format(src, dst)
                print log_message
                write_to_log_file(log_file, log_message + '\n')
                i += 1
            else:
                log_message = 'move {0}\n  to {1}'.format(src, dst)
                print log_message
                write_to_log_file(log_file, log_message + '\n')
                if not os.path.exists(dst):
                    os.makedirs(dst)
                shutil.move(src, dst)
                i += 1

                # try removing empty parent directory
                try:
                    os.rmdir(src_parent)
                    print 'removed src_parent: {0}'.format(src_parent)
                except OSError as ex:
                    if ex.errno == errno.ENOTEMPTY:
                        print 'src_parent not yet empty: {0}'.format(src_parent)

    log_message = '\nFINISHED moving {0}\n'.format(i)
    print log_message
    write_to_log_file(log_file, log_message)

    if branches_not_found:
        log_message = '\nBranches not found: {0}'.format(len(branches_not_found))
        print log_message
        write_to_log_file(log_file, log_message + '\n')
        for branch in branches_not_found:
            log_message = '{0}'.format(branch)
            print log_message
            write_to_log_file(log_file, log_message + '\n')

    print '\nDONE.'


def move_bad_branches(validation_errors_log_file,
                      results_root='../',
                      results_destination=None,
                      log_file=None,
                      test=True):
    bad_branches = list()

    if not os.path.isfile(validation_errors_log_file):
        print 'ERROR: validation_errors_log_file not found: {0}' \
            .format(validation_errors_log_file)
        sys.exit()

    with open(validation_errors_log_file, 'r') as fin:
        for line in fin.readlines():
            bad_branches.append(line.split()[1].strip(',').strip('\''))

    move_branches(bad_branches, results_root, results_destination, log_file, test)


'''
# laplace
move_bad_branches('validate_20151118_165618951244_branches.log',
                  results_root='../',
                  results_destination='.',
                  test=True)
'''

# venti:
'''
move_bad_branches('validation_logs/validate_20151022_215204766407_error.log',
                  results_root='/projects/hamlet/figures/',
                  results_destination='/projects/hamlet/figures/cocktail_bad_20151024/',
                  test=True)
'''

# -----


def find_specified_branches(branch_file, or_list, verbose=False):
    branches = list()
    with open(branch_file, 'r') as fin:
        i = 0
        for line in fin.readlines():
            if i > 0:
                line = line.strip('\n').split()[1]
                for item in or_list:
                    if item in line:
                        branches.append(line)
                        break
                print line
            i += 1

    if verbose:
        print
        for branch in branches:
            print branch

    return branches


'''
# laplace
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='../')
'''

'''
# venti
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='/projects/hamlet/figures')
'''

'''
find_specified_branches('validate_20151118_172302116489_branches.log',
                        ['BestMatchF1', 'BestMatchHamming'],
                        verbose=True)
'''


def move_specified_branches(branch_file,
                            or_list,
                            results_root='../',
                            results_destination=None,
                            log_file=None,
                            test=True):

    if not os.path.isfile(branch_file):
        print 'ERROR: file with list of branches not found: {0}' \
            .format(branch_file)
        sys.exit()

    branches_to_move = find_specified_branches(branch_file, or_list)

    move_branches(branches_to_move, results_root, results_destination, log_file, test)

'''
# venti
move_specified_branches('validate_20151118_172302116489_branches.log',
                        ['BestMatchF1'],  # , 'BestMatchHamming'
                        results_root='/projects/hamlet/figures/',
                        results_destination='../results/bad_best_matches',
                        test=True)
'''


# ------
# Use for moving any branches with 'w1' models

def find_w1_branches(validation_branches_file):
    branches_w1 = list()
    with open(validation_branches_file, 'r') as fin:
        i = 0
        for line in fin.readlines():
            if i > 0:
                branch = line.strip('\n').split()[1]
                if 'w1' in branch.split('/')[3].split('_'):
                    branches_w1.append(branch)
            i += 1

    return branches_w1


def move_w1_branches(validation_branches_file,
                     results_root='../',
                     results_destination=None,
                     log_file=None,
                     test=True):
    branches_w1 = find_w1_branches(validation_branches_file)
    move_branches(branches_w1, results_root, results_destination, log_file, test)


# venti
'''
move_w1_branches('validation_logs/validate_20151024_161304265700_branches.log',
                 results_root='/projects/hamlet/figures/',
                 results_destination='/projects/hamlet/figures/cocktail_old_w1_20151025/',
                 test=True)
'''


# ------
# Use for moving '_BFact_' branches models

def find_bfact_branches(validation_branches_file):
    branches_w1 = list()
    with open(validation_branches_file, 'r') as fin:
        i = 0
        for line in fin.readlines():
            if i > 0:
                branch = line.strip('\n').split()[1]

                print branch.split('/')[3].split('_')

                if 'BFact' in branch.split('/')[3].split('_'):
                    branches_w1.append(branch)
            i += 1

    for b in branches_w1:
        print b

    return branches_w1


def move_bfact_branches(validation_branches_file,
                        results_root='../',
                        results_destination=None,
                        log_file=None,
                        test=True):
    branches_w1 = find_bfact_branches(validation_branches_file)
    move_branches(branches_w1, results_root, results_destination, log_file, test)

# find_bfact_branches('validate_20151214_190939879242_branches.log')

'''
# venti - results
extract_and_log_experiment_directory_branches \
    (results_root_dir='results/cocktail', results_root='../')
'''

'''
move_bfact_branches('validate_20151214_190939879242_branches.log',
                    results_root='../',
                    results_destination='../results/BFact',
                    test=True)
'''
