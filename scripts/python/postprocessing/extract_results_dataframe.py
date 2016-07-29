__author__ = 'clayton'

import glob

from utilities import util


def get_header_names(header_pattern):
    return [ name for name, _, _, _ in header_pattern ]


def extract_params(results_params, header_pattern):
    row = list()
    for column_name, param, fn_val, fn_include_if in header_pattern:
        proceed = True
        if fn_include_if:
            proceed = fn_include_if(results_params)
        if proceed:
            val = 'nan'
            if param and ( param in results_params ):
                val = results_params[param]
            if fn_val:
                val = fn_val(results_params)
            row.append(val)
    return row


def extract_dataframe(results_subdir_list, header_pattern):

    rows = list()

    rows.append(get_header_names(header_pattern))

    for result_subdir in results_subdir_list:
        results_params = util.read_parameter_file_as_dict(result_subdir)

        rows.append(extract_params(results_params, header_pattern))

    return rows


def collect_results_subdirs(results_dirs):

    results_subdir_list = list()
    for results_dir in results_dirs:
        results_subdir_list += glob.glob(results_dir + '*')

    return results_subdir_list


def print_dataframe(dataframe):
    for row in dataframe:
        row_strings = [ '{0}'.format(elm) for elm in row ]
        print ' '.join(row_strings)


def save_dataframe(dataframe, filename):
    print 'Writing dataframe to file: \'{0}\''.format(filename)
    with open(filename, 'w') as of:
        for row in dataframe:
            row_strings = [ '{0}'.format(elm) for elm in row ]
            of.write(' '.join(row_strings) + '\n')
    print 'DONE.'


# script

def lambda_learned_p(results_params):
    if 'Binary_state_model:lambda' in results_params:
        return 1
    return 0

def hdp_only_p(results_params):
    if ':MODULE:TRANSITION_PRIOR' in results_params:
        return results_params[':MODULE:TRANSITION_PRIOR'] == 'HDP'
    return False

header_pattern_1 = (
    # <column_name>, <parameter_name>, <fn-value>, <fn-include-if>
    ('data_dir', ':experiment:data_path', None, None),
    ('results_dir', ':experiment:results_path', None, None),
    ('dynamics', ':MODULE:DYNAMICS', None, None),
    ('transition_prior', ':MODULE:TRANSITION_PRIOR', None, hdp_only_p),
    # True when 'Binary_state_model:lambda' is specified
    ('lambda_learned', None, lambda_learned_p, None),
    ('b_lambda', 'Binary_state_model:b_lambda', None, None),
    ('lambda', 'Binary_state_model:lambda', None, None)
    )

results_dirs_1 = [ '../results/new-results/' ]

results_subdir_list = collect_results_subdirs(results_dirs_1)

df = extract_dataframe(results_subdir_list, header_pattern_1)
print_dataframe(df)

# print
save_dataframe(df, 'dataframe_test.txt')


'''
what I want to be able to do is read in a figures frame to R where rows are
figures sets and columns are pertinent pieces of information about the figures
set (results subdirectory, corresponding figures directory, lambda value/prior),
and use it to locate the relevant output to visualize

it would be great to be able to create (manually) a "query", and generate
such a figures frame-like file where whatever experiments we want to be able
to plot on top of each other appear in the frame

I think the most important things to include are
(1) the name of the subdirectory in results/ where the results of an experiment can be found,
(2) the name of the corresponding subdirectory in figures/ where any potential ground
    truth information can be found,
(3) a flag indicating whether lambda was fixed or learned
    (which is determined by whether the lambda parameter was specified in the param file),
(4) and what the value of either lambda or b_lambda was
(5) the value of the transition module (HDP / Dirichlet) and
(6) the dynamics module (HMM / HSMM) so we can contrast on those as well
(7) figures in *.gt_eval
(8) test_log_likelihood


I was thinking that we'd provide a list of subdirectory names
(which might be constructed by a reg exp, probably using glob).
Then we could provide a list of params to gather from the .config files.
There could be two sets of these determining two collection conditions:
    one where the params are not necessarily present in all files, so this is an OR,
    then one where the param must be present, perhaps with a specified value
    -- these need to be present / satisfy the condition.
the run_experiment_script then runs over all subdirs in the path list and checks the conditions
and gathers the params with their values

'''
