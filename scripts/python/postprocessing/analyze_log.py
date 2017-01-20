import re
import datetime
import numpy
import operator
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ----------------------------------------------------------------------

DAY0 = datetime.datetime.strptime('0:00:00.000000', '%H:%M:%S.%f')


def datetime_string_to_seconds(timestring):
    t = datetime.datetime.strptime(timestring, '%H:%M:%S.%f')
    return (t - DAY0).total_seconds()


def test_string_to_datetime():
    timestring = '0:00:07.547171'
    print timestring
    s = datetime.datetime.strptime('0:00:00.000000', '%H:%M:%S.%f')
    t = datetime.datetime.strptime(timestring, '%H:%M:%S.%f')
    print s
    print t
    print t - s
    print (t - s).total_seconds()
    # print time.mktime(t.timetuple())  # nope

# test_string_to_datetime()


# ----------------------------------------------------------------------

MODEL_NAME_RE = (('LT', re.compile('LT.*')),
                 ('noLT', re.compile('noLT.*')),
                 ('BFact', re.compile('BFact.*')),
                 ('Sticky', re.compile('Sticky[^L][^T].*')),
                 ('StickyLT', re.compile('StickyLT.*')))


def generate_model_name_filter(models):

    def _model_name_filter(str):
        for model_name, model_regex in models:
            if model_regex.match(str):
                return model_name
        return 'NULL'

    return _model_name_filter


# ----------------------------------------------------------------------

def get_experiment_timing(log_filepath, filter_fn=None):

    def get_results_dir(str):
        cmd = str.split()
        return cmd[cmd.index('-r') + 1]

    filtered_times = dict()

    with open(log_filepath, 'r') as fin:
        for line in fin.readlines():
            log_line = eval(line)
            if log_line[0] == 'command':
                command_line = log_line[3]
                results_dir = get_results_dir(command_line)
                exp_duration_string = log_line[-1]
                exp_duration = datetime_string_to_seconds(exp_duration_string)
                model_name = filter_fn(results_dir.split('/')[-1])
                if model_name in filtered_times:
                    filtered_times[model_name].append(exp_duration)
                else:
                    filtered_times[model_name] = [exp_duration]

    return filtered_times


def filtered_times_stats(filtered_times):
    for key, val in filtered_times.iteritems():
        n = len(val)
        if n > 1:
            std = numpy.std(val)
        else:
            std = 'nan'
        print key, len(val), numpy.mean(val), std


def quick_stats(x):
    N = len(x)
    mu = sum(x)/N
    diff = [xi - mu for xi in x]
    sqrdiff = [di**2 for di in diff]
    stdev = sum(sqrdiff) / N
    median = numpy.median(x)
    return N, mu, stdev, median


def plot_filtered_time_stats(filtered_times, title=None, show_p=True):
    actualmax = max(reduce(operator.add, filtered_times.values()))
    bins = numpy.arange(0., actualmax)
    colors = cm.rainbow(numpy.linspace(0, 1, len(filtered_times.keys())))

    plt.figure()
    for i, (model, vals) in enumerate(filtered_times.iteritems()):
        n, bins, patches = plt.hist(vals, bins, normed=0, histtype='bar')
        plt.setp(patches, 'facecolor', colors[i], 'alpha', 0.5)

        N, mean, stdev, median = quick_stats(vals)
        print model, N, mean, stdev, median

    plt.xlabel('Seconds')
    plt.ylabel('Frequency')
    if title:
        plt.title(title)

    if show_p:
        plt.show()


def test_get_experiment_timing():
    filename = 'exp_run_20170118_110217930777.log'
    filtered_times = get_experiment_timing(filename, filter_fn=generate_model_name_filter(MODEL_NAME_RE))
    plot_filtered_time_stats(filtered_times, title=filename)

test_get_experiment_timing()


# ----------------------------------------------------------------------
