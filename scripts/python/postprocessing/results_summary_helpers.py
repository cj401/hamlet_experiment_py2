__author__ = 'clayton'

import collections
import os
import sys

import prettytable

from utilities import util


def append_data_to_dict(d, data_set_name, data_set_LT, data_set_LT_num, value):
    # print d
    if data_set_name not in d:
        d[data_set_name] = collections.OrderedDict()
        d[data_set_name][data_set_LT] = collections.OrderedDict()
        d[data_set_name][data_set_LT][data_set_LT_num] = value
    else:
        if data_set_LT not in d[data_set_name]:
            d[data_set_name][data_set_LT] = collections.OrderedDict()
            d[data_set_name][data_set_LT][data_set_LT_num] = value
        else:
            d[data_set_name][data_set_LT][data_set_LT_num] = value


def print_dict(d):
    for data_set_name, lt_dict in d.iteritems():
        for data_set_LT, num_dict in lt_dict.iteritems():
            print '{0} {1}:'.format(data_set_name, data_set_LT)
            for n, val in num_dict.iteritems():
                print '    {0}: {1}'.format(n, val)


def dict_to_prettytable(d, decimal_precision=3):
    dfmtr = '{{0:.{0}f}}'.format(decimal_precision)
    for data_set_name, lt_dict in d.iteritems():
        for data_set_LT, lt_num_dict in lt_dict.iteritems():
            header = [ 'Iteration' ] \
                     + [ '{0}'.format(key) for key in lt_num_dict.keys() ]
            table = prettytable.PrettyTable(header)

            iteration_keys = util.OrderedSet()

            for value_dict in lt_num_dict.itervalues():
                for iter_key in value_dict.iterkeys():
                    iteration_keys.add(iter_key)

            for iter_key in iteration_keys:
                row = [ iter_key ]
                for value_dict in lt_num_dict.itervalues():
                    if iter_key in value_dict:
                        value = value_dict[iter_key]
                        try:
                            value = float(value)
                            value = dfmtr.format(value)
                        except ValueError:
                            pass
                    else:
                        value = 'Null'
                    row.append(value)
                # print row
                table.add_row(row)

            print '\n{0} {1}:'.format(data_set_name, data_set_LT)
            print table


def display_experiment_summary_tables(data_root, results_file, depth=2):

    owd = os.getcwd()
    os.chdir(data_root)
    all_data = collections.OrderedDict()

    for dirName, subdirList, fileList in os.walk('.'):
        dircomps = dirName.split('/')
        if len(dircomps) == depth+1:
            # print 'dirName:', dirName, 'subdirList:', subdirList, 'fileList:', fileList
            data_set_name = '/'.join(dircomps[0:-1])
            data_set_LT = dircomps[-1].split('_')

            # print data_set_name, data_set_LT

            if results_file in fileList:
                data = collections.OrderedDict()
                with open(dirName + '/' + results_file, 'r') as fin:
                    for line in fin.readlines():
                        comps = [ x.strip() for x in line.split(' ') ]
                        if comps[0] != 'iteration':
                            data[comps[0]] = comps[1]

                append_data_to_dict(all_data, data_set_name, data_set_LT[0], data_set_LT[1], data)

    # print_dict(all_data)

    dict_to_prettytable(all_data)

    os.chdir(owd)


'''
collect_files('../results/cocktail_no_learning/h10.0_cp0/',
              'F1_score.txt',
              depth=2)
'''

'''
collect_files('../results/cocktail/a1b1_nocs_cp0/',
              'F1_score.txt',
              depth=2)
'''

if __name__ == '__main__':

    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print 'PRELIMINARY'
        print 'usage: python collect_files.py <data_root> <results_file>'
        print 'walks figures under data_root and collects and summarizes results_files found'
        sys.exit(1)

    data_root = '../results/cocktail_no_learning/h10.0_cp0/'
    results_file = 'F1_score.txt'

    if len(sys.argv) > 1:
        data_root = sys.argv[1]

    if len(sys.argv) > 2:
        results_file = sys.argv[2]

    display_experiment_summary_tables(data_root, results_file, depth=2)
