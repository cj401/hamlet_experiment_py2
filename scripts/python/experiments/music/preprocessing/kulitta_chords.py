import os
import collections


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
MUSIC_DATA_ROOT = os.path.join(HAMLET_ROOT, 'data/data/music')
KULITTA_DATA_ROOT = os.path.join(MUSIC_DATA_ROOT, 'kulitta')
KULITTA_chord1_DATA_ROOT = os.path.join(KULITTA_DATA_ROOT, 'KulittaData_chord1_20160710')

# print os.listdir(KULITTA_chord1_DATA_ROOT)


# ------------------------------------------------------------

def read_kulitta_chord1_parse_tree_leaves(data_root):
    datapath = os.path.join(data_root, 'testgen.txt')
    with open(datapath, 'r') as fin:
        input_string_list = fin.readlines()
        last_row = input_string_list[-1]
        intuple = False
        data = list()
        # print last_row
        for i, c in enumerate(last_row):
            if c == '(':
                intuple = True
                s = i + 1
            elif c == ')' and intuple is True:
                intuple = False
                datum = last_row[s:i].split('%')
                chord_tuple = datum[0].strip().split(',')
                chord_roman = chord_tuple[0]
                chord_val = int(chord_tuple[1])
                dur = int(datum[1].strip())
                data.append(((chord_roman, chord_val), dur))
        return data


def test_read_kulitta_chord1_parse_tree_leaves():
    data = read_kulitta_chord1_parse_tree_leaves(KULITTA_chord1_DATA_ROOT)
    print data
    data = [chord_roman for (chord_roman, _), _ in data ]
    print data

    chord_counter = collections.Counter(data)

    print len(chord_counter)
    print chord_counter

test_read_kulitta_chord1_parse_tree_leaves()


# ------------------------------------------------------------

def read_kulitta_chord1(data_root):
    """
    Read and parse Kulitta chord1 data.
    Assumes data is in this format:
        (<int,int,int,int>,float),...
    :param data_root: path to data root
    :return: list of tuples: (<chord>, <duration>), where <chord> is tuple of pitch class integers
    """
    datapath = os.path.join(data_root, 'testgenCS.txt')
    with open(datapath, 'r') as fin:
        input_string = fin.readlines()
        intuple = False
        data = list()
        for i, c in enumerate(input_string[0]):
            if c == '(':
                intuple = True
                s = i + 1
            elif c == ')' and intuple is True:
                intuple = False
                datum = input_string[0][s+1:i].split('>')
                chord = tuple([int(val) for val in datum[0].split(',')])
                dur = float(datum[1].strip(','))
                data.append((chord, dur))
        return data


def chord2interval(chord):
    return tuple([n2 - n1 for n1, n2 in zip(chord, chord[1:])])


def chord2intervals(data):
    return [chord2interval(chord) for chord in data]


def kulitta_chord1_simple_stats(data):

    print 'total tokens'
    print len(data)

    data = [datum[0] for datum in data]
    chord_counter = collections.Counter(data)

    print '\nUnique Chords:'
    print len(chord_counter)
    print chord_counter

    intervals = chord2intervals(data)
    interval_counter = collections.Counter(intervals)

    print '\nIntervals:'
    print intervals
    print len(interval_counter)
    print interval_counter

    roots = [chord[0] for chord in data]
    roots_counter = collections.Counter(roots)

    print '\nRoots:'
    print len(roots_counter)
    print roots_counter

    root_intervals = [ r2 - r1 for r1, r2 in zip(roots, roots[1:]) ]
    root_intervals_counter = collections.Counter(root_intervals)

    print '\nsequential root intervals'
    print len (root_intervals)
    print root_intervals_counter


def script():
    data = read_kulitta_chord1(KULITTA_chord1_DATA_ROOT)
    kulitta_chord1_simple_stats(data)


# script()


