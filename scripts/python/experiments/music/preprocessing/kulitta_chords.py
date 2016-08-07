import os
import collections


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
MUSIC_DATA_ROOT = os.path.join(HAMLET_ROOT, 'data/data/music')
KULITTA_CHORD1_ROOT = os.path.join(MUSIC_DATA_ROOT, 'kulitta_chord1')

KULITTA_CHORD1_SOURCE_DATA_ROOT = os.path.join(KULITTA_CHORD1_ROOT, 'source')
KULITTA_chord1_DATA_ROOT = os.path.join(KULITTA_CHORD1_SOURCE_DATA_ROOT, 'KulittaData_chord1_20160710')

KULITTA_chord1_tokens_hamlet_DATA_ROOT = os.path.join(KULITTA_CHORD1_ROOT, 'chord1_tokens')

# print os.listdir(MUSIC_DATA_ROOT)
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

# test_read_kulitta_chord1_parse_tree_leaves()


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


# ------------------------------------------------------------
# Generate hamlet data from Kullita chord1
# ------------------------------------------------------------

def get_chord_to_int_dict(data):
    chord_to_int_dict = dict()
    counter = 0
    for chord, _ in data:
        if chord not in chord_to_int_dict:
            chord_to_int_dict[chord] = counter
            counter += 1
    return chord_to_int_dict


def get_chord_roman_to_int_dict():
    return { 'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7 }


def generate_hamlet_data_from_kulitta_chord1(data_root, destination_root):
    """
    Read Kulitta chord1 data and latent chord Roman numeral latent state
    and save in hamlet format two files: obs.txt, chord_roman.txt
    :param data_root:
    :param destination_root:
    :return:
    """
    data = read_kulitta_chord1(data_root)
    data_chord_roman = read_kulitta_chord1_parse_tree_leaves(data_root)

    if not os.path.exists(destination_root):
        print 'Directory does not exit: {0}'.format(destination_root),
        print '... Creating'
        os.makedirs(destination_root)

    chord_to_int_dict = get_chord_to_int_dict(data)

    print 'Writing obs.txt',
    obs_path = os.path.join(destination_root, 'obs.txt')
    with open(obs_path, 'w') as fout:
        for chord, _ in data:
            fout.write('{0}\n'.format(chord_to_int_dict[chord]))
    print 'Done.'

    print 'Writing state_to_chord_map.txt',
    int_to_chord_dict = dict()
    for chord, i in chord_to_int_dict.iteritems():
        int_to_chord_dict[i] = chord
    state_to_chord_map_path = os.path.join(destination_root,
                                           'state_to_chord_map.txt')
    with open(state_to_chord_map_path, 'w') as fout:
        for i in sorted(int_to_chord_dict.keys()):
            fout.write('{0} {1}\n'.format(i, int_to_chord_dict[i]))

    chord_roman_to_int_dict = get_chord_roman_to_int_dict()
    print 'Done.'

    print 'Writing chord_roman.txt',
    chord_roman_path = os.path.join(destination_root, 'chord_roman.txt')
    with open(chord_roman_path, 'w') as fout:
        for (chord_roman, _), _ in data_chord_roman:
            fout.write('{0}\n'.format(chord_roman_to_int_dict[chord_roman]))
    print 'Done.'

    print 'DONE.'

generate_hamlet_data_from_kulitta_chord1(KULITTA_chord1_DATA_ROOT,
                                         KULITTA_chord1_tokens_hamlet_DATA_ROOT)
