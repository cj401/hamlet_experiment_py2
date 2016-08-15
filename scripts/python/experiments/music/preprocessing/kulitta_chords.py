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
KULITTA_chord1_test1_DATA_ROOT = os.path.join(KULITTA_chord1_DATA_ROOT, 'additional_test_data/seed_100')
KULITTA_chord1_test2_DATA_ROOT = os.path.join(KULITTA_chord1_DATA_ROOT, 'additional_test_data/seed_105')

DATA_SOURCE_CHORD_DICTIONARY_PATHS \
    = ((KULITTA_chord1_DATA_ROOT, 'testgenCS.txt'),
       (KULITTA_chord1_test1_DATA_ROOT, 'testgenCS100.txt'),
       (KULITTA_chord1_test2_DATA_ROOT, 'testgenCS105.txt'))

KULITTA_chord1_tokens_hamlet_DATA_ROOT = os.path.join(KULITTA_CHORD1_ROOT, 'chord1_tokens')

KULITTA_chord1_tokens_test_hamlet_DATA_ROOT = os.path.join(KULITTA_chord1_tokens_hamlet_DATA_ROOT, 'test/chord1_tokens')

# print os.listdir(MUSIC_DATA_ROOT)
# print os.listdir(KULITTA_chord1_DATA_ROOT)


# ------------------------------------------------------------

def read_kulitta_chord1_parse_tree_leaves(data_root, filename='testgen.txt'):
    datapath = os.path.join(data_root, filename)
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

    print '-----'
    for i, datum in enumerate(data):
        print i+1, datum
    print '-----'

    chord_counter = collections.Counter(data)

    print len(chord_counter)
    print chord_counter

# test_read_kulitta_chord1_parse_tree_leaves()


# ------------------------------------------------------------

def read_kulitta_chord1(data_root, filename='testgenCS.txt'):
    """
    Read and parse Kulitta chord1 data.
    Assumes data is in this format:
        (<int,int,int,int>,float),...
    :param data_root: path to data root
    :return: list of tuples: (<chord>, <duration>), where <chord> is tuple of pitch class integers
    """
    datapath = os.path.join(data_root, filename)
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


# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Generate hamlet TEST data from Kullita chord1
# ------------------------------------------------------------

def generate_hamlet_data_from_kulitta_chord1 \
                (data_root,
                 data_source_chord_dictionary_paths,
                 destination_root,
                 data_source_filename='testgenCS.txt',
                 data_chord_roman_source_filename='testgen.txt',
                 destination_filename_prefix='',
                 destination_filename_postfix=''):

    data = read_kulitta_chord1(data_root, data_source_filename)
    data_chord_roman = read_kulitta_chord1_parse_tree_leaves(data_root, data_chord_roman_source_filename)

    chord_dictionary_data = list()
    if data_source_chord_dictionary_paths is not None:
        for root, filename in data_source_chord_dictionary_paths:
            chord_dictionary_data += read_kulitta_chord1(root, filename)

    chord_to_int_dict = get_chord_to_int_dict(chord_dictionary_data)

    if not os.path.exists(destination_root):
        print 'Directory does not exit: {0}'.format(destination_root),
        print '... Creating'
        os.makedirs(destination_root)

    test_obs_filename = '{0}obs{1}.txt'.format(destination_filename_prefix, destination_filename_postfix)
    print 'Writing {0}'.format(test_obs_filename),
    obs_path = os.path.join(destination_root, test_obs_filename)
    with open(obs_path, 'w') as fout:
        for chord, _ in data:
            fout.write('{0}\n'.format(chord_to_int_dict[chord]))
    print 'Done.'

    test_state_to_chord_map_filename = '{0}state_to_chord_map{1}.txt'.format(destination_filename_prefix, destination_filename_postfix)
    print 'Writing {0}'.format(test_state_to_chord_map_filename),
    int_to_chord_dict = dict()
    for chord, i in chord_to_int_dict.iteritems():
        int_to_chord_dict[i] = chord
    state_to_chord_map_path = os.path.join(destination_root,
                                           test_state_to_chord_map_filename)
    with open(state_to_chord_map_path, 'w') as fout:
        for i in sorted(int_to_chord_dict.keys()):
            fout.write('{0} {1}\n'.format(i, int_to_chord_dict[i]))

    chord_roman_to_int_dict = get_chord_roman_to_int_dict()
    print 'Done.'

    test_chord_roman_filename = '{0}chord_roman{1}.txt'.format(destination_filename_prefix, destination_filename_postfix)
    print 'Writing chord_roman.txt',
    chord_roman_path = os.path.join(destination_root, test_chord_roman_filename)
    with open(chord_roman_path, 'w') as fout:
        for (chord_roman, _), _ in data_chord_roman:
            fout.write('{0}\n'.format(chord_roman_to_int_dict[chord_roman]))
    print 'Done.'

    print 'DONE.'


def generate_chord1_script():
    generate_hamlet_data_from_kulitta_chord1 \
        (data_root=KULITTA_chord1_DATA_ROOT,
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_DICTIONARY_PATHS,
         destination_root=KULITTA_chord1_tokens_hamlet_DATA_ROOT)

    generate_hamlet_data_from_kulitta_chord1 \
        (data_root=KULITTA_chord1_test1_DATA_ROOT,
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_DICTIONARY_PATHS,
         destination_root=KULITTA_chord1_tokens_hamlet_DATA_ROOT,
         data_source_filename='testgenCS100.txt',
         data_chord_roman_source_filename='testgen100.txt',
         destination_filename_prefix='test_',
         destination_filename_postfix='')

    generate_hamlet_data_from_kulitta_chord1 \
        (data_root=KULITTA_chord1_test1_DATA_ROOT,
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_DICTIONARY_PATHS,
         destination_root=KULITTA_chord1_tokens_test_hamlet_DATA_ROOT,
         data_source_filename='testgenCS100.txt',
         data_chord_roman_source_filename='testgen100.txt',
         destination_filename_prefix='test_',
         destination_filename_postfix='_seed100')

    generate_hamlet_data_from_kulitta_chord1 \
        (data_root=KULITTA_chord1_test2_DATA_ROOT,
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_DICTIONARY_PATHS,
         destination_root=KULITTA_chord1_tokens_test_hamlet_DATA_ROOT,
         data_source_filename='testgenCS105.txt',
         data_chord_roman_source_filename='testgen105.txt',
         destination_filename_prefix='test_',
         destination_filename_postfix='_seed105')


# ------------------------------------------------------------

'''
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
'''
