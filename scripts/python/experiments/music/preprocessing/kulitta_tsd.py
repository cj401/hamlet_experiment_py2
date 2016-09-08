import collections
import os


HAMLET_ROOT = '../../../../../../'
MUSIC_DATA_ROOT = os.path.join(HAMLET_ROOT, 'data/data/music')
KULITTA_TSD_ROOT = os.path.join(MUSIC_DATA_ROOT, 'kulitta_tsd')

KULITTA_TSD_SOURCE_DATA_ROOT = os.path.join(KULITTA_TSD_ROOT, 'source')
KULITTA_tsd_v1_DATA_ROOT = os.path.join(KULITTA_TSD_SOURCE_DATA_ROOT, 'v1_seed300')
KULITTA_tsd_v2_DATA_ROOT = os.path.join(KULITTA_TSD_SOURCE_DATA_ROOT, 'v2_seed400')

DATA_SOURCE_CHORD_CS_DICTIONARY_PATHS \
    = ((KULITTA_tsd_v1_DATA_ROOT, 'testgenCS300.txt'),
       (KULITTA_tsd_v2_DATA_ROOT, 'testgenCS400.txt'))

KULITTA_tsd_v1_CS_hamlet_DATA_ROOT = os.path.join(KULITTA_TSD_ROOT, 'music_tsd_v1_CS')

DATA_SOURCE_CHORD_TRIADS_DICTIONARY_PATHS \
    = ((KULITTA_tsd_v1_DATA_ROOT, 'testgenTriads300.txt'),
       (KULITTA_tsd_v2_DATA_ROOT, 'testgenTriads400.txt'))

KULITTA_tsd_v1_triads_hamlet_DATA_ROOT = os.path.join(KULITTA_TSD_ROOT, 'music_tsd_v1_triads')

# print os.listdir(MUSIC_DATA_ROOT)
# print os.listdir(KULITTA_tsd_DATA_ROOT)


# ------------------------------------------------------------

CHORD_ROMAN_TO_NUM_DICT = { 'I': 1, 'IV': 4, 'V': 5 }


def read_kulitta_tsd_parse_tree_leaves(data_root, filename='testgen300.txt'):
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
                # chord_val = int(chord_tuple[1])
                # dur = int(datum[1].strip())
                # data.append(((chord_roman, chord_val), dur))
                data.append(CHORD_ROMAN_TO_NUM_DICT[chord_roman])
        return data


def read_kulitta_tsd_parse_tree_leaves_batch(data_root, filename_base='testgen300_', s=0, e=49):
    """
    Iterate over the trace files, collecting the parse tree leaves of each and concatenating
    :param data_root:
    :param filename_base:
    :param s:
    :param e:
    :return:
    """
    data = list()
    for i in range(s, e+1):
        filename = filename_base + '{0}.txt'.format(i)
        data += read_kulitta_tsd_parse_tree_leaves(data_root=data_root, filename=filename)
    return data


def test_read_kulitta_tsd_parse_tree_leaves():
    data = read_kulitta_tsd_parse_tree_leaves(data_root=os.path.join(KULITTA_tsd_v1_DATA_ROOT, 'traces'),
                                              filename='testgen300_0.txt')

    print len(data)
    print data

    chord_counter = collections.Counter(data)
    print chord_counter
    print len(chord_counter)

    print '-----------'

    data = read_kulitta_tsd_parse_tree_leaves_batch(data_root=os.path.join(KULITTA_tsd_v1_DATA_ROOT, 'traces'),
                                                    filename_base='testgen300_')

    print len(data)
    #print data

    chord_counter = collections.Counter(data)
    print chord_counter
    print len(chord_counter)


# test_read_kulitta_tsd_parse_tree_leaves()


# ------------------------------------------------------------

def read_kulitta_tsd(data_root, filename='testgenCS300.txt'):
    """
    Read and parse Kulitta tsd data.
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

def get_chord_to_idx_dict(data):
    chord_to_int_dict = dict()
    counter = 0
    for chord, _ in data:
        if chord not in chord_to_int_dict:
            chord_to_int_dict[chord] = counter
            counter += 1
    return chord_to_int_dict


# ------------------------------------------------------------

def generate_hamlet_data_from_kulitta_tsd \
                (data_root,
                 data_source_filename='testgenCS300.txt',
                 data_source_chord_dictionary_paths=None,
                 traces_data_root=None,
                 traces_filename_base=None,
                 destination_root=None,
                 destination_filename_prefix='',
                 destination_filename_postfix=''):

    print '\n========================================================'
    print 'Generating from {0}'.format(data_source_filename)
    print 'Desination: {0}'.format(destination_root)

    # read data
    data = read_kulitta_tsd(data_root, data_source_filename)
    # data_chord_idx = read_kulitta_tsd_parse_tree_leaves(data_root, data_chord_roman_source_filename)
    data_chord_idx = read_kulitta_tsd_parse_tree_leaves_batch(traces_data_root, traces_filename_base)

    # create dictionary of cords across potentially multiple obs sequences
    chord_dictionary_data = list()
    if data_source_chord_dictionary_paths is not None:
        for root, filename in data_source_chord_dictionary_paths:
            chord_dictionary_data += read_kulitta_tsd(root, filename)

    chord_to_idx_dict = get_chord_to_idx_dict(chord_dictionary_data)

    # create destination directory tree if it does not already exist
    if not os.path.exists(destination_root):
        print 'Directory does not exit: {0}'.format(destination_root),
        print '... Creating'
        os.makedirs(destination_root)

    # save observations (mapping chord to chord_idx)
    obs_filename = '{0}obs{1}.txt'.format(destination_filename_prefix, destination_filename_postfix)
    print 'Writing {0}'.format(obs_filename),
    obs_path = os.path.join(destination_root, obs_filename)
    with open(obs_path, 'w') as fout:
        for chord, _ in data:
            fout.write('{0}\n'.format(chord_to_idx_dict[chord]))
    print 'Done.'

    # save emission_to_chord_map
    state_to_chord_map_filename \
        = '{0}emission_to_chord_map{1}.txt'.format(destination_filename_prefix,
                                                   destination_filename_postfix)
    print 'Writing {0}'.format(state_to_chord_map_filename),
    idx_to_chord_dict = dict()
    for chord, i in chord_to_idx_dict.iteritems():
        idx_to_chord_dict[i] = chord
    state_to_chord_map_path = os.path.join(destination_root,
                                           state_to_chord_map_filename)
    with open(state_to_chord_map_path, 'w') as fout:
        for i in sorted(idx_to_chord_dict.keys()):
            fout.write('{0} {1}\n'.format(i, idx_to_chord_dict[i]))
    print 'Done.'

    # save chord_roman (chord_idx roman tokens in integer value form)
    chord_roman_filename \
        = '{0}chord_roman{1}.txt'.format(destination_filename_prefix, destination_filename_postfix)
    print 'Writing {0}'.format(chord_roman_filename),
    chord_roman_path = os.path.join(destination_root, chord_roman_filename)
    with open(chord_roman_path, 'w') as fout:
        for chord_roman_idx in data_chord_idx:
            fout.write('{0}\n'.format(chord_roman_idx))
    print 'Done.'

    print 'DONE.'


def generate_tsd_script():

    # hamlet data for tsd v1 CS
    generate_hamlet_data_from_kulitta_tsd \
        (data_root=KULITTA_tsd_v1_DATA_ROOT,
         data_source_filename='testgenCS300.txt',
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_CS_DICTIONARY_PATHS,
         traces_data_root=os.path.join(KULITTA_tsd_v1_DATA_ROOT, 'traces'),
         traces_filename_base='testgen300_',
         destination_root=KULITTA_tsd_v1_CS_hamlet_DATA_ROOT)

    # hamlet data for tsd v2 CS - as TEST data for v1 CS
    generate_hamlet_data_from_kulitta_tsd \
        (data_root=KULITTA_tsd_v2_DATA_ROOT,
         data_source_filename='testgenCS400.txt',
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_CS_DICTIONARY_PATHS,
         traces_data_root=os.path.join(KULITTA_tsd_v2_DATA_ROOT, 'traces'),
         traces_filename_base='testgen400_',
         destination_root=KULITTA_tsd_v1_CS_hamlet_DATA_ROOT,
         destination_filename_prefix='test_',
         destination_filename_postfix='')

    # hamlet data for tsd v1 Triads
    generate_hamlet_data_from_kulitta_tsd \
        (data_root=KULITTA_tsd_v1_DATA_ROOT,
         data_source_filename='testgenTriads300.txt',
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_TRIADS_DICTIONARY_PATHS,
         traces_data_root=os.path.join(KULITTA_tsd_v1_DATA_ROOT, 'traces'),
         traces_filename_base='testgen300_',
         destination_root=KULITTA_tsd_v1_triads_hamlet_DATA_ROOT)

    # hamlet data for tsd v2 Triads - as TEST data for v1 Triads
    generate_hamlet_data_from_kulitta_tsd \
        (data_root=KULITTA_tsd_v2_DATA_ROOT,
         data_source_filename='testgenTriads400.txt',
         data_source_chord_dictionary_paths=DATA_SOURCE_CHORD_TRIADS_DICTIONARY_PATHS,
         traces_data_root=os.path.join(KULITTA_tsd_v2_DATA_ROOT, 'traces'),
         traces_filename_base='testgen400_',
         destination_root=KULITTA_tsd_v1_triads_hamlet_DATA_ROOT,
         destination_filename_prefix='test_',
         destination_filename_postfix='')

generate_tsd_script()
