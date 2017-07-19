import os
import sys
import shutil
import random
import xlrd
import music21

import numpy


"""
20170120
Adapted from music/data/bach_analysis/extract_chorale_names.py
"""


"""
Command used by DeepBach:
music21.corpus.getBachChorales(fileExtensions='xml')
"""


# ----------------------------------------------------------------------
# Ensure <hamlet>/experiment/scripts/python/ in sys.path (if possible)
# ----------------------------------------------------------------------

def seq_equal(s1, s2):
    if len(s1) != len(s2):
        return False
    for e1, e2 in zip(s1, s2):
        if e1 != e2:
            return False
    return True


def print_sys_path():
    for i, path in enumerate(sys.path):
        print i, path


def find_path_context(target_path_components):
    for path in sys.path:
        path_components = path.split('/')
        if 'hamlet' in path_components:
            if seq_equal(target_path_components,
                         path_components[-len(target_path_components):]):
                return True
    return False


def optional_add_relative_path(current, parent, relative_path, verbose=False):
    """
    If executing in current directory and parent path is not in sys.path,
    then add parent path.
    :return:
    """
    if not find_path_context(parent):
        if find_path_context(current):
            parent_path = os.path.realpath(os.path.join(os.getcwd(), relative_path))
            if verbose:
                print 'NOTICE: experiment_tools.py'
                print '    Executing from:     {0}'.format(os.getcwd())
                print '    Adding to sys.path: {0}'.format(parent_path)
            sys.path.insert(1, parent_path)


optional_add_relative_path\
    (current=('scripts', 'python', 'experiments', 'cocktail_party', 'experiment'),
     parent=('scripts', 'python'),
     relative_path='../../../',
     verbose=True)


from run import experiment_tools


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
DATA_ROOT = experiment_tools.DATA_ROOT
PARAMETERS_ROOT = experiment_tools.PARAMETERS_ROOT
RESULTS_ROOT = experiment_tools.RESULTS_ROOT

DATA_MUSIC_ROOT = os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT), 'music')
BACH_CHORALE_NOMINAL_ROOT = os.path.join(DATA_MUSIC_ROOT, 'bach_chorale_nominal')
BACH_CHORALE_NOMINAL_SOURCE_ROOT = os.path.join(BACH_CHORALE_NOMINAL_ROOT, 'source')

# print os.listdir(BACH_CHORALE_NOMINAL_ROOT)


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def read_chorales_from_xls(filename):
    book = xlrd.open_workbook(filename)
    sh = book.sheet_by_index(0)
    chorale_names = set()
    for row in range(1, sh.nrows):  # range(sh.nrows):
        chorale_names.add(sh.cell_value(row, 7))

        # for col in range(sh.ncols):
        #     print col, sh.cell_value(row, col)

    return chorale_names

# names = read_chorales_from_xls('BachChoraleXmlListOfChords_tab.xls')
# print names
# print len(names)


# ----------------------------------------------------------------------

def find_music21_chorale_matches(chorale_xls_filename):

    chorale_names_raw = read_chorales_from_xls(chorale_xls_filename)

    # note: *should* handle potential exception if encoding doesn't work...
    chorale_names = map(lambda cnu: cnu.encode('utf8'), chorale_names_raw)

    music21_bc_path_list = music21.corpus.getBachChorales()
    print 'music21 chorales: {0}'.format(len(music21_bc_path_list))

    # music21_bc_names = map(lambda fname: os.path.basename(fname)[:-4], music21_bc_path_list)

    music21_bc_name_to_path_dict = dict()
    for path in music21_bc_path_list:
        name = os.path.basename(path)[:-4]
        music21_bc_name_to_path_dict[name] = path

    match_dict = dict()
    miss_list = list()

    for cname in chorale_names:
        if cname in music21_bc_name_to_path_dict:
            match_dict[cname] = music21_bc_name_to_path_dict[cname]
        else:
            miss_list.append(cname)

    print 'matches:', len(match_dict.keys())
    print 'misses:', len(miss_list)
    print miss_list

    for i, (cname, path) in enumerate(match_dict.iteritems()):
        print i, cname, path

# find_music21_chorale_matches('BachChoraleXmlListOfChords_tab.xls')


# ----------------------------------------------------------------------

def chordify_4voice_chorale(stream, chord_to_symbol_map=None, sym_idx=None,
                            chord_count_dict=None,
                            symbol_to_chord_map=None,
                            transpose_p=True,
                            verbose=False):

    if chord_to_symbol_map is None: chord_to_symbol_map = dict()
    if sym_idx is None: sym_idx = 0
    if chord_count_dict is None: chord_count_dict = dict()
    if symbol_to_chord_map is None: symbol_to_chord_map = dict()

    keys = dict()

    chord_symbol_list = list()

    num_voices = 0
    for elm in list(stream):
        if type(elm) is music21.stream.Part:
            if elm.id not in ('Soprano', 'Alto', 'Tenor', 'Bass',
                              'S.', 'A.', 'T.', 'B.'):
                stream.remove(elm)
            else:
                num_voices += 1
                for pelm in elm.flat:
                    # print pelm, type(pelm)
                    if type(pelm) is music21.key.Key:
                        if elm.id in keys:
                            keys[elm.id].append(pelm)
                        else:
                            keys[elm.id] = [ pelm ]

    # stream.show('text')

    more_than_1_keys = False
    different_keys = False

    first_key = None
    for i, (voice, key_list) in enumerate(keys.iteritems()):

        if verbose: print 'voice', voice, ', key_list', key_list

        # test for any cases of voices being in different keys
        if len(key_list) > 1: more_than_1_keys = True

        # test for any cases of key change in chorale
        if i == 0 and key_list is not None:
            first_key = key_list[0]
        else:
            if key_list[0] != first_key:
                different_keys = True

    key_mode = None
    valid_score = False
    if num_voices == 4 and different_keys is False and more_than_1_keys is False:

        # Since has 4 voices and no different keys and no more than 1 key, is valid
        valid_score = True

        # print 'keys', keys
        # print 'keys.keys()', keys.keys()
        chorale_key = keys[keys.keys()[0]][0]
        # chorale_tonic = chorale_key.getTonic()
        # chorale_tonic_midi = chorale_tonic.midi
        key_mode = chorale_key.mode
        key_pitchClass = chorale_key.tonic.pitchClass

        if verbose:
            print 'chorale_key', chorale_key
            # print 'chorale_tonic', chorale_tonic
            # print 'chorale_tonic_midi', chorale_tonic_midi
            print 'key_mode', key_mode
            print 'key_pitchClass', key_pitchClass

        chords = stream.chordify()

        # chords.show('text')
        # chords.show()
        # print '----'

        for elm in chords.flat:
            if type(elm) is music21.chord.Chord:

                midi_pitch_list = list()
                for p in list(elm.pitches):
                    if transpose_p:
                        # transposes to C
                        p.midi = p.midi - key_pitchClass
                    midi_pitch_list.append(p.midi)

                # midi_pitch_tuple = tuple(sorted(midi_pitch_list))
                midi_pitch_tuple = tuple(midi_pitch_list)

                if midi_pitch_tuple in chord_to_symbol_map:
                    chord_symbol_list.append(chord_to_symbol_map[midi_pitch_tuple])
                    chord_count_dict[midi_pitch_tuple] += 1
                else:
                    chord_to_symbol_map[midi_pitch_tuple] = sym_idx
                    symbol_to_chord_map[sym_idx] = midi_pitch_tuple
                    sym_idx += 1
                    chord_symbol_list.append(chord_to_symbol_map[midi_pitch_tuple])
                    chord_count_dict[midi_pitch_tuple] = 1

        # print '------'
        # chords.show('text')
        # chords.show()

    else:
        # if different_keys is True or more_than_1_keys is True:
        print 'Number of voices', num_voices
        print 'Key change', different_keys
        print 'More than one key', more_than_1_keys
        for voice, key_list in keys.iteritems():
            print '    ', voice, len(key_list), key_list

    return chord_symbol_list, chord_to_symbol_map, sym_idx, \
           chord_count_dict, symbol_to_chord_map, key_mode, valid_score


def test_chordify_4voice_chorale():
    music21_bc_path_list = music21.corpus.getBachChorales()
    chorale_path = music21_bc_path_list[1]
    s = music21.converter.parse(chorale_path)
    print 'chorale_path', chorale_path
    # s.show('text')
    chord_symbol_list, chord_to_symbol_map, sym_idx, \
        chord_count_dict, symbol_to_chord_map, key_mode, valid_score \
        = chordify_4voice_chorale(s, transpose_p=False, verbose=True)
    print 'Valide Score', valid_score
    print 'chorale_length', len(chord_symbol_list)
    print 'chord_symbol_list', chord_symbol_list
    print 'chord_to_symbol_map', chord_to_symbol_map
    print 'sym_idx', sym_idx
    print 'chord_count_dict', chord_count_dict
    print 'symbol_to_chord_map', symbol_to_chord_map
    print 'key_mode', key_mode

# test_chordify_4voice_chorale()


# ----------------------------------------------------------------------

def chordify_and_symbolize_batch(transpose_p=True, limit=None, verbose=False):
    """

    processed_score_dict : <name> -> <list-of-ints>
    score_by_mode : <mode> -> <list-of-names>
    chord_to_symbol_map : <chord> -> <int>
    symbol_to_chord_map : <int> -> <chord>
    chord_count_dict : <chord> -> <int: frequency of chord in corpus>

    :param transpose_p:
    :param limit:
    :param verbose:
    :return:
    """
    music21_bc_path_list = music21.corpus.getBachChorales()
    chord_to_symbol_map = dict()
    sym_idx = 0
    chord_count_dict = dict()
    symbol_to_chord_map = dict()
    processed_score_dict = dict()
    score_by_mode = dict()  # key=mode, val=list-of-score_names

    for i, bc_path in enumerate(music21_bc_path_list):  # [:100]
        name = os.path.splitext(os.path.basename(bc_path))[0]

        if verbose: print '-----', i, name,

        s = music21.converter.parse(bc_path)
        chord_symbol_list, chord_to_symbol_map, sym_idx, chord_count_dict, \
            symbol_to_chord_map, key_mode, valid_score \
            = chordify_4voice_chorale(s, chord_to_symbol_map, sym_idx,
                                      chord_count_dict, symbol_to_chord_map,
                                      transpose_p=transpose_p)

        if valid_score:
            processed_score_dict[name] = chord_symbol_list
            if key_mode in score_by_mode:
                score_by_mode[key_mode].append(name)
            else:
                score_by_mode[key_mode] = [name]

        if verbose: print len(chord_symbol_list), key_mode

        if limit and i >= limit:
            break

    # for name, csl in processed_score_dict.iteritems():
    #     print name, len(csl), csl

    return processed_score_dict, score_by_mode, chord_to_symbol_map, \
           chord_count_dict, symbol_to_chord_map


def test_chordify_and_symbolize_batch(transpose_p=True):
    processed_score_dict, score_by_mode, chord_to_symbol_map, \
        chord_count_dict, symbol_to_chord_map \
        = chordify_and_symbolize_batch(transpose_p=transpose_p, verbose=True)

    if transpose_p:
        print 'Transpose ON'
    else:
        print 'Transpose OFF'

    print 'Total number of unique chords', len(chord_to_symbol_map.keys())
    print 'Modes:'
    for mode, scores in score_by_mode.iteritems():
        print '    ', mode, len(scores)

# test_chordify_and_symbolize_batch(transpose_p=True)


"""
Transpose ON
Total number of unique chords 5895
----- 0 bwv1.6 [60, 55, 52, 48]

Transpose OFF:
Reason for fewer unique chords: sharing of chords across keys:
    e.g., C-G-C-E is a I chord in GM, a IV in FM
    But transposing these all to C then makes C-G-C-E become
    two different chords
Total number of unique chords 5659
----- 0 bwv1.6 [65, 60, 57, 53]

Total number of scores 398
Modes:
     major 217
     minor 181

Very long chorale (882 chords):
    226 bwv328 882 minor
"""


# ----------------------------------------------------------------------

def batch_process_and_save_bach_chorales(dest_dir, limit=None, transpose_p=True):
    """
    Chordify and symbolize Bach chorales and save in dest_dir

    processed_score_dict : <name> -> <list-of-ints>
    score_by_mode_dict : <mode> -> <list-of-names>
    chord_to_symbol_map : <chord> -> <int>
    symbol_to_chord_map : <int> -> <chord>
    chord_count_dict : <chord> -> <int: frequency of chord in corpus>

    :param dest_dir:
    :param limit: (default None) when integer, represents limit to number of scores to process
    :param transpose_p: (default True) Flag for whether to transpose scores to C
    :return:
    """

    symbol_sources_dest_path = os.path.join(dest_dir, 'sources')

    if not os.path.exists(symbol_sources_dest_path):
        print '[NOTE] batch_process_and_save_bach_chorales(): dir does not exit, creating:'
        print '       \'{0}\''.format(symbol_sources_dest_path)
        os.makedirs(symbol_sources_dest_path)

    processed_score_dict, score_by_mode_dict, chord_to_symbol_map, \
        chord_count_dict, symbol_to_chord_map \
        = chordify_and_symbolize_batch(transpose_p=transpose_p, limit=limit, verbose=True)

    # save score_mode.txt
    score_mode_filepath = os.path.join(dest_dir, 'score_mode.txt')
    print 'saving {0}'.format(score_mode_filepath)
    with open(score_mode_filepath, 'w') as fout:
        for mode, names in score_by_mode_dict.iteritems():
            for name in names:
                fout.write('{0} {1}\n'.format(name, mode))

    # save symbol_to_chord_map.txt
    symbol_to_chord_map_filepath = os.path.join(dest_dir, 'symbol_to_chord_map.txt')
    print 'saving {0}'.format(symbol_to_chord_map_filepath)
    with open(symbol_to_chord_map_filepath, 'w') as fout:
        for symbol, chord in symbol_to_chord_map.iteritems():
            fout.write('{0} {1}\n'.format(symbol, chord))

    # save chord_corpus_frequency.txt
    chord_corpus_frequency_filepath = os.path.join(dest_dir, 'chord_corpus_frequency.txt')
    print 'saving {0}'.format(chord_corpus_frequency_filepath)
    with open(chord_corpus_frequency_filepath, 'w') as fout:
        for chord, frequency in chord_count_dict.iteritems():
            fout.write('{0} {1}\n'.format(chord, frequency))

    # save symbolized scores
    print 'Saving symboliced scores to {0}'.format(symbol_sources_dest_path)
    for name, symbol_list in processed_score_dict.iteritems():
        filepath = os.path.join(symbol_sources_dest_path, '{0}.txt'.format(name))
        with open(filepath, 'w') as fout:
            for symbol in symbol_list:
                fout.write('{0}\n'.format(symbol))

    print 'DONE.'

# batch_process_and_save_bach_chorales(BACH_CHORALE_NOMINAL_SOURCE_ROOT, limit=None)


# ----------------------------------------------------------------------

def partition_shuffled_set(names, train_size):
    """
    Randomly shuffle names and partition into train set of size train_size
    with the remainder returned as test set
    :param names: list of chorale names
    :param train_size: integer for train_set size ; rest will be test
    :return: train_names and test_names
    """
    names = list(names)
    if len(names) >= train_size:
        random.shuffle(names)
        return names[:train_size], names[train_size:]
    else:
        print 'ERROR: partition_training_set(): train_size {0} > len(names) {1}'\
            .format(train_size, len(names))
        sys.exit()


def test_partition_training_set():
    names = list(range(10))
    print 'names', names
    train, test = partition_shuffled_set(names, 8)
    print 'train', train
    print 'test', test
    print 'names', names

# test_partition_training_set()


# ----------------------------------------------------------------------

def generate_music21_bach_chorale_hamlet_dataset\
                (source_root, dest_dir, num_train, mode=None, gen_analysis_meta_p=True):
    """

    score_by_mode_dict: <mode> -> <list-of-names>
    symbol_to_chord_map: <int> -> <chord>

    :param source_root:
    :param dest_dir:
    :param num_train:
    :param mode:
    :param gen_analysis_meta_p: optionally produce analysis metadata (chord frequencies)
    :return:
    """

    print '\n======================================================'
    print 'Running generate_music21_bach_chorale_hamlet_dataset()'

    # store sequence of notes about properties of the data to add to data_properties.txt
    data_properties = list()
    data_properties.append(('data_source', '{0}'.format(source_root)))
    data_properties.append(('data_root', '{0}'.format(dest_dir)))
    data_properties.append(('# in following: train = obs/ , test = test_obs/', ''))
    data_properties.append(('mode', '{0}'.format(mode)))
    data_properties.append(('num_train_files', '{0}'.format(num_train)))

    # read score_mode.txt into score_mode_dict to get all chorale names and their mode
    print '(1) Reading <source> score_mode.txt'
    score_by_mode_dict = dict()
    source_score_mode_filepath = os.path.join(source_root, 'score_mode.txt')
    with open(source_score_mode_filepath, 'r') as fin:
        for line in fin.readlines():
            name, m = line.strip().split()
            if m in score_by_mode_dict:
                score_by_mode_dict[m].append(name)
            else:
                score_by_mode_dict[m] = [name]

    # create <dest>/, <dest>/obs/ and <dest>/test_obs/ directories
    print '(2) Creating <dest>/, <dest>/obs/ and <dest>/test_obs/ directories, if don\'t already exist'
    obs_dir = os.path.join(dest_dir, 'obs')
    test_dir = os.path.join(dest_dir, 'test_obs')
    properties_dir = os.path.join(dest_dir, 'data_properties')
    if not os.path.exists(dest_dir):
        print '[NOTE] generate_music21_bach_chorale_hamlet_dataset(): dir does not exit, creating:'
        print '       \'{0}\''.format(dest_dir)
        os.makedirs(dest_dir)
    if not os.path.exists(obs_dir):
        print '[NOTE] generate_music21_bach_chorale_hamlet_dataset(): dir does not exit, creating:'
        print '       \'{0}\''.format(obs_dir)
        os.makedirs(obs_dir)
    if not os.path.exists(test_dir):
        print '[NOTE] generate_music21_bach_chorale_hamlet_dataset(): dir does not exit, creating:'
        print '       \'{0}\''.format(test_dir)
        os.makedirs(test_dir)
    if not os.path.exists(properties_dir):
        print '[NOTE] generate_music21_bach_chorale_hamlet_dataset(): dir does not exit, creating:'
        print '       \'{0}\''.format(properties_dir)
        os.makedirs(properties_dir)

    # copy <source_root>/score_mode.txt
    print '(3) Copying <source_root>/score_mode.txt to <dest_dir/properties_dir>'
    dest_score_mode_filepath = os.path.join(properties_dir, 'score_mode.txt')
    shutil.copy(source_score_mode_filepath, dest_score_mode_filepath)

    # copy <source_root>/symbol_to_chord_map.txt
    print '(4) Copying <source_root>/symbol_to_chord_map.txt to <dest_dir/properties_dir>/orig_symbol_to_chord_map.txt'
    source_symbol_to_chord_map_filepath = os.path.join(source_root, 'symbol_to_chord_map.txt')
    dest_symbol_to_chord_map_filepath = os.path.join(properties_dir, 'orig_symbol_to_chord_map.txt')
    shutil.copy(source_symbol_to_chord_map_filepath, dest_symbol_to_chord_map_filepath)

    # determine names based on mode: None=all, major, minor
    if mode is None:
        names = list()
        for names_list in score_by_mode_dict.values():
            names += names_list
    else:
        names = score_by_mode_dict[mode]

    # get random partition train_names, test_names
    train_names, test_names = partition_shuffled_set(names, num_train)

    source_score_root = os.path.join(source_root, 'sources')

    data_properties.append(('num_test_files', '{0}'.format(len(test_names))))

    # bookkeeping of mapping of original symbols to new symbols
    orig_to_new_symbol_map = dict()
    new_to_orig_symbol_map = dict()
    new_symbol_idx = 0

    # read train_names score sources
    # renumber, store renumbering in new_to_orig_symbol_map
    # save to obs/
    print '(5) Reading train_names original scores, renumber, save to obs/'
    obs_manifest_map = dict()
    for i, name in enumerate(train_names):
        src_score_filename = '{0}.txt'.format(name)
        src_path = os.path.join(source_score_root, src_score_filename)
        dst_score_filename = '{0}.txt'.format(str(i + 1).zfill(3))  # filenames start at 001
        dst_path = os.path.join(obs_dir, dst_score_filename)
        with open(src_path, 'r') as fin:
            with open(dst_path, 'w') as fout:
                for line in fin.readlines():
                    orig_symbol = int(line.strip())

                    if orig_symbol in orig_to_new_symbol_map:
                        new_symbol = orig_to_new_symbol_map[orig_symbol]
                    else:
                        new_symbol = new_symbol_idx
                        orig_to_new_symbol_map[orig_symbol] = new_symbol_idx
                        new_to_orig_symbol_map[new_symbol_idx] = orig_symbol
                        new_symbol_idx += 1

                    fout.write('{0}\n'.format(new_symbol))

        obs_manifest_map[dst_score_filename] = name

    '''
    # copy scores for train_names to obs/
    print '(5) Copying scores for train_names to obs/'
    obs_manifest_map = dict()
    for i, name in enumerate(train_names):
        score_filename = '{0}.txt'.format(name)
        src_path = os.path.join(source_score_root, score_filename)
        dst_filename = '{0}.txt'.format(str(i + 1).zfill(3))  # filenames start at 001
        dst_path = os.path.join(obs_dir, dst_filename)
        shutil.copy(src_path, dst_path)
        obs_manifest_map[dst_filename] = name
    '''

    # save manifest_obs.txt
    print '(6) Saving manifest_obs.txt'
    manifest_obs_path = os.path.join(properties_dir, 'manifest_obs.txt')
    with open(manifest_obs_path, 'w') as fout:
        for obs_filename in sorted(obs_manifest_map.keys()):
            fout.write('{0} {1}\n'.format(obs_filename, obs_manifest_map[obs_filename]))

    # read test_names score sources
    # renumber, store renumbering in new_to_orig_symbol_map
    # save to test_obs/
    print '(7) Reading test_names original scores, renumber, save to test_obs/'
    test_manifest_map = dict()
    for i, name in enumerate(test_names):
        src_score_filename = '{0}.txt'.format(name)
        src_path = os.path.join(source_score_root, src_score_filename)
        dst_score_filename = '{0}.txt'.format(str(i + 1).zfill(3))  # filenames start at 001
        dst_path = os.path.join(test_dir, dst_score_filename)
        with open(src_path, 'r') as fin:
            with open(dst_path, 'w') as fout:
                for line in fin.readlines():
                    orig_symbol = int(line.strip())

                    if orig_symbol in orig_to_new_symbol_map:
                        new_symbol = orig_to_new_symbol_map[orig_symbol]
                    else:
                        new_symbol = new_symbol_idx
                        orig_to_new_symbol_map[orig_symbol] = new_symbol_idx
                        new_to_orig_symbol_map[new_symbol_idx] = orig_symbol
                        new_symbol_idx += 1

                    fout.write('{0}\n'.format(new_symbol))

        # make copy of <root>/test_obs/001.txt to <root>/test_obs.txt
        if i == 0:
            dst_test_obs_path = os.path.join(dest_dir, 'test_obs.txt')
            shutil.copy(dst_path, dst_test_obs_path)

        test_manifest_map[dst_score_filename] = name

    '''
    # copy scores for test_name to test_obs/
    print '(7) Copying scores for test_name to test_obs/'
    test_manifest_map = dict()
    for i, name in enumerate(test_names):
        score_filename = '{0}.txt'.format(name)
        src_path = os.path.join(source_score_root, score_filename)
        dst_filename = '{0}.txt'.format(str(i + 1).zfill(3))  # filenames start at 001
        dst_path = os.path.join(test_dir, dst_filename)
        shutil.copy(src_path, dst_path)
        if i == 0:
            dst_test_obs_path = os.path.join(dest_dir, 'test_obs.txt')
            shutil.copy(src_path, dst_test_obs_path)
        test_manifest_map[dst_filename] = name
    '''

    # save manifest_test_obs.txt
    print '(8) Saving manifest_test_obs.txt'
    manifest_test_obs_path = os.path.join(properties_dir, 'manifest_test_obs.txt')
    with open(manifest_test_obs_path, 'w') as fout:
        for obs_filename in sorted(test_manifest_map.keys()):
            fout.write('{0} {1}\n'.format(obs_filename, test_manifest_map[obs_filename]))

    # save new_to_orig_symbol_map
    print '(9) Saving new_to_orig_symbol_map.txt'
    new_to_orig_symbol_map_path = os.path.join(properties_dir, 'new_to_orig_symbol_map.txt')
    with open(new_to_orig_symbol_map_path, 'w') as fout:
        for new_symbol, orig_symbol in new_to_orig_symbol_map.iteritems():
            fout.write('{0} {1}\n'.format(new_symbol, orig_symbol))

    # Record total number of new_symbols
    # NOTE: there are three possible interpretations
    # (a) The total number of symbols in a 0-based index scheme (i.e.: [0,...,K-1]) (so there are K symbols)
    # (b) The highest index of a 1-based index scheme (i.e.: [1,...,K]) (so there are K symbols)
    # (c) The highest index of a 0-based index scheme (i.e.: [0,...,K]) (so there are K+1 symbols)
    # Current scheme assumes (a) or (c) b/c symbols start 0-based.
    # Currently assuming (c) is the case, by: new_symbol_idx - 1
    # if (a) is correct, then need just: new_symbol_idx
    data_properties.append(('# total_symbols', '{0}'.format(new_symbol_idx - 1)))
    data_properties.append(('Dirichlet_multinomial_emissions K', '{0}'.format(new_symbol_idx - 1)))

    # ----- Analysis metadata: chord frequencies in obs / test_obs:

    if gen_analysis_meta_p:
        print '(10) Generating analysis metadata for chord frequencies in obs and test_obs'

        # read <source> symbol_to_chord_map.txt into symbol_to_chord_map dict
        # NOTE: these are orig_symbols
        print '(10a) Reading <source_root>/symbol_to_chord_map.txt'
        symbol_to_chord_map = dict()
        with open(source_symbol_to_chord_map_filepath, 'r') as fin:
            for line in fin.readlines():
                elms = line.strip().split()
                symbol = int(elms[0])
                chord = eval(''.join(elms[1:]))
                symbol_to_chord_map[symbol] = chord

        # read <dest_obs>/*.txt into obs_symbol_corpus_frequency_dict, counting symbol freq
        # NOTE: these are new_symbols
        print '(10b) Reading <dest_obs>/*.txt symbols'
        total_obs_chords = 0
        obs_symbol_corpus_frequency_dict = dict()
        for obs_filename in os.listdir(obs_dir):
            obs_file_path = os.path.join(obs_dir, obs_filename)
            with open(obs_file_path, 'r') as fin:
                for line in fin.readlines():
                    total_obs_chords += 1
                    symbol = int(line.strip())
                    if symbol in obs_symbol_corpus_frequency_dict:
                        obs_symbol_corpus_frequency_dict[symbol] += 1
                    else:
                        obs_symbol_corpus_frequency_dict[symbol] = 1

        # read <dest_test_obs>/*.txt into test_obs_symbol_corpus_frequency_dict, counting symbol freq
        # NOTE: these are new_symbols
        print '(10c) Reading <dest_test_obs>/*.txt symbols'
        total_test_obs_chords = 0
        test_obs_symbol_corpus_frequency_dict = dict()
        for test_obs_filename in os.listdir(test_dir):
            test_obs_file_path = os.path.join(test_dir, test_obs_filename)
            with open(test_obs_file_path, 'r') as fin:
                for line in fin.readlines():
                    total_test_obs_chords += 1
                    symbol = int(line.strip())
                    if symbol in test_obs_symbol_corpus_frequency_dict:
                        test_obs_symbol_corpus_frequency_dict[symbol] += 1
                    else:
                        test_obs_symbol_corpus_frequency_dict[symbol] = 1

        # create chord_corpus_frequency_obs.txt
        # iterate through symbol_to_chord_map and save chord paired with value as follows:
        # (1) if chord symbol is in obs_symbol_corpus_frequency_dict, save the int
        # (2) if not in obs_symbol_corpus_frequency_dict
        #     but IS in test_obs_symbol_corpus_frequency_dict, save value as 0
        # (3) otherwise, set as -1
        # Semantics:
        #     positive int : frequency in obs/
        #     0 : occurs in test_obs/ but not obs/
        #     -1 : occurs in source corpus but neither obs/ or test_obs/
        print '(10d) Saving chord_corpus_frequency_obs.txt'
        unique_obs_symbols = set()
        chords_in_test_not_in_train = set()
        chord_corpus_frequency_obs_filepath = os.path.join(properties_dir, 'chord_corpus_frequency_obs.txt')

        with open(chord_corpus_frequency_obs_filepath, 'w') as fout:
            for orig_symbol, chord in symbol_to_chord_map.iteritems():

                # map orig_symbol to new_symbol, if was used in new corpus
                new_symbol = None
                if orig_symbol in orig_to_new_symbol_map:
                    new_symbol = orig_to_new_symbol_map[orig_symbol]

                if new_symbol in obs_symbol_corpus_frequency_dict:
                    unique_obs_symbols.add(new_symbol)
                    fout.write('{0} {1}\n'.format(chord, obs_symbol_corpus_frequency_dict[new_symbol]))
                elif new_symbol in test_obs_symbol_corpus_frequency_dict:
                    chords_in_test_not_in_train.add(new_symbol)
                    fout.write('{0} {1}\n'.format(chord, 0))
                else:
                    fout.write('{0} {1}\n'.format(chord, -1))

        # create chord_corpus_frequency_test_obs.txt
        print '(10e) Saving chord_corpus_frequency_test_obs.txt'
        unique_test_obs_symbols = set()
        chords_in_train_not_in_test = set()
        chord_corpus_frequency_test_obs_filepath = os.path.join(properties_dir, 'chord_corpus_frequency_test_obs.txt')
        with open(chord_corpus_frequency_test_obs_filepath, 'w') as fout:
            for orig_symbol, chord in symbol_to_chord_map.iteritems():

                # map orig_symbol to new_symbol, if it was used in new corpus
                new_symbol = None
                if orig_symbol in orig_to_new_symbol_map:
                    new_symbol = orig_to_new_symbol_map[orig_symbol]

                if new_symbol in test_obs_symbol_corpus_frequency_dict:
                    unique_test_obs_symbols.add(new_symbol)
                    fout.write('{0} {1}\n'.format(chord, test_obs_symbol_corpus_frequency_dict[new_symbol]))
                elif new_symbol in obs_symbol_corpus_frequency_dict:
                    chords_in_train_not_in_test.add(new_symbol)
                    fout.write('{0} {1}\n'.format(chord, 0))
                else:
                    fout.write('{0} {1}\n'.format(chord, -1))

        data_properties.append(('total_train_chords', '{0}'.format(total_obs_chords)))
        data_properties.append(('total_unique_train_chords', '{0}'.format(len(unique_obs_symbols))))
        data_properties.append(
            ('total_unique_chords_in_train_not_in_test', '{0}'.format(len(chords_in_train_not_in_test))))
        data_properties.append(('total_test_chords', '{0}'.format(total_test_obs_chords)))
        data_properties.append(('total_unique_test_chords', '{0}'.format(len(unique_test_obs_symbols))))
        data_properties.append(
            ('total_unique_chords_in_test_not_in_train', '{0}'.format(len(chords_in_test_not_in_train))))

        data_properties.append(('# per test: total_chords, num_unique_chords, total_chords_not_in_train,'
                                + ' num_unique_chords_not_in_train', ''))
        print '(10f) Collect number of chords in test sets NOT in training corpus'
        for test_obs_filename in os.listdir(test_dir):
            dst_test_obs_path = os.path.join(test_dir, test_obs_filename)
            with open(dst_test_obs_path, 'r') as fin:
                total_chords = 0
                unique_chords = set()
                total_chords_in_test_not_in_train = 0
                unique_chords_in_test_not_in_train = set()
                for line in fin.readlines():
                    symbol = int(line.strip())
                    total_chords += 1
                    unique_chords.add(symbol)
                    if symbol in test_obs_symbol_corpus_frequency_dict \
                            and symbol not in obs_symbol_corpus_frequency_dict:
                        unique_chords_in_test_not_in_train.add(symbol)
                        total_chords_in_test_not_in_train += 1
                data_properties.append((test_obs_filename, '{0} {1} {2} {3}'\
                                        .format(total_chords,
                                                len(unique_chords),
                                                total_chords_in_test_not_in_train,
                                                len(unique_chords_in_test_not_in_train))))

    print '(11) Saving summary.txt'
    data_properties_filepath = os.path.join(properties_dir, 'summary.txt')
    with open(data_properties_filepath, 'w') as fout:
        for prop, value in data_properties:
            fout.write('{0} {1}\n'.format(prop, value))

    print 'DONE.'


# ----------------------------------------------------------------------

def generate_bach_data(data_postfix='_test'):
    
    bach_major_root = os.path.join(BACH_CHORALE_NOMINAL_ROOT, 'bach_major{0}'.format(data_postfix))
    
    generate_music21_bach_chorale_hamlet_dataset(BACH_CHORALE_NOMINAL_SOURCE_ROOT,
                                                 bach_major_root,
                                                 num_train=200,
                                                 mode='major',
                                                 gen_analysis_meta_p=True)

    '''
    bach_minor_root = os.path.join(BACH_CHORALE_NOMINAL_ROOT, 'bach_minor{0}'.format(data_postfix))
    
    generate_music21_bach_chorale_hamlet_dataset(BACH_CHORALE_NOMINAL_SOURCE_ROOT,
                                                 bach_minor_root,
                                                 num_train=160,
                                                 mode='minor',
                                                 gen_analysis_meta_p=True)
    '''

# generate_bach_data(data_postfix='_01')


# ----------------------------------------------------------------------
# Debugging: trying to figure out why test_obs/016.txt breaks hamlet
# ----------------------------------------------------------------------

BACH_MAJOR_ROOT = os.path.join(BACH_CHORALE_NOMINAL_ROOT, 'bach_major_01')

print BACH_MAJOR_ROOT
print os.listdir(BACH_MAJOR_ROOT)


def read_test_obs_016():
    values = numpy.loadtxt(os.path.join(BACH_MAJOR_ROOT, 'test_obs/016.txt'), dtype=int)
    print set(values)
    print 'total chords', values.shape[0]
    print 'num unique chords', len(set(values))
    print 'min chord symbol', numpy.max(values)
    print 'max chord symbol', numpy.min(values)

# read_test_obs_016()


# ----------------------------------------------------------------------
# Deprecated
# ----------------------------------------------------------------------

def score_batch_analysis(processed_score_dict, symbol_to_chord_map, chord_count_dict):

    # for key, val in chord_to_symbol_map.iteritems():
    #     print key, val, chord_count_dict[key]
    # print collections.Counter(chord_count_dict.values())

    print '=========='

    # count singletons
    singletons_count_list = list()
    for name, symbol_list in processed_score_dict.iteritems():
        singletons = list()
        for symbol in symbol_list:
            chord_tuple = symbol_to_chord_map[symbol]
            if chord_count_dict[chord_tuple] == 1:
                singletons.append((symbol, chord_tuple))
        # print name, len(singletons), singletons
        singletons_count_list.append(len(singletons))
    # print collections.Counter(singletons_count_list)

    # compute overlap:
    # for each chorale, see if it contains all of the chords or each other chorale
    # n^2
    for name1, symbol_list1 in processed_score_dict.iteritems():
        overlap_list = list()
        for name2, symbol_list2 in processed_score_dict.iteritems():
            if name1 != name2:
                remainder = set(symbol_list2) - set(symbol_list1)
                if len(remainder) == 0:
                    overlap_list.append(name2)
        if len(overlap_list) > 0:
            print name1, ':', overlap_list
        else:
            print name1, 'None'


# chordify_and_symbolize()


# ----------------------------------------------------------------------

def view_chorale():
    music21_bc_path_list = music21.corpus.getBachChorales()

    chorales_with_less4 = list()

    for i, chorale in enumerate(music21_bc_path_list):
        print i, chorale

        s = music21.converter.parse(chorale)
        # s.show('text')
        # s.show()

        voices = 0
        for elm in list(s):

            if type(elm) is music21.stream.Part:
                if elm.id not in ('Soprano', 'Alto', 'Tenor', 'Bass',
                                  'S.', 'A.', 'T.', 'B.'):
                    # print '    ', elm.id
                    pass
                else:
                    # print '*** ', elm.id
                    voices += 1

        if voices < 4:
            chorales_with_less4.append(chorale)
        else:
            chordify_4voice_chorale(s)

    print '-----'
    for c in chorales_with_less4:
        print c

    # chords = s.chordify()
    # chords.show('text')
    # chords.show()

# view_chorale()

# ----------------------------------------------------------------------
