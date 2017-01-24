import xlrd
import music21
import os
import collections


"""
20170120
Adapted from music/data/bach_analysis/extract_chorale_names.py
"""


# --------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------

def chordify_4voice_chorale(stream, chord_to_symbol_map, sym_idx,
                            chord_count_dict,
                            symbol_to_chord_map):

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

        # test for any cases of voices being in different keys
        if len(key_list) > 1: more_than_1_keys = True

        # test for any cases of key change in chorale
        if i == 0 and key_list is not None:
            first_key = key_list[0]
        else:
            if key_list[0] != first_key:
                different_keys = True

    if num_voices == 4 and different_keys is False and more_than_1_keys is False:
        # print 'keys', keys
        # print 'keys.keys()', keys.keys()
        # print keys[keys.keys()[0]][0]

        key_pitchClass = keys[keys.keys()[0]][0].tonic.pitchClass
        chords = stream.chordify()

        # chords.show('text')
        # chords.show()
        # print '----'

        for elm in chords.flat:
            if type(elm) is music21.chord.Chord:

                midi_pitch_list = list()
                for p in list(elm.pitches):
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
           chord_count_dict, symbol_to_chord_map


def test_chordify_4voice_chorale():
    music21_bc_path_list = music21.corpus.getBachChorales()
    s = music21.converter.parse(music21_bc_path_list[0])
    print music21_bc_path_list[0]
    # s.show('text')
    # chordify_4voice_chorale(s)

# test_chordify_4voice_chorale()


def chordify_and_symbolize():
    music21_bc_path_list = music21.corpus.getBachChorales()
    chord_to_symbol_map = dict()
    sym_idx = 0
    chord_count_dict = dict()
    symbol_to_chord_map = dict()
    processed_score_dict = dict()

    for i, bc_path in enumerate(music21_bc_path_list):  # [:100]
        name = os.path.splitext(os.path.basename(bc_path))[0]
        print '-----', i, name
        # if name == 'bwv171.6':
        s = music21.converter.parse(bc_path)
        chord_symbol_list, chord_to_symbol_map, sym_idx, chord_count_dict, symbol_to_chord_map \
            = chordify_4voice_chorale(s, chord_to_symbol_map, sym_idx,
                                      chord_count_dict, symbol_to_chord_map)
        processed_score_dict[name] = chord_symbol_list

    # for name, csl in processed_score_dict.iteritems():
    #     print name, len(csl), csl

    print len(chord_to_symbol_map.keys())

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


# --------------------------------------------------------------------------------

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
