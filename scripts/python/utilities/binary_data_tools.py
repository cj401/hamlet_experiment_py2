import numpy as np
import collections

__author__ = 'clayton'


# ---------------------------------------------------------------------
# read_binary_vector_time_series_as_lol  (lol = list-of-lists)
# ---------------------------------------------------------------------


def read_binary_vector_time_series_as_lol(data_filename):
    """
    Read binary vector time series as list of lists
    :param data_filename:
    :return:
    """
    data = []
    with open(data_filename, 'r') as fin:
        for row in fin.readlines():
            # print row.strip(' \n').split(' ')
            v = []
            for elm in row.strip(' \n').split(' '):
                # print elm
                v.append(int(elm))
            data.append(v)
    return data


# ---------------------------------------------------------------------
# find_binary_segments
# ---------------------------------------------------------------------


def find_binary_segments(bit_sequence):
    """
    :param bit_sequence: sequence of 1's and 0's
    :return: list of (<start>, <end>, <bit>)
    """
    prev_bit = bit_sequence[0]
    segments = list()
    prev_idx = 0
    for index, bit in enumerate(bit_sequence):
        if bit != prev_bit:
            segments.append((prev_idx, index, prev_bit))
            prev_bit = bit
            prev_idx = index
    segments.append((prev_idx, len(bit_sequence), prev_bit))
    return segments


def test_find_binary_segments(verbose=False):

    def a2b(string):
        """
        ascii to binary array
        """
        return np.array([ int(elm) for elm in string ])

    bexample1 = a2b('000011000001001011110000')
    bexample1_indices = [(0, 4, 0), (4, 6, 1), (6, 11, 0), (11, 12, 1), (12, 14, 0),
                         (14, 15, 1), (15, 16, 0), (16, 20, 1), (20, 24, 0)]
    bexample2 = a2b('100011000001001011110001')
    bexample2_indices = [(0, 1, 1), (1, 4, 0), (4, 6, 1), (6, 11, 0), (11, 12, 1),
                         (12, 14, 0), (14, 15, 1), (15, 16, 0), (16, 20, 1), (20, 23, 0),
                         (23, 24, 1)]
    bexample3 = a2b('101010101010101010101010')
    bexample3_indices = [(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 4, 0), (4, 5, 1), (5, 6, 0),
                         (6, 7, 1), (7, 8, 0), (8, 9, 1), (9, 10, 0), (10, 11, 1),
                         (11, 12, 0), (12, 13, 1), (13, 14, 0), (14, 15, 1), (15, 16, 0),
                         (16, 17, 1), (17, 18, 0), (18, 19, 1), (19, 20, 0), (20, 21, 1),
                         (21, 22, 0), (22, 23, 1), (23, 24, 0)]

    if verbose:
        print bexample1
        print bexample1_indices
        print find_binary_segments(bexample1)

        print bexample2
        print bexample2_indices
        print find_binary_segments(bexample2)

        print bexample3
        print bexample3_indices
        print find_binary_segments(bexample3)

    assert find_binary_segments(bexample1) == bexample1_indices
    assert find_binary_segments(bexample2) == bexample2_indices
    assert find_binary_segments(bexample3) == bexample3_indices

    print 'TEST find_binary_segments() PASSED'

test_find_binary_segments()


# ---------------------------------------------------------------------
# ColumnData and get_columns_with_at_least_one
# ---------------------------------------------------------------------


ColumnData = collections.namedtuple\
    ('ColumnData',
     ['figures',
      'indices',
      'idx_to_downscale_dict',
      'sum',
      'off_indices'])


def get_columns_with_at_least_one(data, ignore_list=(), include_list=None):
    """

    :param data:
    :param ignore_list:
    :param include_list:
    :return:
    """
    # rows are time ticks, columns are context
    data = np.array(data)
    column_on_indices = list()
    column_on_sum = list()
    column_off_indices = list()

    # print figures.shape
    # print 'figures number of columns:', figures.shape[1]

    for cidx in range(data.shape[1]):
        column = data[:, cidx]
        column_sum = np.sum(column)
        if include_list:
            # if there is an include_list, then only collect context indices in that list
            if (column_sum > 0) and (cidx in include_list):
                column_on_indices.append(cidx)
                column_on_sum.append(column_sum)
            else:
                column_off_indices.append(cidx)
        else:
            # no include_list
            # only collect those NOT in the ignore_list
            if (column_sum > 0) and (cidx not in ignore_list):
                column_on_indices.append(cidx)
                column_on_sum.append(column_sum)
            else:
                column_off_indices.append(cidx)

    # column_on_idx_to_downscale_dict: maps orig_context_idx to new (downscaled) index
    column_on_idx_to_downscale_dict = dict()
    # column_on_indicies already maps the new (downscaled) index to the orig_context_idx:
    #    its element position index is the new downscaled index, and the value is the orig_context_idx
    for downscale_idx, orig_context_idx in enumerate(column_on_indices):
        column_on_idx_to_downscale_dict[orig_context_idx] = downscale_idx

    column_on_data = np.take(data, column_on_indices, axis=1)

    return ColumnData(data=column_on_data,
                      indices=column_on_indices,
                      idx_to_downscale_dict=column_on_idx_to_downscale_dict,
                      sum=column_on_sum,
                      off_indices=column_off_indices)


def test_get_columns_with_at_least_one(verbose=False):
    data = np.array([[0, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])
    cd = get_columns_with_at_least_one(data)

    if verbose:
        print 'context_on_data:\n{0}'.format(cd.data)
        print 'context_on_indices: {0}'.format(cd.indices)
        print 'context_on_idx_to_downscale_dict: {0}'.format(cd.idx_to_downscale_dict)
        print 'context_on_sum: {0}'.format(cd.sum)
        print 'context_off_indices: {0}'.format(cd.off_indices)

    assert np.all(np.equal(cd.data, np.array([[0, 0],
                                              [1, 0],
                                              [0, 0],
                                              [1, 0],
                                              [0, 1]])))
    assert np.all(np.equal(np.array(cd.indices), np.array([1, 2])))
    assert np.all(np.equal(np.array(cd.sum), np.array([2, 1])))
    assert np.all(np.equal(np.array(cd.off_indices), np.array([0, 3])))
    print 'TEST test_get_active_context_indices() PASSED'


test_get_columns_with_at_least_one()  # (verbose=True)

# ---------------------------------------------------------------------
