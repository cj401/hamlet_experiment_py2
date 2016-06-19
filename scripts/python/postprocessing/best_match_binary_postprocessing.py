import itertools
import os
import shutil
import sys
import timeit
from distutils.dir_util import copy_tree

import nose
import numpy as np
import scipy.spatial.distance

from utilities import util

__author__ = 'clayton'

"""
Find best pairing between two sets of binary vectors
"Best" means minimizing sum L1 (Hamming) distance across the pairs
Consider that bits could be flipped, so test each pair (a) directly
and (b) flipping one vector, and return the score and pairing with
the smallest distance

The core script does the following:
() Reads results parameters.config to get path to original figures with ground-truth states
() Creates new directory called 'old_statistics' and moves
    'accuracy.txt', 'precision.txt', 'recall.txt', 'F1_score.txt' to old_statistics
() Creates new versions of 'accuracy.txt', 'precision.txt', 'recall.txt', 'F1_score.txt'
    that only have 'iterations value' header
() Moves current thetastar to thetastar_orig
() Creates new thetastar directory
() Creates new thetastar_match_specs (for files describing the permutation and
    flip bit vectors for any best-match)
() Reads gold-standard figures states.txt
() Reads thetastar_orig directory file listing
() Iterates through each file in thetastar_orig:
    () Read thetastar_orig file states
    () Find best-match of data_states to thetastar_orig_file_states
    () Permute thetastar_orig_file_states according to best match and
        save in the new thetastar directory
        (same name as thetastar_orig_file)
    () Save best match permutation spec to thetastar_match_specs/
        (same name as thetastar_orig_file)
    () Compute performance statistics of best-fit thetastar_orig_file_states
        to data_states:
            accuracy:  TP + TN / (TP + FP + TN + FN)
            precision: TP / (TP + FP)
            recall:    TP / (TP + FN)
            F1_score:  2 * (Precision * Recall) / (Precision + Recall)
        and append iteration (thetastar_orig_file base name) and
        statistics value to new statistics files in results root
"""

"""
R scripts:
install abind  (array binding -- stacking multi-dimensional arrays)
run R from <hamlet>/experiment/r/scripts/
source("define_functions.R")

batch function: make_scalar_plots_batch()
query file name (quoted)
data_set (quoted): within results directory root, specify figures
    e.g., "cocktail/"
    then depth 2 (b/c figures, model subdirs)
    if want subset of figures, then specify figures, with depth 1
burn-in samples - often 11

ndot == number of different states used

make_scalar_plots_batch("cocktail_all.txt", "cocktail/", 11)

make_scalar_plots_batch("w1_all.txt", "cocktail/", 11, "*")

# on laplace -- need to change project_root
make_scalar_plots_batch("w1_best_match_test.txt", "cocktail", 11, "*", project_root = "../../../")

# on venti -- need to change project_root
make_scalar_plots_batch("w1_best_match_test.txt", "cocktail", 11, "*", project_root = "/projects/hamlet/figures/")

cocktail_w1_F1.txt

"""


bstate_vec_file1 = '../figures/cocktail/noise_a1b1/cp_0/states.txt'
bstate_vec_file2 = '01000.txt'


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_binary_vector(filepath, fail_if_not_digit=False):
    matrix = []
    with open(filepath, 'r') as fp:
        for line in fp.readlines():
            vec = []
            for elm in line.split(' '):
                if elm:
                    if elm.isdigit():
                        vec.append(int(elm))
                    elif is_number(elm):
                        vec.append(int(float(elm)))
                    else:
                        print 'not a digit: \'{0}\''.format(elm)
                        if fail_if_not_digit:
                            sys.exit(-1)
            matrix.append(vec)
    return np.array(matrix)


def comparable_p(matrix1, matrix2):
    if matrix1.shape[0] == matrix2.shape[0]:
        return True
    return False


def flip(vec):
    nv = np.array(vec)
    for i in range(len(nv)):
        if nv[i] == 0:
            nv[i] = 1
        else:
            nv[i] = 0
    return nv


# ---------------------------------------------------------------------
# HAMMING min distance

def min_hamming_distance(u, v):
    """
    Computes the hamming distance between u,v and u,flip(v), returning
    the smaller distance of the two.
    Also report whether the smallest distance involved the flip (1) or not (0)
    :param u: vector length n
    :param v: vector length n
    :return: shortest hamming distance, whether v was flipped to get shortest distance
    """
    d = scipy.spatial.distance.hamming(u, v)
    f = scipy.spatial.distance.hamming(u, flip(v))
    if d < f:
        return d, int(0)
    else:
        return f, int(1)


def test_min_hamming_distance(verbose=False):
    u = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    v = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    w = np.array([1, 1, 0, 1, 1, 1, 0, 1])

    d, f = min_hamming_distance(u, v)
    if verbose:
        print '{0}\n{1}\n{2}, {3}'.format( u, v, d, f )
    assert d == 0.375
    assert f == 0

    d, f = min_hamming_distance(u, w)
    if verbose:
        print '{0}\n{1}\n{2}, {3}'.format( u, w, d, f )
    assert d == 0.0
    assert f == 1

    d, f = min_hamming_distance(v, w)
    if verbose:
        print '{0}\n{1}\n{2}, {3}'.format( v, w, d, f )
    assert d == 0.375
    assert f == 1

    print 'TEST min_hamming_distance() PASSED'

test_min_hamming_distance()


# ---------------------------------------------------------------------
# F1 max score

def compute_f1(states_gt, states_inferred):
    """
    Calculate F1 for two binary matrices, where the first is the ground-truth,
    the second is inferred.
    Calculates:
        precision: TP / (TP + FP)
        recall:    TP / (TP + FN)
        F1_score:  2 * (Precision * Recall) / (Precision + Recall)
    :param states_gt: binary matrix considered ground-truth
    :param states_inferred: binary matrix considered inferred hypothesis
    :return: F1 score
    """

    tp = np.sum(np.logical_and(states_gt, states_inferred))
    fp = np.sum(states_inferred) - tp
    fn = np.sum(states_gt) - tp

    precision = 0.0
    if float(tp + fp) > 0.0:
        precision = float(tp) / float(tp + fp)

    recall = 0.0
    if float(tp + fn) > 0.0:
        recall = float(tp) / float(tp + fn)

    if precision + recall > 0.0:
        return 2.0 * (precision * recall) / (precision + recall)
    else:
        return 0.0


def compute_f1_1d(states_gt, states_inferred):
    """
    Calculate F1
    This is FASTER when computing over 1 dimensional (row or column) 2-d vectors,
    but SLOWER for larger matrices or 1-d vectors.  Go figure...
    Calculate
        precision: TP / (TP + FP)
        recall:    TP / (TP + FN)
        F1_score:  2 * (Precision * Recall) / (Precision + Recall)

    :param states_gt:
    :param states_inferred:
    """
    tp, fp, fn = 0, 0, 0
    for i in range(states_inferred.shape[0]):
        if states_gt[i] == 1:
            if states_inferred[i] == 1:
                tp += 1  # 1 1
            else:
                fn += 1  # 1 0
        else:
            if states_inferred[i] == 1:
                fp += 1  # 0 1
                # else:
                #     TN += 1  # 0 0
    precision = 0.0
    if float(tp + fp) > 0.0:
        precision = float(tp) / float(tp + fp)

    recall = 0.0
    if float(tp + fn) > 0.0:
        recall    = float(tp) / float(tp + fn)

    if precision + recall > 0.0:
        return 2.0 * (precision * recall) / (precision + recall)
    else:
        return 0.0


# a = np.array([[1, 0, 0, 1, 0],
#               [1, 0, 1, 0, 0],
#               [1, 0, 1, 1, 0],
#               [1, 1, 1, 0, 0],
#               [0, 1, 1, 0, 0],
#               [0, 1, 0, 1, 0]])
# b = np.array([[1, 0, 1, 0, 0],
#               [0, 0, 0, 1, 1],
#               [0, 0, 1, 1, 1],
#               [1, 0, 1, 1, 1],
#               [1, 0, 1, 0, 0],
#               [0, 0, 1, 1, 0]])
# c = np.array([[1], [1], [1], [1], [0], [0]])
# d = np.array([[1], [0], [0], [1], [1], [0]])
# e = np.array([[1, 1, 1, 1, 0, 0]])
# f = np.array([[1, 0, 0, 1, 1, 0]])
a = np.array([1, 1, 1, 1, 0, 0])
b = np.array([1, 0, 0, 1, 1, 0])


def test_compute_f1(verbose=False, timeit_p=False):

    # f1_1 = compute_f1(a, b)
    # f1_2 = compute_f1_1d(a, b)

    # f1_1cd = compute_f1(c, d)
    # f1_2cd = compute_f1_1d(c, d)

    # f1_1ef = compute_f1(e, f)
    # f1_2ef = compute_f1_1d(e, f)

    f1_1 = compute_f1(a, b)
    f1_2 = compute_f1_1d(a, b)

    if verbose:
        print 'f1_1: {0}'.format(f1_1)
        print 'f1_2: {0}'.format(f1_2)
        # print 'f1_1cd: {0}'.format(f1_1cd)
        # print 'f1_2cd: {0}'.format(f1_2cd)
        # print 'f1_1ef: {0}'.format(f1_1ef)
        # print 'f1_2ef: {0}'.format(f1_2ef)

    # nose.tools.assert_almost_equal(f1_1, 0.48275862069)
    # nose.tools.assert_almost_equal(f1_2, 0.48275862069)
    #
    # nose.tools.assert_almost_equal(f1_1cd, 0.571428571429)
    # nose.tools.assert_almost_equal(f1_2cd, 0.571428571429)
    #
    # nose.tools.assert_almost_equal(f1_1ef, 0.571428571429)
    # nose.tools.assert_almost_equal(f1_2ef, 0.571428571429)

    nose.tools.assert_almost_equal(f1_1, 0.571428571429)
    nose.tools.assert_almost_equal(f1_2, 0.571428571429)

    if timeit_p:
        print 'timeit f1_1   ', timeit.timeit('compute_f1(a, b)',
                                              setup='from __main__ import compute_f1, a, b',
                                              number=100000)
        print 'timeit f1_2 1d', timeit.timeit('compute_f1_1d(a, b)',
                                              setup='from __main__ import compute_f1_1d, a, b',
                                              number=1000000)
        # print 'timeit f1_1     ', timeit.timeit('compute_f1(a, b)',
        #                                         setup='from __main__ import compute_f1, a, b',
        #                                         number=100000)
        # print 'timeit f1_2 slow', timeit.timeit('compute_f1_column(a, b)',
        #                                         setup='from __main__ import compute_f1_column, a, b',
        #                                         number=100000)
        # print 'timeit f1_1cd     ', timeit.timeit('compute_f1(c, d)',
        #                                           setup='from __main__ import compute_f1, c, d',
        #                                           number=1000000)
        # print 'timeit f1_2cd slow', timeit.timeit('compute_f1_column(c, d)',
        #                                           setup='from __main__ import compute_f1_column, c, d',
        #                                           number=1000000)
        # print 'timeit f1_1ef     ', timeit.timeit('compute_f1(e, f)',
        #                                           setup='from __main__ import compute_f1, e, f',
        #                                           number=1000000)
        # print 'timeit f1_2ef slow', timeit.timeit('compute_f1_column(e, f)',
        #                                           setup='from __main__ import compute_f1_column, e, f',
        #                                           number=1000000)
    print 'TEST compute_f1() PASSED'

test_compute_f1()
# test_compute_f1(verbose=True, timeit_p=True)


def max_f1_score(states_gt, states_inferred):
    """
    Compute the F1 score of
        states_gt, states_inferred
    and
        states_gt, flip(states_inferred)
    and return the LARGER F1 score of the two.
    Also report whether the largest F1 involved the flip (1) or not (0)
    :param states_gt: vector of length n
    :param states_inferred: vector of length n
    :return: largest F1 score, whether states_inferred was flipped to get the largest F1 score
    """
    score_direct = compute_f1(states_gt, states_inferred)
    score_flipped = compute_f1(states_gt, flip(states_inferred))

    # print 's={0}, f={1}'.format(score_direct, score_flipped)

    # Handle case where state_gt is all zeros; if so, minimize sum
    if np.sum(states_gt) == 0:
        d_sum = np.sum(states_inferred)
        f_sum = np.sum(flip(states_inferred))
        if d_sum < f_sum:
            return score_direct, int(0)
        else:
            return score_flipped, int(1)

    if score_direct > score_flipped:
        # print 'returning s'
        return score_direct, int(0)
    else:
        # print 'returning f: {0}'.format(score_flipped)
        return score_flipped, int(1)


def test_max_f1_score_2():

    # ground_truth_columns
    c1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    c2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    c3 = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    c4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # flip of ground_truth
    f1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    f2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0])
    f3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    f4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # variation of ground_truth
    t1 = np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    t2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    t3 = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
    t4 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # t4 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # flip of variation of ground_truth
    tf1 = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    tf2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    tf3 = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    tf4 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # ground_truth
    ground_truth = np.zeros((len(c1), 4))
    for c, col in enumerate([c1, c2, c3, c4]):
        ground_truth[:, c] = col

    target_vary = np.zeros(ground_truth.shape)
    for c, col in enumerate([t1, t2, t3, t4]):
        target_vary[:, c] = col

    target_vary_perm = np.zeros(ground_truth.shape)
    for c, col in enumerate([t4, t1, t2, t3]):
        target_vary_perm[:, c] = col

    target_vary_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([tf1, tf2, tf3, tf4]):
        target_vary_flip[:, c] = col

    target_vary_perm_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([tf4, tf1, tf2, tf3]):
        target_vary_perm_flip[:, c] = col

    for c in range(4):
        score_direct = compute_f1(ground_truth[:, c], target_vary[:, c])
        score_flipped = compute_f1(ground_truth[:, c], flip(target_vary[:, c]))
        score, flip_bit = max_f1_score(ground_truth[:, c], target_vary[:, c])
        print '     c={0}: {1}, {2} ; s={3}, f={4}'.format(c, flip_bit, score, score_direct, score_flipped)
        """
        print '     {0}'.format(ground_truth[:, c])
        if flip_bit == 0:
            print '     {0}'.format(target_vary[:, c])
        else:
            print '     {0}'.format(flip(target_vary[:, c]))
        """

# test_max_f1_score_2()


def test_max_f1_score(verbose=False):
    u = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    v = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    w = np.array([1, 1, 0, 1, 1, 1, 0, 1])

    s, f = max_f1_score(u, v)
    if verbose: print '{0}\n{1}\n{2}, {3}'.format( u, v, s, f )
    assert s == 0.4
    assert f == 0

    s, f = max_f1_score(u, w)
    if verbose: print '{0}\n{1}\n{2}, {3}'.format( u, w, s, f )
    assert s == 1.0
    assert f == 1

    s, f = max_f1_score(v, w)
    if verbose: print '{0}\n{1}\n{2}, {3}'.format( v, w, s, f )
    nose.tools.assert_almost_equal(s, 0.444444444444)
    assert f == 0

    print 'TEST max_f1_score() PASSED'

# test_max_f1_score(verbose=True)
test_max_f1_score()


# ---------------------------------------------------------------------

def make_comparison_matrix(matrix1, matrix2, comparison_fn=min_hamming_distance):
    """
    For each matrix, compute the comparison_fn of each
    column of matrix1 to each column of matrix2, where comparison_fn
    is a function such as:
        min_hamming_distance
        max_f1_score
    The comparison, cm, and flip, fm, matrices are indexed as follows:
        row-index = matrix1 column vector index
        col-index = matrix2 column vector index
    :param matrix1: binary matrix NxM
    :param matrix2: binary matrix NxM
    :return: cm=comparison matrix, fm=flip matrix (0=no flip, 1=flip)
    """
    cm = np.zeros(shape=(matrix1.shape[1], matrix2.shape[1]))
    fm = np.zeros(shape=(matrix1.shape[1], matrix2.shape[1]))
    for i in range(matrix1.shape[1]):  # iterate across matrix1 columns
        for j in range(matrix2.shape[1]):  # iterate across matrix2 columns
            d, f = comparison_fn(matrix1[:, i], matrix2[:, j])
            cm[i][j] = d
            fm[i][j] = f
    return cm, fm


# ---------------------------------------------------------------------

def find_min_match(cm):
    """
    Given a comparison matrix, find the minimum sum of values
        for each permutation of columns,
            sum distances from rows and permuted columns
            (equivalently: for each permutation of columns, sum diagonal)
        return the permutation with the smallest sum.
    :param cm: comparison matrix, where
        rows = matrix1 col vector index
        cols = matrix2 col vector index
        cell = comparison value for col vector indexed by row (matrix1) and col (matrix2)
    :return: min_perm: the permutation with the smallest sum,
             min_sum : the minimum sum of values for the min_perm
             no_perm_sum : the sum of values with no permutation
    """
    idx = range(cm.shape[0])
    min_perm = None
    min_sum = float('inf')
    no_perm_sum = sum([ cm[i, i] for i in idx ])
    for perm in itertools.permutations(idx):
        s = sum([ cm[i, p] for i, p in enumerate(perm) ])
        if s < min_sum:
            min_sum = s
            min_perm = perm
    return min_perm, min_sum, no_perm_sum


def find_max_match(cm):
    """
    Find the maximum sum of values
    (Identical to find_min_match except opposite done for the two lines with end-comments)
    :param cm:
    :return:
    """
    idx = range(cm.shape[0])
    max_perm = None
    max_sum = float('-inf')  #
    no_perm_sum = sum([ cm[i, i] for i in idx ])
    for perm in itertools.permutations(idx):
        s = sum([ cm[i, p] for i, p in enumerate(perm) ])
        if s > max_sum:      #
            max_sum = s
            max_perm = perm
    return max_perm, max_sum, no_perm_sum


# ---------------------------------------------------------------------

def get_flip_bits(fm, perm):
    """
    Collect the flipped bits for a given permutation
    I.e.,: for each row and corresponding column permutation index,
        look up the flip bit
    :param fm: matrix of flip bits; i is original order,
        j is permuted order, i,j entry is 0=not flipped, 1=flipped
    :param perm: permutation vector (j)
    :return: vector of bits in order of permutation indicating whether
        the column vector indexed by the permutation was flipped.
    """
    bits = list()
    for i, j in enumerate(perm):
        bits.append(fm[i][j])
    return bits


def test_get_flip_bits(verbose=False):
    fm = np.array([[0, 0, 1, 0, 0],
                   [1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0],
                   [1, 0, 0, 1, 1]])
    bits = get_flip_bits(fm, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(bits, np.array([0, 0, 0, 0, 1]))
    if verbose: print bits
    bits = get_flip_bits(fm, np.array([1, 0, 3, 4, 2]))
    assert np.array_equal(bits, np.array([0, 1, 0, 0, 0]))
    if verbose: print bits
    bits = get_flip_bits(fm, np.array([2, 3, 4, 1, 0]))
    assert np.array_equal(bits, np.array([1, 1, 0, 1, 1]))
    if verbose: print bits
    print 'TEST get_flip_bits() PASSED'

test_get_flip_bits()


def print_perms(obj):
    for p in itertools.permutations(obj):
        print p


# ---------------------------------------------------------------------

def find_best_match(m1, m2, comparison_fn, best_match_fn=find_min_match, verbose=False):
    cm, fm = make_comparison_matrix(m1, m2, comparison_fn=comparison_fn)

    if verbose:
        print 'find_best_match()'
        print '    comparison_fn: {0}'.format(comparison_fn.__name__)
        print '    best_match_fn: {0}'.format(best_match_fn.__name__)

        print 'comparison matrix:'
        print cm

        print 'flip bits:'
        print fm

    best_perm, best_sum, no_perm_sum = best_match_fn(cm)
    flip_bits = get_flip_bits(fm, best_perm)

    if verbose:
        print 'best_sum = {0}'.format(best_sum)
        print 'best matches:'
        for idx in range(len(best_perm)):
            print ' {0}'.format(idx),
        print '  m1'  # + states_file_1
        for idx in best_perm:
            print ' {0}'.format(idx),
        print '  m2'  # + states_file_2
        for b in flip_bits:
            print ' {0}'.format(int(b)),
        print '  flip bits'

    return best_perm, flip_bits, best_sum, no_perm_sum


def permute_columns(matrix, perm, flip_bits):
    new_matrix = np.zeros(matrix.shape)
    for i, ( p , f ) in enumerate(zip(perm, flip_bits)):
        if f == 0:
            new_matrix[:, i] = matrix[:, p]
        elif f == 1:
            new_matrix[:, i] = flip( matrix[:, p] )
        else:
            print 'ERROR: flip_bits was something unexpected!: {0}\n{1}'\
                .format(flip, flip_bits)
            sys.exit()
    return new_matrix


def test_permute_columns():
    m = np.array([[0, 1, 1],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 0, 0]])
    mp = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [0, 1, 1],
                   [0, 0, 1]])
    mpf = np.array([[1, 0, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    [0, 1, 0]])

    p = (1, 2, 0)
    m2 = permute_columns(m, p, [0, 0, 0])
    assert np.array_equal(m2, mp)
    m3 = permute_columns(m, p, [0, 1, 1])
    assert np.array_equal(m3, mpf)
    print 'TEST permute_columns() PASSED'

test_permute_columns()


def test_find_best_match(verbose=True):

    a =  np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    b =  np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    c =  np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    cn = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1])

    m1 = np.zeros((len(a), 3))   # matrix
    m1c = np.zeros((len(a), 3))  # copy of m1
    m2 = np.zeros((len(a), 3))   # different ordering of m1
    m3 = np.zeros((len(a), 3))   # same of m2, but flip first column
    m4 = np.zeros((len(a), 3))   # variant of m2 with flipped cn as first column

    for i, v in enumerate([a, b, c]):
        m1[:, i] = v
    for i, v in enumerate([a, b, c]):
        m1c[:, i] = v
    for i, v in enumerate([c, a, b]):
        m2[:, i] = v
    for i, v in enumerate([flip(c), a, b]):
        m3[:, i] = v
    for i, v in enumerate([flip(cn), a, b]):
        m4[:, i] = v

    # return m1, m1c, m2, m3

    if verbose: print '\n>>> m1 == m1c'
    min_perm, flip_bits, min_sum, no_perm_sum = \
        find_best_match(m1, m1c,
                        comparison_fn=min_hamming_distance,
                        best_match_fn=find_min_match,
                        verbose=verbose)
    assert np.array_equal(min_perm, np.array([0, 1, 2]))
    assert np.array_equal(flip_bits, np.array([0, 0, 0]))
    assert min_sum == 0.0

    if verbose: print '\n>>> m1 == m2'
    min_perm, flip_bits, min_sum, no_perm_sum = \
        find_best_match(m1, m2,
                        comparison_fn=min_hamming_distance,
                        best_match_fn=find_min_match,
                        verbose=verbose)
    assert np.array_equal(min_perm, np.array([1, 2, 0]))
    assert np.array_equal(flip_bits, np.array([0, 0, 0]))
    assert min_sum == 0.0

    if verbose: print '\n>>> m1 == m3'
    min_perm, flip_bits, min_sum, no_perm_sum = \
        find_best_match(m1, m3,
                        comparison_fn=min_hamming_distance,
                        best_match_fn=find_min_match,
                        verbose=verbose)
    assert np.array_equal(min_perm, np.array([1, 2, 0]))
    assert np.array_equal(flip_bits, np.array([0, 0, 1]))
    assert min_sum == 0.0

    if verbose: print '\n>>> m1 == m4'
    min_perm, flip_bits, min_sum, no_perm_sum = \
        find_best_match(m1, m4,
                        comparison_fn=min_hamming_distance,
                        best_match_fn=find_min_match,
                        verbose=verbose)
    assert np.array_equal(min_perm, np.array([1, 2, 0]))
    assert np.array_equal(flip_bits, np.array([0, 0, 1]))
    nose.tools.assert_almost_equal(min_sum, 0.117647058824)
    # assert (min_sum - 0.117647058824) < 0.000000001

    print 'TEST find_best_match() PASSED'

test_find_best_match(verbose=False)


# ---------------------------------------------------------------------

def compute_statistics(states_gt, states_inferred):
    """
    Calculate
        accuracy:  TP + TN / (TP + FP + TN + FN)
        precision: TP / (TP + FP)
        recall:    TP / (TP + FN)
        F1_score:  2 * (Precision * Recall) / (Precision + Recall)
    :param states_gt: NxM 2-d array
    :param states_inferred: NxM 2-d array
    :return:
    """
    def count_stats():
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(states_inferred.shape[0]):
            for j in range(states_inferred.shape[1]):
                if states_gt[i, j] == 1:
                    if states_inferred[i, j] == 1:
                        TP += 1  # 1 1
                    else:
                        FN += 1  # 1 0
                else:
                    if states_inferred[i, j] == 1:
                        FP += 1  # 0 1
                    else:
                        TN += 1  # 0 0
        return TP, FP, TN, FN

    TP, FP, TN, FN = count_stats()

    accuracy  = float(TP + TN) / float(TP + FP + TN + FN)

    precision = 0.0
    if float(TP + FP) > 0.0:
        precision = float(TP) / float(TP + FP)

    recall = 0.0
    if float(TP + FN) > 0.0:
        recall    = float(TP) / float(TP + FN)

    f1_score = 0.0
    if precision + recall > 0.0:
        f1_score  = 2.0 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


# ---------------------------------------------------------------------
# process_best_match_for_results
# ---------------------------------------------------------------------

def test_process_best_match():

    # ground_truth_columns
    c1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    c2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    c3 = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    c4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # flip of ground_truth
    f1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    f2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0])
    f3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    f4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # variation of ground_truth
    t1 = np.array([1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    t2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    t3 = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
    t4 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # flip of variation of ground_truth
    tf1 = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    tf2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    tf3 = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    tf4 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    ##

    # ground_truth
    ground_truth = np.zeros((len(c1), 4))
    for c, col in enumerate([c1, c2, c3, c4]):
        ground_truth[:, c] = col

    ##

    # Target as exact copy of ground_truth
    target_copy = np.copy(ground_truth)

    # permutation of target (copy of ground_truth)
    target_perm = np.zeros(ground_truth.shape)
    for c, col in enumerate([c4, c1, c2, c3]):
        target_perm[:, c] = col

    # flip of target (copy of ground_truth)
    target_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([f1, f2, f3, f4]):
        target_flip[:, c] = col

    # permutation and flip of target (copy of ground_truth)
    target_perm_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([f4, f1, f2, f3]):
        target_perm_flip[:, c] = col

    target_vary = np.zeros(ground_truth.shape)
    for c, col in enumerate([t1, t2, t3, t4]):
        target_vary[:, c] = col

    target_vary_perm = np.zeros(ground_truth.shape)
    for c, col in enumerate([t4, t1, t2, t3]):
        target_vary_perm[:, c] = col

    target_vary_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([tf1, tf2, tf3, tf4]):
        target_vary_flip[:, c] = col

    target_vary_perm_flip = np.zeros(ground_truth.shape)
    for c, col in enumerate([tf4, tf1, tf2, tf3]):
        target_vary_perm_flip[:, c] = col

    ##

    # variation of ground_truth

    for (comparison_fn_name, comparison_fn), (best_match_fn_name, best_match_fn) in \
            [#(('min_hamming_distance', min_hamming_distance),
             # ('find_min_match', find_min_match)),
             (('max_f1_score', max_f1_score),
              ('find_max_match', find_max_match)),
             ]:

        print
        print comparison_fn_name
        print best_match_fn_name

        for label, target in (('target_copy:    ***   ', target_copy),
                              ('target_perm:          ', target_perm),
                              ('target_flip:          ', target_flip),
                              ('target_perm_flip:     ', target_perm_flip),
                              ('target_vary:    ***   ', target_vary),
                              ('target_vary_perm:     ', target_vary_perm),
                              ('target_vary_flip:     ', target_vary_flip),
                              ('target_vary_perm_flip ', target_vary_perm_flip)):

            accuracy, precision, recall, f1_score \
                = compute_statistics(ground_truth, target)
            print 'ground_truth == {0}'.format(label), accuracy, precision, recall, f1_score

            best_perm, flip_bits, best_sum, no_perm_sum = \
                find_best_match(ground_truth,
                                target,
                                comparison_fn=comparison_fn,
                                best_match_fn=best_match_fn,
                                verbose=False)

            # permute AND flip states_thetastar to the best match permutation (min_perm)
            states_permuted = permute_columns(target, best_perm, flip_bits)

            accuracy, precision, recall, f1_score \
                = compute_statistics(ground_truth, states_permuted)
            print 'ground_truth == states_permuted:    > ', accuracy, precision, recall, f1_score

            """
            for c in range(ground_truth.shape[1]):
                print 'gt{0}: {1}'.format(c, ground_truth[:, c])
                print ' >{0}: {1}'.format(c, states_permuted[:, c])
            """

# test_process_best_match()


# ---------------------------------------------------------------------
# process_best_match_for_results
# ---------------------------------------------------------------------

def process_best_match_for_results \
                (results_path,
                 comparison_fn,
                 best_match_fn,
                 main_path='../',
                 test_p=True,
                 verbose=True):
    """

    :param results_path:
    :param comparison_fn: { min_hamming_distance , max_f1_score }
    :param best_match_fn: { find_min_match , find_max_match }
    :param main_path:
    :param test_p:
    :param verbose:
    :return:
    """

    # NOTE: Should probably start with test of whether this looks like
    #       a valid results directory, using same logic of util.is_results_dir()

    owd = os.getcwd()
    os.chdir(main_path)

    if test_p or verbose:
        print 'Executing process_results()'
        print '    \'{0}\''.format(results_path)
        print '    executing from directory \'{0}\''.format(os.getcwd())
        print '    comparison_fn: {0}'.format(comparison_fn.__name__)
        print '    best_match_fn: {0}'.format(best_match_fn.__name__)

    parameter_filename = 'parameters.config'

    pdict = util.read_parameter_file_as_dict(parameter_filepath=results_path,
                                             parameter_filename=parameter_filename)
    if ':experiment:data_path' in pdict:
        data_path = pdict[':experiment:data_path']
    else:
        print 'ERROR: parameter file at {0}{1}'.format(results_path, parameter_filename)
        print '       does not contain :experiment:data_path'
        sys.exit(-1)

    if test_p or verbose:
        print 'Found data_path: \'{0}\''.format(data_path)

    # read gold-standard figures states.txt
    data_states_path = os.path.join(data_path, 'states.txt')
    if verbose:
        print 'Reading states_data from \'{0}\''.format(data_states_path)
    states_data = read_binary_vector(data_states_path)
    if verbose:
        print '    Done.'

    # -------------------------------------------------------
    # Move original figures and create new directories and files

    def create_new_directory(new_directory_path, must_be_new_p=False, should_have_been_moved_to=None):
        if test_p:
            print 'Would create new \'{0}\''.format(new_directory_path)
        else:
            if os.path.exists(new_directory_path):
                if must_be_new_p:
                    print 'ERROR: should not exist yet: {0}'.format(new_directory_path)
                    sys.exit(-1)
                if should_have_been_moved_to:
                    print 'ERROR: should have been moved to \'{0}\''.format(should_have_been_moved_to)
                    print '       but still exists: {0}'.format(new_directory_path)
                    sys.exit(-1)
            else:
                if verbose:
                    print 'Creating new \'{0}\''.format(new_directory_path)
                os.makedirs(new_directory_path)
                if verbose:
                    print '    Done.'

    def move_from_src_to_dst(src, dst):
        if test_p:
            print 'Would move from: \'{0}\''.format(src)
            print '             to: \'{0}\''.format(dst)
        else:
            if verbose:
                print 'moving from: \'{0}\''.format(src)
                print '         to: \'{0}\''.format(dst)
            shutil.move(src, dst)
            if verbose:
                print '    Done.'

    # mkdir statistics_original
    statistics_original_dir = os.path.join(results_path, 'statistics_original')
    create_new_directory(statistics_original_dir, must_be_new_p=True)

    # mv accuracy.txt, precision.txt, recall.txt, F1_score.txt to statistics_original/
    stat_files = ['accuracy.txt', 'precision.txt', 'recall.txt', 'F1_score.txt']
    for stat_file in stat_files:
        src_path = os.path.join(results_path, stat_file)
        dst_path = os.path.join(statistics_original_dir, stat_file)
        move_from_src_to_dst(src_path, dst_path)

    # create new versions of statistics files with 'iteration value' as first line
    for stat_file in stat_files:
        new_stats_file_path = os.path.join(results_path, stat_file)
        if test_p:
            print 'Would make new \'{0}\''.format(new_stats_file_path)
        else:
            if verbose:
                print 'Creating new \'{0}\''.format(new_stats_file_path)
            if os.path.isfile(new_stats_file_path):
                print 'ERROR: file \'{0}\' already exists,'\
                    .format(new_stats_file_path)
                print '       should have been moved to \'statistics_original\''
                sys.exit(-1)
            with open(new_stats_file_path, 'w') as fp:
                fp.write('iteration value\n')
            if verbose:
                print '    Done.'

    thetastar_path = os.path.join(results_path, 'thetastar')

    # test that thetastar directory contains all valid states
    if test_p:
        print 'TESTING that thetastar directory contents contain'
        print '        valid binary matrices:\n>>\'{0}\''.format(thetastar_path)
        thetastar_file_list = util.read_directory_files(thetastar_path)
        i = 0
        for states_thetastar_filename in thetastar_file_list:
            states_thetastar_source_filepath = os.path.join(thetastar_path, states_thetastar_filename)
            if i % 10 == 0:
                print '    ',
            print '{0}'.format(states_thetastar_filename),
            if ((i+1) % 10) == 0:
                print
            # get thetastar file states
            read_binary_vector(states_thetastar_source_filepath,
                               fail_if_not_digit=True)
            # print 'DONE'
            i += 1
        print '\nAll thetastar contents appear to be valid binary matrices'

    thetastar_orig_path = os.path.join(results_path, 'thetastar_original')
    thetastar_match_specs_path = os.path.join(results_path, 'thetastar_match_specs')

    # mv thetastar directory to new thetastar_orig
    if os.path.exists(thetastar_path):
        move_from_src_to_dst(thetastar_path, thetastar_orig_path)
    else:
        print 'ERROR: missing \'{0}\''.format(thetastar_path)
        sys.exit(-1)

    # mkdir thetastar : create 'new' empty thetastar (now that original has been moved to thetastar_orig)
    create_new_directory(thetastar_path, should_have_been_moved_to='thetastar_original')

    # mkdir thetastar_match_specs
    create_new_directory(thetastar_match_specs_path, must_be_new_p=True)

    # get thetastar_orig directory listing for iterating
    if not test_p:
        if verbose:
            print 'Reading directory listing of \'{0}\''.format(thetastar_orig_path)
        thetastar_file_list = util.read_directory_files(thetastar_orig_path)
        if verbose:
            print '    Done.  ({0} files)'.format(len(thetastar_file_list))
    else:
        test_thetastar_file_list = util.read_directory_files(thetastar_path)
        print 'Would read {0} files from {1}'\
            .format(len(test_thetastar_file_list), thetastar_orig_path)
        print '    (based on current \'{0}\')'.format(thetastar_path)

    # iterate through each file in thetastar_orig
    if test_p:
        print 'Would iterate through each file in thetastar_orig, find best match, save stats'
    else:

        if verbose:
            print 'Iterating through each file in thetastar_orig'

        for states_thetastar_filename in thetastar_file_list:

            if verbose:
                print '    processing {0}'.format(states_thetastar_filename),

            states_thetastar_filename_base = states_thetastar_filename.split('.')[0]

            states_thetastar_source_filepath = os.path.join(thetastar_orig_path, states_thetastar_filename)

            # get thetastar_orig file states
            states_thetastar = read_binary_vector(states_thetastar_source_filepath)

            # find best match
            best_perm, flip_bits, best_sum, no_perm_sum = \
                find_best_match(states_data,
                                states_thetastar,
                                comparison_fn=comparison_fn,
                                best_match_fn=best_match_fn,
                                verbose=False)

            # permute AND flip states_thetastar to the best match permutation (min_perm)
            states_permuted = permute_columns(states_thetastar, best_perm, flip_bits)

            # save best match permutation to same thetastar/<filename>
            np.savetxt(os.path.join(thetastar_path, states_thetastar_filename), states_permuted, fmt='%d')

            # save best match permutation spec to thetastar/<filename>_match_spec.txt
            with open(os.path.join(thetastar_match_specs_path, states_thetastar_filename), 'w') as fout:
                for p in best_perm:
                    fout.write('{0} '.format(p))
                fout.write('\n')
                for b in flip_bits:
                    fout.write('{0} '.format(int(b)))
                fout.write('\n{0} {1}'.format(best_sum, no_perm_sum))

            # compute performance statistics and *append* to stats files
            # (with <filename> iteration, value)
            accuracy, precision, recall, f1_score \
                = compute_statistics(states_data, states_permuted)
            for stat_filename, statistic_value \
                    in zip(stat_files, [accuracy, precision, recall, f1_score]):
                # append the statistic_value to the stat file
                with open(os.path.join(results_path, stat_filename), 'a') as fout:
                    # format: '<states_thetastar_filename_base> <statistic_value>'
                    fout.write('{0} {1}\n'.format(states_thetastar_filename_base,
                                                  statistic_value))
            if verbose:
                print 'Done.'

    # save 'best_match_complete.txt' file with date-time.
    bmc_path = os.path.join(results_path, 'best_match_complete.txt')
    if test_p:
        print 'Would save \'{0}\''.format(bmc_path)
    else:
        if verbose:
            print 'Saving \'{0}\''.format(bmc_path)
        with open(bmc_path, 'w') as fout:
            fout.write('{0}'.format(util.get_timestamp()))
            # fout.write('{0}'.format(util.get_timestamp(verbose=True)))
        if verbose:
            print '    Done.'

    os.chdir(owd)

    if test_p or verbose:
        print 'DONE executing process_best_match_for_results(\'{0}\')'.format(results_path)


'''
process_best_match_for_results(results_path='results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0/LT_01/',
                               main_path='../',
                               test_p=False,
                               verbose=True)
'''


# ---------------------------------------------------------------------

def find_directory_level_if(directory_branch,
                            target_string='w1',
                            reject_list=('BestMatchHamming', 'BestMatchF1')):
    branch_parts = directory_branch.split('/')
    i = 0
    for bc in branch_parts:
        bc_parts = bc.split('_')
        if target_string in bc_parts:
            for reject_string in reject_list:
                if reject_string in bc_parts:
                    return None
            return i
        i += 1
    return None


def find_branches_in_list(branches_list,
                          target_string='w1',
                          reject_list=('BestMatchHamming', 'BestMatchF1')):
    found_branches = list()
    for branch in branches_list:
        # NOTE: not currently using w1_level information,
        # but could return it if desire to note where model
        # spec occurs along branch components
        level = find_directory_level_if(branch, target_string, reject_list)
        if level:
            found_branches.append(branch)
    return found_branches


def test_find_branches_in_list(verbose=False):

    if verbose:
        print 'Find w1, not { BestMatchHamming, BestMatchF1 }:'
    branches = find_branches_in_list \
        (['results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0/LT_01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1/LT_01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0_LT/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchNotReject/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchHamming/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchF1/01/'],
         target_string='w1',
         reject_list=('BestMatchHamming', 'BestMatchF1'))
    for branch, target \
            in zip(branches, ['results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1/LT_01/',
                              'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT/01/',
                              'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchNotReject/01/']):
        if verbose:
            print branch, target
        assert branch == target

    if verbose:
        print 'Find BestMatchHamming:'

    branches = find_branches_in_list \
        (['results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0/LT_01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1/LT_01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w0_LT/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchNotReject/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchHamming/01/',
          'results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchF1/01/'],
         target_string='BestMatchHamming',
         reject_list=list())

    for branch, target \
            in zip(branches, ['results/cocktail/h0.5_nocs_cp0/hmm_hdp_w1_LT_BestMatchHamming/01/']):
        if verbose:
            print branch, target
        assert branch == target

    print 'TEST find_branches_in_list() PASSED'

test_find_branches_in_list()


def copy_branch(src_path, dst_path, main_path='../', test_p=True, verbose=True):

    if not test_p:

        owd = os.getcwd()
        os.chdir(main_path)

        copy_tree(src_path, dst_path)

        os.chdir(owd)

        if verbose:
            print 'copy_tree from {0}'.format(src_path)
            print '            to {0}'.format(dst_path)
    else:
        if verbose:
            print 'would copy_tree from \'{0}\''.format(src_path)
            print '                  to \'{0}\''.format(dst_path)

    return dst_path


def copy_target_branches(results_root,
                         dst_model_postfix,
                         target_model_component='w1',
                         reject_list=('BestMatchHamming', 'BestMatchF1'),
                         main_path='../',
                         log_target_copy_paths_p=True,
                         test_p=True,
                         verbose=True):

    if log_target_copy_paths_p:
        log_file = 'copy_paths_' + util.get_timestamp() + '.log'

    # find all branches
    branches = util.get_directory_branches(results_root, main_path=main_path)

    # find all branches with specified models
    src_branches = find_branches_in_list \
        (branches_list=branches,
         target_string=target_model_component,
         reject_list=reject_list)

    src_dst_pairs = list()

    # create src,dst pairs and avoid duplicates
    for src_branch in src_branches:
        branch_components = src_branch.split('/')
        model_parts = branch_components[3].split('_')
        new_model_parts = model_parts + [dst_model_postfix]
        new_model_component = ['_'.join(new_model_parts)]

        src_path = '/'.join(branch_components[:-1])
        dst_path = '/'.join(branch_components[:-2] + new_model_component)

        pair = (src_path, dst_path)

        if pair not in src_dst_pairs:
            src_dst_pairs.append(pair)

    if log_target_copy_paths_p:
        with open(log_file, 'a') as fout:
            fout.write('{0}\n'.format(len(src_dst_pairs)))

    if verbose:
        print '{0}'.format(len(src_dst_pairs))

    target_result_paths = list()

    i = 0
    for src, dst in src_dst_pairs:

        if log_target_copy_paths_p:
            with open(log_file, 'a') as fout:
                fout.write('{0}'.format(i))

        if verbose:
            print '{0}'.format(i)

        target_result_path = copy_branch\
            (src,
             dst,
             main_path=main_path,
             test_p=test_p,
             verbose=verbose)

        if log_target_copy_paths_p:
            with open(log_file, 'a') as fout:
                fout.write(' {0}\n'.format(target_result_path))

        i += 1

        target_result_paths.append(target_result_path)

    if verbose:
        print
        for path in target_result_paths:
            print path

    return target_result_paths

'''
# laplace:
copy_target_branches(results_root='results/cocktail',
                     dst_model_postfix='BestMatchHamming',
                     target_model_component='w1',
                     reject_list=('BestMatchHamming', 'BestMatchF1'),
                     main_path='../',
                     log_target_copy_paths_p=True,
                     test_p=False,
                     verbose=True)
'''

'''
# laplace:
copy_target_branches(results_root='results/cocktail',
                     dst_model_postfix='BestMatchF1',
                     target_model_component='w1',
                     reject_list=('BestMatchHamming',),
                     main_path='../',
                     log_target_copy_paths_p=True,
                     test_p=False,
                     verbose=True)
'''

'''
# venti:
copy_target_branches(results_root='results/cocktail',
                     dst_model_postfix='BestMatchHamming',
                     target_model_component='w1',
                     reject_list=('BestMatchHamming', 'BestMatchF1'),
                     main_path='/projects/hamlet/figures',
                     log_target_copy_paths_p=True,
                     test_p=True,
                     verbose=True)
'''

'''
# venti:
copy_target_branches(results_root='results/cocktail',
                     dst_model_postfix='BestMatchF1',
                     target_model_component='w1',
                     reject_list=('BestMatchHamming',),
                     main_path='/projects/hamlet/figures',
                     log_target_copy_paths_p=True,
                     test_p=False,
                     verbose=True)
'''


# ---------------------------------------------------------------------

def process_best_matches\
                (results_root,
                 target_string='BestMatchHamming',
                 reject_list=('BestMatchF1',),
                 comparison_fn=min_hamming_distance,
                 best_match_fn=find_min_match,
                 main_path='../',
                 log_process_best_matches_p=True,
                 test_p=True,
                 verbose=True):
    """

    :param results_root:
    :param target_string: { 'BestMatchHamming' ,   'BestMatchF1' }
    :param reject_list:   { ('BestMatchF1',) ,     ('BestMatchHamming',) }  # not really needed
    :param comparison_fn: { min_hamming_distance , max_f1_score }
    :param best_match_fn: { find_min_match ,       find_max_match }
    :param main_path:
    :param test_p:
    :param verbose:
    :return:
    """

    branches = util.get_directory_branches(results_root, main_path=main_path)
    target_branches = find_branches_in_list\
        (branches, target_string=target_string, reject_list=reject_list)

    if log_process_best_matches_p:
        log_file = 'process_best_matches_{0}_'.format(target_string) + util.get_timestamp() + '.log'
        with open(log_file, 'a') as fout:
            fout.write('{0}\n'.format(len(target_branches)))

    # for path in target_branches:
    #     print path
    # sys.exit()

    i = 0
    for target_path in target_branches:

        if verbose:
            print '\n======================================================'
            print 'Best match for: \'{0}\''.format(target_path)

        if log_process_best_matches_p:
            with open(log_file, 'a') as fout:
                fout.write('{0} {1}\n'.format(i, target_path))

        # process best_match for target
        process_best_match_for_results\
            (target_path,
             comparison_fn=comparison_fn,
             best_match_fn=best_match_fn,
             main_path=main_path,
             test_p=test_p,
             verbose=verbose)

        i += 1

    print '\n======================================================'
    print '\nDONE'

# dst_model_postfix: { 'BestMatchHamming' , 'BestMatchF1' }
# comparison_fn: { min_hamming_distance , max_f1_score }
# best_match_fn: { find_min_match , find_max_match }


def process_best_matches_hamming(results_root='results/cocktail',
                                 main_path='../',
                                 log_process_best_matches_p=True,
                                 test_p=True,
                                 verbose=True):
    print '>>> Running process_best_matches_hamming()'
    process_best_matches(results_root=results_root,
                         target_string='BestMatchHamming',
                         reject_list=('BestMatchF1',),
                         comparison_fn=min_hamming_distance,
                         best_match_fn=find_min_match,
                         main_path=main_path,
                         log_process_best_matches_p=log_process_best_matches_p,
                         test_p=test_p,
                         verbose=verbose)

'''
# laplace:
process_best_matches_hamming(results_root='results/cocktail',
                             main_path='../',
                             log_process_best_matches_p=True,
                             test_p=True,
                             verbose=True)
'''

'''
# venti:
process_best_matches_hamming(results_root='results/cocktail',
                             main_path='/projects/hamlet/figures',
                             log_process_best_matches_p=True,
                             test_p=True,
                             verbose=True)
'''


def process_best_matches_f1(results_root='results/cocktail',
                            main_path='../',
                            log_process_best_matches_p=True,
                            test_p=True,
                            verbose=True):
    print 'Running process_best_matches_f1()'
    process_best_matches(results_root=results_root,
                         target_string='BestMatchF1',
                         reject_list=('BestMatchHamming',),
                         comparison_fn=max_f1_score,
                         best_match_fn=find_max_match,
                         main_path=main_path,
                         log_process_best_matches_p=log_process_best_matches_p,
                         test_p=test_p,
                         verbose=verbose)

'''
# laplace:
process_best_matches_f1(results_root='results/cocktail',
                        main_path='../',
                        log_process_best_matches_p=True,
                        test_p=True,
                        verbose=True)
'''

'''
# venti:
process_best_matches_f1(results_root='results/cocktail',
                        main_path='/projects/hamlet/figures',
                        log_process_best_matches_p=True,
                        test_p=True,
                        verbose=True)
'''


# ---------------------------------------------------------------------
# UBER SCRIPTS
# ---------------------------------------------------------------------


def script_laplace_copy_and_best_match_F1(test=True):
    print '\nLAPLACE'
    print '>>>>>> copy_target_branches -- F1'
    copy_target_branches(results_root='results/cocktail',
                         dst_model_postfix='BestMatchF1',
                         target_model_component='w1',
                         reject_list=('BestMatchHamming',),
                         main_path='../',
                         log_target_copy_paths_p=True,
                         test_p=test,
                         verbose=True)
    print '>>>>>> process_best_matches_F1'
    process_best_matches_f1(results_root='results/cocktail',
                            main_path='../',
                            log_process_best_matches_p=True,
                            test_p=test,
                            verbose=True)
    print '>>>>>> DONE'


'''
script_laplace_copy_and_best_match_F1(test=False)
'''

'''
# run best-match on version of pre-best-match copies from venti
process_best_matches_f1(results_root='results/cocktail_venti_orig_best_match_laplace',
                        main_path='../',
                        log_process_best_matches_p=True,
                        test_p=False,
                        verbose=True)
'''


def script_venti_copy_and_best_match_F1(test=True):
    print '\nVENTI'
    print '>>>>>> copy_target_branches -- F1'
    copy_target_branches(results_root='results/cocktail',
                         dst_model_postfix='BestMatchF1',
                         target_model_component='w1',
                         reject_list=('BestMatchHamming',),
                         main_path='/projects/hamlet/figures',
                         log_target_copy_paths_p=True,
                         test_p=test,
                         verbose=True)
    print '>>>>>> process_best_matches_F1'
    process_best_matches_f1(results_root='results/cocktail',
                            main_path='/projects/hamlet/figures',
                            log_process_best_matches_p=True,
                            test_p=test,
                            verbose=True)
    print '>>>>>> DONE'


def script_venti_copy_and_best_match_F1_LOCAL(test=True):
    print '\nVENTI'
    print '>>>>>> copy_target_branches -- F1'
    copy_target_branches(results_root='results/cocktail',
                         dst_model_postfix='BestMatchF1',
                         target_model_component='w1',
                         reject_list=('BestMatchHamming',),
                         main_path='../',
                         log_target_copy_paths_p=True,
                         test_p=test,
                         verbose=True)
    print '>>>>>> process_best_matches_F1'
    process_best_matches_f1(results_root='results/cocktail',
                            main_path='../',
                            log_process_best_matches_p=True,
                            test_p=test,
                            verbose=True)
    print '>>>>>> DONE'


'''
script_venti_copy_and_best_match_F1_LOCAL(test=True)
'''


def script_venti_copy_and_best_match_HAMMING_DATA(test=True):
    print '\nVENTI'
    print '>>>>>> copy_target_branches -- Hamming'
    copy_target_branches(results_root='results/cocktail',
                         dst_model_postfix='BestMatchHamming',
                         target_model_component='w1',
                         reject_list=('BestMatchF1',),
                         main_path='/projects/hamlet/figures',
                         log_target_copy_paths_p=True,
                         test_p=test,
                         verbose=True)
    print '>>>>>> process_best_matches_hamming'
    process_best_matches_hamming(results_root='results/cocktail',
                                 main_path='/projects/hamlet/figures',
                                 log_process_best_matches_p=True,
                                 test_p=test,
                                 verbose=True)
    print '>>>>>> DONE'

'''
script_venti_copy_and_best_match_HAMMING_DATA(test=True)
'''

# ---------------------------------------------------------------------


'''
def read_directory(dir_path):
    dir_contents = util.read_directory_files(dir_path)
    i = 0
    for item in dir_contents:
        print i, item
        i += 1

read_directory('../results/cocktail/h10.0_nocs_cp0/hmm_BFact_w1_fm0.3/01/thetastar')
'''
