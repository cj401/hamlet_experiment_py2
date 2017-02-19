import numpy as np
import itertools
import scipy.spatial

__author__ = 'clayton'

"""
Block Matrix:
    T x (D x K)

"""

test_matrix_hypothesis = 'test_data_raw/categorical_best_match_matrices/00010.txt'
test_matrix_gt = 'test_data_raw/categorical_best_match_matrices/train.txt'


def datarray_to_3D(data, dim1):
    nr, nc = data.shape
    return data.reshape(nr, nc/dim1, dim1)


'''
def datarray_to_3D(figures, dim1):
    nr, nc = figures.shape
    figures = figures.reshape(nr, nc/dim1, dim1)
    return np.rollaxis(figures, 2, 1)
'''


def test_datarray_to_3D():
    m = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    m = datarray_to_3D(m, 3)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print '({0},{1}) {2}'.format(i, j, m[i][j])

# test_datarray_to_3D()


def l1_norm(m):
    m1 = m.reshape(-1)
    v = 0
    for i in range(m1.shape[0]):
        v += np.abs(m1[i])
    return v


def find_best_match(gt_path, hyp_path, D, K):
    """

    :param gt_path: string path to ground-truth matrix file
    :param hyp_path: string path to hypothesis matrix file
    :param D: integer number of observations
    :param K: integer dimension of observations
    :return: tuple: best-match permutation, best-match score
    """
    gt_matrix = datarray_to_3D(np.loadtxt(gt_path), K)
    hyp_matrix = datarray_to_3D(np.loadtxt(hyp_path), K)

    best_distance = np.infty
    best_perm = None

    for perm in itertools.permutations(range(D)):
        sdist = 0
        for r in range(gt_matrix.shape[0]):
            for c1, c2 in enumerate(perm):
                sdist += scipy.spatial.distance.euclidean(gt_matrix[r][c1], hyp_matrix[r][c2])
        if sdist < best_distance:
            best_distance = sdist
            best_perm = perm

    gt_l1_norm = l1_norm(gt_matrix)

    print 'gt_l1_norm', gt_l1_norm
    print 'best_distance', best_distance

    return best_distance / gt_l1_norm, best_perm


def test_find_best_match():
    best_distance, best_perm = \
        find_best_match(test_matrix_gt, test_matrix_hypothesis, 3, 3)
    print 'best_distance', best_distance
    print 'best_perm', best_perm

test_find_best_match()
