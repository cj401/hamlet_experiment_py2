import os
import numpy
import matplotlib.pyplot as plt


DATA_ROOT = '../../../../../data'
DATA_gamma1_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_gamma1/A.txt')
DATA_gamma2_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_gamma2/A.txt')
DATA_gamma5_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_gamma5/A.txt')
DATA_gamma10_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_gamma10/A.txt')

DATA_gamma1_10x4_80pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_80percent_gamma1/A.txt')
DATA_gamma10_10x4_80pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_80percent_gamma10/A.txt')

DATA_gamma01_10x4_70pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_70percent_gamma0.1/A.txt')
DATA_gamma05_10x4_70pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_70percent_gamma0.5/A.txt')
DATA_gamma1_10x4_70pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_70percent_gamma1/A.txt')
DATA_gamma10_10x4_70pct_ROOT = os.path.join(DATA_ROOT, 'block_diag10x4_70percent_gamma10/A.txt')


def read_matrix(matrix_path):
    return numpy.loadtxt(matrix_path)


def plot_heatmap(matrix, title=None):

    y, x = numpy.mgrid[slice(0, matrix.shape[0]), slice(0, matrix.shape[1])]

    vmin = numpy.min(matrix)
    vmax = numpy.max(matrix)

    plt.figure()
    plt.pcolor(x, y, matrix, cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.title('Matrix Heatmap {0}\nvmin={1}, vmax={2}'.format(title, vmin, vmax))
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()


def read_and_plot_heatmap(matrix_path, title=None):
    matrix = read_matrix(matrix_path)
    plot_heatmap(matrix, title)

read_and_plot_heatmap(DATA_gamma1_10x4_80pct_ROOT, '10x4 80% g1')
read_and_plot_heatmap(DATA_gamma10_10x4_80pct_ROOT, '10x4 80% g10')
read_and_plot_heatmap(DATA_gamma01_10x4_70pct_ROOT, '10x4 70% g0.1')
read_and_plot_heatmap(DATA_gamma05_10x4_70pct_ROOT, '10x4 70% g0.5')
read_and_plot_heatmap(DATA_gamma1_10x4_70pct_ROOT, '10x4 70% g1')
read_and_plot_heatmap(DATA_gamma10_10x4_70pct_ROOT, '10x4 70% g10')
#read_and_plot_heatmap(DATA_gamma1_ROOT, 'gamma=1')
#read_and_plot_heatmap(DATA_gamma2_ROOT, 'gamma=2')
#read_and_plot_heatmap(DATA_gamma5_ROOT, 'gamma=5')
#read_and_plot_heatmap(DATA_gamma10_ROOT, 'gamma=10')

plt.show()


'''
def plot_heatmap2():
    # make these smaller to increase the resolution
    dx, dy = 0.15, 0.05

    # generate 2 2d grids for the x & y bounds
    y, x = numpy.mgrid[slice(-3, 3 + dy, dy), slice(-3, 3 + dx, dx)]
    z = (1 - x / 2. + x ** 5 + y ** 3) * numpy.exp(-x ** 2 - y ** 2)
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = -numpy.abs(z).max(), numpy.abs(z).max()

    plt.figure()
    plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.title('pcolor')
    # set the limits of the plot to the limits of the data
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()

    plt.show()
'''

