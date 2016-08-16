import os
import numpy
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
RESULTS_ROOT = os.path.join(HAMLET_ROOT, 'results')

LT_DATA_PATH = os.path.join(RESULTS_ROOT, 'music/music_chord1_10000iter/LT_hdp_hmm_w0')
LT_100K_DATA_PATH = os.path.join(RESULTS_ROOT, 'music/music_chord1_100000iter/LT_hdp_hmm_w0')

PATH_01_ROOT = os.path.join(LT_DATA_PATH, '01')
PATH_02_ROOT = os.path.join(LT_DATA_PATH, '02')


# ----------------------------------------------------------------------
# Read data
# ----------------------------------------------------------------------

def get_filename_last_iteration(path):
    maxint = 0
    filename_last_iteration = None
    for filename in os.listdir(path):
        # TODO: unsafe, add exception handling to gracefully exit if doesn't fit
        filename_int = int(filename.split('.')[0])
        if filename_int > maxint:
            maxint = filename_int
            filename_last_iteration = filename
    return filename_last_iteration


def read_data(results_root):
    # A: transition matrix; row_idx=current_state, col_idx=next_state
    path_A = os.path.join(results_root, 'A')
    filename_last_iter_A = get_filename_last_iteration(path_A)
    A = numpy.loadtxt(os.path.join(path_A, filename_last_iter_A))

    # X: rows=latent_states, cols=emission_states
    path_X = os.path.join(results_root, 'X')
    filename_last_iter_X = get_filename_last_iteration(path_X)
    X = numpy.loadtxt(os.path.join(path_X, filename_last_iter_X))

    # theta: latent state representation, in continuous space
    path_theta = os.path.join(results_root, 'theta')
    filename_last_iter_theta = get_filename_last_iteration(path_theta)
    theta = numpy.loadtxt(os.path.join(path_theta, filename_last_iter_theta))

    # pi0: initial latent state distribution
    path_pi0 = os.path.join(results_root, 'pi0')
    filename_last_iter_pi0 = get_filename_last_iteration(path_pi0)
    pi0 = numpy.loadtxt(os.path.join(path_pi0, filename_last_iter_pi0))

    return A, X, theta, pi0


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_heatmap(matrix, title=''):

    if len(matrix.shape) == 2:
        # need to add one more to dim in x and y
        # http://stackoverflow.com/questions/29016210/how-to-plot-matplotlib-pcolor-with-last-row-column-and-logarithmic-axis
        y, x = numpy.mgrid[slice(0, matrix.shape[0] + 1), slice(0, matrix.shape[1] + 1)]
    else:
        print 'ERROR plot_heatmap(): matrix not the right shape, needs to be 1 or 2 dim, instead {0}' \
            .format(matrix.shape)
        import sys
        sys.exit()

    vmin = numpy.min(matrix)
    vmax = numpy.max(matrix)

    plt.figure()

    # http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolor(x, y, matrix, cmap='Greys', vmin=vmin, vmax=vmax)  # cmap='RdBu'

    plt.title('Matrix Heatmap {0}\nvmin={1}, vmax={2}'.format(title, vmin, vmax))
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()


def plot_scatter(points, title=''):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.title('Scatterplot {0}'.format(title))
    plt.axis([points[:, 0].min() - 1, points[:, 0].max() + 1,
              points[:, 1].min() - 1, points[:, 1].max() + 1])


def plot_bar(data, title='', val_threshold=0.01):
    fig, ax = plt.subplots()
    width = 0.8
    idx = numpy.arange(len(data))
    rects = ax.bar(idx, data, width, color='b')
    ax.set_xticks(idx + width*0.5)
    ax.set_xticklabels(idx)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height >= val_threshold:
                ax.text(rect.get_x() + rect.get_width()/2., 1.05 * height,
                        '{0:.2f}'.format(height),
                        ha='center', va='bottom')

    autolabel(rects)
    plt.title('Bar {0}'.format(title))


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------

def plot_script(root, title=' '):
    A, X, theta, pi0 = read_data(root)
    plot_heatmap(A, title='A{0}{1}'.format(title, A.shape))
    plot_heatmap(X, title='X{0}{1}'.format(title, X.shape))
    # plot_scatter(theta, title='theta{}{0}'.format(title, theta.shape))
    # plot_bar(pi0, title='pi0{}'.format(title))

'''
plot_script(PATH_01_ROOT, ' 01 ')
plot_script(PATH_02_ROOT, ' 02 ')
plt.show()
'''


# ----------------------------------------------------------------------
# Collect multiple datasets
# ----------------------------------------------------------------------

def collect_multiple_datasets(path_base):
    firsttime = True
    subdirs = os.listdir(path_base)
    for i, subdir in enumerate(subdirs):
        path = os.path.join(path_base, subdir)
        A, X, theta, pi0 = read_data(path)

        if firsttime:
            firsttime = False
            A_all = numpy.zeros((A.shape[0], A.shape[1], len(subdirs)))
            X_all = numpy.zeros((X.shape[0], X.shape[1], len(subdirs)))
            theta_all = numpy.zeros((theta.shape[0], theta.shape[1], len(subdirs)))
            pi0_all = numpy.zeros((len(pi0), len(subdirs)))

        A_all[:, :, i] = A
        X_all[:, :, i] = X
        theta_all[:, :, i] = theta
        pi0_all[:, i] = pi0

    return A_all, X_all, theta_all, pi0_all


def multidataset_stats(path_base, show_p=False):
    A, X, theta, pi0 = collect_multiple_datasets(path_base)

    A_mu = numpy.mean(A, axis=2)
    X_mu = numpy.mean(X, axis=2)
    pi0_mu = numpy.mean(pi0, axis=1)

    A_std = numpy.std(A, axis=2)
    X_std = numpy.std(X, axis=2)
    pi0_std = numpy.std(pi0, axis=1)

    plot_heatmap(A_mu, title='A{0}{1}'.format(' mu ', A.shape))
    plt.savefig('A_mu.pdf', format='pdf')

    plot_heatmap(X_mu, title='X{0}{1}'.format(' mu ', X.shape))
    plt.savefig('X_mu.pdf', format='pdf')
    # plot_scatter(theta, title='theta{}{0}'.format(title, theta.shape))
    plot_bar(pi0_mu, title='pi0{0}'.format(' mu'))
    plt.savefig('pi0_mu.pdf', format='pdf')

    plot_heatmap(A_std, title='A{0}{1}'.format(' std ', A.shape))
    plt.savefig('A_std.pdf', format='pdf')
    plot_heatmap(X_std, title='X{0}{1}'.format(' std ', X.shape))
    plt.savefig('X_std.pdf', format='pdf')
    # plot_scatter(theta, title='theta{}{0}'.format(title, theta.shape))
    plot_bar(pi0_std, title='pi0{0}'.format(' std'))
    plt.savefig('pi0_std.pdf', format='pdf')

    if show_p:
        plt.show()


# multidataset_stats(LT_DATA_PATH)
multidataset_stats(LT_100K_DATA_PATH)
