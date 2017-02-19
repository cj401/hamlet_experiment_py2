import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as numpy

from utilities.binary_data_tools \
    import find_binary_segments, read_binary_vector_time_series_as_lol

__author__ = 'clayton'


# ---------------------------------------------------------------------
# plot_binary_vector_time_series
# ---------------------------------------------------------------------

def plot_binary_vector_time_series(data, figw=1000, figh=800, mydpi=96):

    fig = plt.figure(figsize=(figw/mydpi, figh/mydpi), dpi=mydpi)
    ax = plt.gca()

    level_max = data.shape[1]
    end_max = data.shape[0]

    for level in range(level_max):
        level_data = data[:, level]

        # find 0 and 1 chunk indices
        segments = find_binary_segments(level_data)
        for (start_idx, end_idx, bit) in segments:
            color = 'b' if bit == 1 else 'y'

            interval_xrange = [start_idx, end_idx]
            interval_yrange = numpy.array([level, level])
            ax.fill_between(interval_xrange, interval_yrange-0.2, interval_yrange+0.2,
                            facecolor=color, edgecolor=color, alpha=0.5)

    return fig, level_max, end_max

# plot_binary_vector_time_series('../figures/biology_papers/20150709_01/figures/p01/states.txt')
# plot_binary_vector_time_series('../figures/biology_papers/20150709_02/p01/states.txt')


# ---------------------------------------------------------------------
# plot_categorical_vector_time_series
# ---------------------------------------------------------------------

def plot_categorical_vector_time_series(data, figw=1000, figh=800, mydpi=96, colors=None):

    if colors is None:
        num_colors = len(numpy.unique(data)) - 1
        colors = cm.rainbow(numpy.linspace(0, 1, num_colors))
        colors = ['y'] + list(colors)
        # print 'colors', colors

    fig = plt.figure(figsize=(figw/mydpi, figh/mydpi), dpi=mydpi)

    level_max = data.shape[1]
    end_max = data.shape[0]

    for level in range(level_max):
        level_data = data[:, level]

        # find 0 and 1 chunk indices
        segments = find_binary_segments(level_data)
        for (start_idx, end_idx, bit) in segments:

            print 'bit', bit

            interval_xrange = [start_idx, end_idx]
            interval_yrange = numpy.array([level, level])
            ax = plt.gca()
            ax.fill_between(interval_xrange, interval_yrange-0.2, interval_yrange+0.2,
                            facecolor=colors[bit], edgecolor=colors[bit], alpha=0.5)

    return fig, level_max, end_max


# ---------------------------------------------------------------------
# plot_cp_data
# ---------------------------------------------------------------------

def plot_cp_data(path='../figures/cocktail_s16_m12/h2.0_nocs/cp0/states.txt', show_p=True):
    """
    Helper fn for plot_binary_vector_time_series for plotting cocktail party latent state figures
    :param path:
    :return:
    """
    fig, level_max, end_max = \
        plot_binary_vector_time_series(numpy.array(read_binary_vector_time_series_as_lol(path)))

    if show_p:
        plt.show()


# ---------------------------------------------------------------------
# SCRIPT
# ---------------------------------------------------------------------

# plot_cp_data(path='../figures/cocktail_s16_m12/h2.0_nocs/cp0/states.txt')
# plot_cp_data(path='../figures/cocktail_s14_m12/h2.0_nocs/cp0/states.txt')
