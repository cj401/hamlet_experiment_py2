import collections
import datetime
import glob
import math
import os

import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy
from matplotlib.widgets import Slider, SpanSelector

from scripts.python.experiments.REDD.preprocessing import dp_alpha

'''
Procedure for creating downsampled data:
(1) Used extract_median_downsampled_house_data_seconds_batch() which does the following:
    (a) Read original data (ignore mains)
    (b) Use median_filter_and_downsample_seconds_batch() to select 20-second intervals to median filter and downsample
    (c) Save each downsampled channel to
        <hamlet>/data/data/REDD/jw2013_downsampled/house_<#>/
    (d) Extract intervals and save
        <hamlet>/data/data/REDD/jw2013_downsampled_intervals/house_<#>_<start>_<end>/
            house_<#>_<start>_<end>.pdf  # plot of channels
            num_active_channels.txt      # integer rep. number of active channels
            obs.txt                      # observations
            sources/
                channel_<#>.txt
                ...
                labels.txt

'''


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

HAMLET_ROOT = '../../../../../'
REDD_DATA_ROOT = os.path.join(HAMLET_ROOT, 'data/data/REDD')
REDD_DATA_jw2013_ROOT = os.path.join(REDD_DATA_ROOT, 'johnson_willsky')
REDD_DATA_jw2013_extracted_ROOT = os.path.join(REDD_DATA_ROOT, 'jw2013_subset_extracted')
REDD_DATA_jw2013_downsampled_ROOT = os.path.join(REDD_DATA_ROOT, 'jw2013_downsampled')
REDD_DATA_jw2013_downsampled_intervals_ROOT = os.path.join(REDD_DATA_ROOT, 'jw2013_downsampled_intervals')

# NOTE: Use the following test current working directory and verify HAMLET_ROOT
# print os.getcwd()
# print os.listdir(REDD_DATA_ROOT)

## old paths...
# REDD_data_root = '../../figures/REDD/johnson_willsky'
# REDD_data_extracted_jw2013_root = '../../figures/REDD/extracted_jw2013'
# REDD_data_extracted_jw2013_downsampled_root = '../../figures/REDD/extracted_jw2013_downsampled'


# ------------------------------------------------------------
# REDD house data : jw2013
# ------------------------------------------------------------

'''
Houses 1, 2, 3, 6
refrigerator, lighting , dishwasher, microwave, furnace
Number represent 'channel_{0}.dat'
'''

devices = ('refrigerator', 'lighting', 'dishwasher', 'microwave', 'furnace')

house_1 = {
    'house_num': '1',
    'refrigerator': 5,
    'lighting': (17, 18),
    'dishwasher': 6,
    'microwave': 11,
    'furnace': 13  # electric_heat
}

house_2 = {
    'house_num': '2',
    'refrigerator': 9,
    'lighting': 4,
    'dishwasher': 10,
    'microwave': 6,
    'furnace': None
}

house_3 = {
    'house_num': '3',
    'refrigerator': 7,
    'lighting': (5, 11, 15, 17, 19),
    'dishwasher': 9,
    'microwave': 16,
    'furnace': 10
}

house_6 = {
    'house_num': '6',
    'refrigerator': 8,
    'lighting': 14,
    'dishwasher': 9,
    'microwave': None,
    'furnace': 12  # electric_heat
}


# ------------------------------------------------------------
# Read data
# ------------------------------------------------------------

def read_data(data_root, house_num, appliance_num, verbose=False):

    if verbose:
        print 'Reading house_{0} : channel_{1}.dat'.format(house_num, appliance_num)

    house_dir = os.path.join(data_root, 'house_{0}'.format(house_num))
    channel_path = os.path.join(house_dir, 'channel_{0}.dat'.format(appliance_num))
    data = numpy.loadtxt(channel_path)
    return data


def load_data_channel_batch(data_root, house_num, channels,
                            return_dict=False, verbose=False):

    print '>>> load_data_channel_batch(return_dict={})'.format(return_dict)

    if verbose:
        print 'load_all_data(): {0}, house_{1}, channels={2}'\
            .format(data_root, house_num, channels)

    house_path = os.path.join(data_root, 'house_{0}'.format(house_num))

    if return_dict:
        data = dict()
    else:
        data = list()

    for channel_num in channels:
        path = os.path.join(house_path, 'channel_{0}.dat'.format(channel_num))
        datum = numpy.loadtxt(path)

        if verbose:
            print '    channel {0}, datum.shape={1}'\
                .format(channel_num, datum.shape)

        if return_dict:
            data[channel_num] = datum
        else:
            data.append(datum)

    return data

# load_data_channel_batch(REDD_data_root, 1, channels=(3, 4, 7, 8, 15, 16, 19, 20, 9, 17, 18), verbose=True)


# ------------------------------------------------------------
# Median Filtering
# ------------------------------------------------------------

def median_filter(data, w_limit_indices, verbose=False):
    """
    NOTE: Assumes data is Nx2 array, where index 0 is timestamps, index 1 is
    Applies median filter with fixed +/- w_limit figures points
    :param data:
    :param w_limit_indices: number of indices before and after current to include in window
    :param verbose:
    :return:
    """

    if verbose:
        print 'median filter w_limit={0}, figures.shape={1}'.format(w_limit_indices, data.shape)

    newdata = numpy.copy(data[w_limit_indices:-w_limit_indices])
    r = (2 * w_limit_indices) + 1
    for i in range(newdata.shape[0]):
        newdata[i, 1] = numpy.median(data[i:(i + r), 1])
    return newdata


def median_filter_seconds_precise(data, w_limit_seconds, verbose=False):
    """
    NOTE: Assumes data is Nx2 array, where index 0 is timestamps, index 1 is
    Applies median filter with +/- window, in seconds
    Recomputes at each step lower and upper window bound to ensure w_limit in seconds.
    BUT, doesn't seem much more expensive than median_filter
    :param data:
    :param w_limit_seconds: seconds before and after current to include in window
    :param verbose:
    :return:
    """

    if verbose:
        print 'median filter seconds_precise w_limit={0} sec, figures.shape={1}'\
            .format(w_limit_seconds, data.shape)

    newdata = numpy.copy(data)
    ws = 0
    wst = data[0, 0]
    we = 0
    wet = data[0, 0]
    data_end = data.shape[0] - 1
    for i, (tstamp, cv) in enumerate(data):
        # advance ws to lower window boundary
        while datetime.datetime.utcfromtimestamp(tstamp - wst).second > w_limit_seconds \
                and ws <= i:
            ws += 1
            wst = data[ws, 0]
        # advance we to upper window boundary
        while datetime.datetime.utcfromtimestamp(wet - tstamp).second <= w_limit_seconds \
                and we < data_end:
            we += 1
            wet = data[we, 0]
        newdata[i, 1] = numpy.median(data[ws:we, 1])
    return newdata


def median_filter_and_downsample(data, step, verbose=False):
    """
    Downsample data, summarizing each interval of size step by median of values in interval
    :param data:
    :param step: size of downsample factor (in terms of indices)
    :param verbose:
    :return:
    """

    if verbose:
        print 'median_filter_and_downsample(): steps={0}, figures.shape={1}'.format(step, data.shape)

    s = 0
    e = step
    data_end = data.shape[0]
    samples = list()
    while e < data_end:
        idx = data[int(s + (math.floor(e - s) / 2)), 0]
        val = numpy.median(data[s:e, 1])
        samples.append((idx, val))
        s = e
        e += step
    return numpy.array(samples)


def median_filter_and_downsample_seconds(data, seconds, verbose=False, test=False):
    """
    Downsample data, where interval is determined by number of seconds
    :param data:
    :param seconds:
    :param verbose:
    :param test:
    :return:
    """

    if verbose:
        print 'median_filter_and_downsample(): seconds={0}, figures.shape={1}'.format(seconds, data.shape)

    s = 0
    e = 0
    data_end = data.shape[0]
    samples = list()
    stop = False

    if test:
        time_diffs = list()
        step_diffs = list()
        badness = list()

    while not stop:

        new_steps = 0
        stop_e_increment = False
        while not stop_e_increment:
            if e+1 >= data_end:
                stop_e_increment = True
                stop = True
                e = data_end - 1
            elif data[e+1, 0] - data[s, 0] <= seconds:
                e += 1
                new_steps += 1
            else:
                if new_steps == 0:
                    e += 1
                stop_e_increment = True

        if test:
            if e - s > 1 and data[e, 0] - data[s, 0] > seconds:
                badness.append((e-s, data[e, 0] - data[s, 0], s, data[s, 0], e, data[e, 0],))
            time_diffs.append(data[e, 0] - data[s, 0])
            step_diffs.append(e - s)

        idx = data[int(s + math.floor((e - s) / 2)), 0]
        val = numpy.median(data[s:e, 1])
        samples.append((idx, val))
        s = e

    if test:
        print 'time diffs', collections.Counter(time_diffs)
        print 'step diffs', collections.Counter(step_diffs)
        print 'badness   ', badness

    return numpy.array(samples)


def test_median_filter_and_downsample_seconds(house_num, channel_num):
    path = os.path.join(REDD_DATA_jw2013_ROOT, 'house_{0}/channel_{1}.dat'
                        .format(house_num, channel_num))
    data = numpy.loadtxt(path)
    median_filter_and_downsample_seconds(data, 20, verbose=True, test=True)

# test_median_filter_and_downsample_seconds(1, 1)


# ------------------------------------------------------------

def median_filter_and_downsample_seconds_batch\
                (data_store, seconds, return_dict=False, verbose=False, test=False):
    """
    Version of median_filter_and_downsample_second performed simultaneously over a list of data
    Reason: Found some evidence that not all channels had the same timestamps, so when
     process separately, ended up with different-lengths
     This method uses the first data stream in the data_list as the reference and
     identifies all other intervals for median-filter downsampling based on it.
    NOTE: The correct way would be to have some method to align each data stream's timestamps
     and then identify corresponding intervals for each.
    NOTE: using return_dict assumes that data_store is also a dictionary
    :param data_store: list or dictionary of data (sequence of sequences)
    :param seconds:
    :param return_dict:
    :param verbose:
    :param test:
    :return:
    """

    if verbose:
        print 'median_filter_and_downsample_seconds_batch():'
        if return_dict:
            print '    DICTIONARY'
            array_size = data_store[sorted(data_store.keys)[0]].shape
        else:
            print '    LIST'
            array_size = data_store[0].shape
        print '    num_arrays={0}, seconds={1}, figures.shape={2}'\
            .format(len(data_store), seconds, array_size)

    if return_dict:
        data_list = list()
        for idx in sorted(data_store.keys()):
            data_list.append(data_store[idx])
    else:
        data_list = data_store

    reference_datum = data_list[0]

    s = 0
    e = 0
    data_end = reference_datum.shape[0]

    sampled_data_list = [list() for i in range(len(data_list))]

    stop = False

    if test:
        time_diffs = list()
        step_diffs = list()
        badness = list()

    ### debug
    #j = 0

    while not stop:

        new_steps = 0
        stop_e_increment = False

        # find the next end point...
        while not stop_e_increment:
            if e+1 >= data_end:
                stop_e_increment = True
                stop = True
                e = data_end - 1
            elif reference_datum[e+1, 0] - reference_datum[s, 0] <= seconds:
                e += 1
                new_steps += 1
            else:
                if new_steps == 0:
                    e += 1
                stop_e_increment = True

        if test:
            if e - s > 1 and reference_datum[e, 0] - reference_datum[s, 0] > seconds:
                badness.append((e - s, reference_datum[e, 0] - reference_datum[s, 0],
                                s, reference_datum[s, 0], e, reference_datum[e, 0],))
            time_diffs.append(reference_datum[e, 0] - reference_datum[s, 0])
            step_diffs.append(e - s)

        idx = reference_datum[int(s + math.floor((e - s) / 2)), 0]

        ### debug
        #if j % 100 == 0:
        #    print j, idx,

        for i, datum in enumerate(data_list):
            val = numpy.median(datum[s:e, 1])
            sampled_data_list[i].append((idx, val))

            ### debug
            #if j % 100 == 0:
            #    print val,

        ### debug
        #if j % 100 == 0:
        #    print

        ### debug
        #j += 1

        s = e

    if test:
        print 'time diffs', collections.Counter(time_diffs)
        print 'step diffs', collections.Counter(step_diffs)
        print 'badness   ', badness

    if return_dict:
        data_array_store = dict()
        for sampled_data, data_store_idx in zip(sampled_data_list, data_store):
            data_array_store[data_store_idx] = numpy.array(sampled_data)
    else:
        data_array_store = list()
        for sampled_data in sampled_data_list:
            data_array_store.append(numpy.array(sampled_data))

    return data_array_store


'''
# Original version as of 20160723, prior to adding return_dict option
def median_filter_and_downsample_seconds_batch\
                (data_list, seconds, verbose=False, test=False):
    """
    Version of median_filter_and_downsample_second performed simultaneously over a list of data
    Reason: Found some evidence that not all channels had the same timestamps, so when
     process separately, ended up with different-lengths
     This method uses the first data stream in the data_list as the reference and
     identifies all other intervals for median-filter downsampling based on it.
    NOTE: The correct way would be to have some method to align each data stream's timestamps
     and then identify corresponding intervals for each.
    :param data_list: list of data (sequence of sequences)
    :param seconds:
    :param return_dict:
    :param verbose:
    :param test:
    :return:
    """

    if verbose:
        print 'median_filter_and_downsample_seconds_batch():'
        print '    datums={0}, seconds={1}, figures.shape={2}'\
            .format(len(data_list), seconds, data_list[0].shape)

    reference_datum = data_list[0]

    s = 0
    e = 0
    data_end = reference_datum.shape[0]

    sampled_data_list = [list() for i in range(len(data_list))]

    stop = False

    if test:
        time_diffs = list()
        step_diffs = list()
        badness = list()

    ### debug
    #j = 0

    while not stop:

        new_steps = 0
        stop_e_increment = False

        # find the next end point...
        while not stop_e_increment:
            if e+1 >= data_end:
                stop_e_increment = True
                stop = True
                e = data_end - 1
            elif reference_datum[e+1, 0] - reference_datum[s, 0] <= seconds:
                e += 1
                new_steps += 1
            else:
                if new_steps == 0:
                    e += 1
                stop_e_increment = True

        if test:
            if e - s > 1 and reference_datum[e, 0] - reference_datum[s, 0] > seconds:
                badness.append((e - s, reference_datum[e, 0] - reference_datum[s, 0],
                                s, reference_datum[s, 0], e, reference_datum[e, 0],))
            time_diffs.append(reference_datum[e, 0] - reference_datum[s, 0])
            step_diffs.append(e - s)

        idx = reference_datum[int(s + math.floor((e - s) / 2)), 0]

        ### debug
        #if j % 100 == 0:
        #    print j, idx,

        for i, datum in enumerate(data_list):
            val = numpy.median(datum[s:e, 1])
            sampled_data_list[i].append((idx, val))

            ### debug
            #if j % 100 == 0:
            #    print val,

        ### debug
        #if j % 100 == 0:
        #    print

        ### debug
        #j += 1

        s = e

    if test:
        print 'time diffs', collections.Counter(time_diffs)
        print 'step diffs', collections.Counter(step_diffs)
        print 'badness   ', badness

    data_array_list = list()
    for sampled_data in sampled_data_list:
        data_array_list.append(numpy.array(sampled_data))

    return data_array_list
'''


# ------------------------------------------------------------
# Aggregating channels
# ------------------------------------------------------------

def aggregate_data_sum(data_store, data_store_is_dict=False, verbose=False):
    """
    Aggregates channels (each array in data_list is a channel)
    by summing within indices
    :param data_store:
    :param data_store_is_dict:
    :param verbose:
    :return: array
    """
    if verbose:
        print 'Aggregating by summing {0} channels'.format(len(data_store))
        if data_store_is_dict:
            print '    data_store is DICTIONARY'

    if data_store_is_dict:
        new_data = numpy.copy(data_store[data_store.keys()[0]])
        if len(data_store) > 1:
            for i in range(new_data.shape[0]):
                for idx in data_store.keys():
                    if data_store[idx] is not None:
                        new_data[i, 1] += data_store[idx][i, 1]
    else:
        new_data = numpy.copy(data_store[0])
        if len(data_store) > 1:
            for i in range(new_data.shape[0]):
                for datum in data_store:
                    if datum is not None:
                        new_data[i, 1] += datum[i, 1]

    return new_data


# ------------------------------------------------------------
# Find hour boundaries (visualization)
# ------------------------------------------------------------

def find_midnight_pts(data):
    midnight_pts = list()
    within_zero_hour = False
    for i, tstamp in enumerate(data):
        tstamp = int(tstamp)
        if datetime.datetime.utcfromtimestamp(tstamp).hour == 0:
            if within_zero_hour is False:
                midnight_pts.append(i)
                within_zero_hour = True
        else:
            if within_zero_hour is True:
                within_zero_hour = False
    return midnight_pts


def find_hour_pts(data, s, e):
    within_hour = datetime.datetime.utcfromtimestamp(int(data[0])).hour
    hour_pts = list()
    hour_pts.append((0, within_hour))
    for i, tstamp in enumerate(data[s:e]):
        tstamp = int(tstamp)
        current_hour = datetime.datetime.utcfromtimestamp(tstamp).hour
        if current_hour is not within_hour:
            within_hour = current_hour
            hour_pts.append((i + s, within_hour))
    return hour_pts


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def plot_data_value_histogram(data, ax, color='b', num_bins=100):
    # plt.figure()
    x_max = numpy.max(data)
    print 'x_max', x_max
    bins = numpy.linspace(0, x_max, num_bins)
    n, bins, patches = ax.hist(data, bins, histtype='bar', facecolor=color, alpha=0.5)  # normed=0,
    # ax.setp(patches, 'facecolor', color, 'alpha', 0.5)
    # plt.ylim((0, 100000))

    return x_max  # , n.max()


# ------------------------------------------------------------

def plot_data_values(data, s=None, e=None):
    fig = plt.figure()

    if s is None:
        s = 0
    if e is None:
        e = data.shape[0]

    plt.plot(range(s, e), data[s:e])
    max_y = numpy.max(data[s:e])
    # midnight_pts = find_midnight_pts(figures[s:e, :])
    hour_pts = find_hour_pts(data, s, e)
    # print hour_pts
    for (xp, hour) in hour_pts:
        if hour is 0:
            plt.plot((xp, xp), (0, max_y), 'r--')
        else:
            plt.plot((xp, xp), (0, max_y), 'y--')
    # plt.ylim((0, 1000))
    plt.xlim(s, e)

    return fig, s, e, max_y


# ------------------------------------------------------------

def plot_hour_markers(fig_spec, timestamps, s=None, e=None, new_fig=False):

    ax, xmin, xmax, ymax = fig_spec

    if new_fig:
        plt.figure()

    if s is None:
        s = 0
    if e is None:
        e = timestamps.shape[0]

    # midnight_pts = find_midnight_pts(figures[s:e, :])
    hour_pts = find_hour_pts(timestamps, s, e)
    # print hour_pts
    for (xp, hour) in hour_pts:
        if hour is 0:
            ax.plot((xp, xp), (0, ymax), 'r--')
        else:
            ax.plot((xp, xp), (0, ymax), 'y--')

    return ax, s, e, ymax


# ------------------------------------------------------------

def plot_data_values_list_overlay(data_list, s=None, e=None):
    fig = plt.figure()
    max_y = 0

    if s is None:
        s = 0
    if e is None:
        e = data_list[0].shape[0]

    for data in data_list:
        plt.plot(range(s, e), data[s:e])
        max_y1 = numpy.max(data[s:e])
        if max_y1 > max_y:
            max_y = max_y1

    # plt.ylim((0, 1000))
    plt.xlim(s, e)

    return fig, s, e, max_y


# ------------------------------------------------------------


def plot_data_values_dict_stacked(data_dict, labels_dict, channel_indices,
                                  s=None,
                                  e=None,
                                  data_dim=0,
                                  hour_marker_timestamps=None,
                                  title=None):
    """

    :param data_dict:
    :param labels_dict:
    :param channel_indices:
    :param hour_marker_timestamps:
    :param title:
    :return:
    """
    font = {'family': 'serif',
            'color': 'darkblue',  # 'darkred'
            'weight': 'normal',
            'size': 10,
            }

    if 'all_summed' in data_dict:
        num_channels = len(channel_indices) + 1
    else:
        num_channels = len(channel_indices)

    colors = cm.rainbow(numpy.linspace(0, 1, num_channels))
    fig, axarr = plt.subplots(num_channels, sharex=True)

    # dynamically resize depending on size of
    if num_channels - 6 > 0:
        height = num_channels
        # print '>>>>>>>>>> HEIGHT', height
        fig.set_size_inches(8.0, height, forward=True)

    max_y = 0

    if s is None:
        s = 0
    if e is None:
        e = data_dict[data_dict.keys()[0]].shape[0]

    if title:
        axarr[0].set_title(title)

    if 'all_summed' in data_dict:
        channel_indices = ['all_summed'] + list(channel_indices)

    for i, idx in enumerate(channel_indices):
        if data_dim > 0:
            data = data_dict[idx][:, data_dim]
        else:
            data = data_dict[idx]
        axarr[i].plot(range(s, e), data[s:e], color=colors[i])
        axarr[i].set_ylabel('Power\n(Watts)', fontdict=font)

        for tick in axarr[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(7)

        max_y = numpy.max(data[s:e])

        '''
        if max_y1 > max_y:
            max_y = max_y1
        '''

        if hour_marker_timestamps is not None:
            # print hour_marker_timestamps
            plot_hour_markers((axarr[i], s, e, max_y), hour_marker_timestamps)

        # axarr[i].set_ylim((0, max_y1))

        axarr[i].text(.5, .75, '{0}'.format(labels_dict[idx]),
                      horizontalalignment='center',
                      transform=axarr[i].transAxes,
                      fontdict=font)

    # plt.ylim((0, 1000))
    plt.xlim(s, e)

    return fig, s, e, max_y


def plot_data_values_list_stacked(data_list, s=None, e=None, labels=None,
                                  hour_marker_timestamps=None,
                                  title=None):
    font = {'family': 'serif',
            'color': 'darkblue',  # 'darkred'
            'weight': 'normal',
            'size': 10,
            }

    num_channels = len(data_list)

    colors = cm.rainbow(numpy.linspace(0, 1, num_channels))
    fig, axarr = plt.subplots(num_channels, sharex=True)

    # dynamically resize depending on size of
    if num_channels - 6 > 0:
        height = num_channels
        # print '>>>>>>>>>> HEIGHT', height
        fig.set_size_inches(8.0, height, forward=True)

    # figure(figsize=(cm2inch(12.8), cm2inch(9.6)))
    max_y = 0

    if s is None:
        s = 0
    if e is None:
        e = data_list[0].shape[0]

    if title:
        axarr[0].set_title(title)

    for i, data in enumerate(data_list):
        axarr[i].plot(range(s, e), data[s:e], color=colors[i])
        axarr[i].set_ylabel('Power\n(Watts)', fontdict=font)

        for tick in axarr[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(7)

        max_y = numpy.max(data[s:e])

        '''
        if max_y1 > max_y:
            max_y = max_y1
        '''

        if hour_marker_timestamps is not None:
            # print hour_marker_timestamps
            plot_hour_markers((axarr[i], s, e, max_y), hour_marker_timestamps)

        # axarr[i].set_ylim((0, max_y1))

        if labels and i < len(labels):
            axarr[i].text(.5, .75, '{0}'.format(labels[i]),
                          horizontalalignment='center',
                          transform=axarr[i].transAxes,
                          fontdict=font)
            # axarr[i].set_title('{0}'.format(labels[i]), fontdict=font)

    # plt.ylim((0, 1000))
    plt.xlim(s, e)

    return fig, s, e, max_y


# ------------------------------------------------------------

def plot_data_values_list_histograms_stacked\
                (data_list, channel_labels, title, clip=0, max_y=None,
                 verbose=False):
    """
    TODO: This fn is a bit convoluted; it's also very similar
    to plot_data_values_list_stacked(), perhaps some generalization
    is in order.
    :param data_list:
    :param channel_labels:
    :param title:
    :param verbose:
    :return:
    """

    font = {'family': 'serif',
            'color': 'darkblue',  # 'darkred'
            'weight': 'normal',
            'size': 10,
            }

    colors = cm.rainbow(numpy.linspace(0, 1, len(data_list)))

    xlim_max = 0  # should get filled in by call to get_plot

    print len(data_list)

    fig, axarr = plt.subplots(len(data_list), sharex=True)

    # dynamically resize depending on size of
    num_channels = len(data_list)
    if num_channels - 6 > 0:
        height = num_channels
        # print '>>>>>>>>>> HEIGHT', height
        fig.set_size_inches(8.0, height, forward=True)

    # figure(figsize=(cm2inch(12.8), cm2inch(9.6)))

    axarr[0].set_title(title)

    for i, data in enumerate(data_list):

        # construct histogram
        # get x_max from histogram and update xlim_max
        clipped_data = sorted(data[:, 1])
        if clip is not None and clip > 0:
            clipped_data = clipped_data[:-clip]
        print '>>>', numpy.max(data[:, 1]), numpy.max(clipped_data)
        x_max = plot_data_value_histogram(clipped_data, axarr[i], color=colors[i])

        if xlim_max < x_max:
            xlim_max = x_max

        axarr[i].set_ylabel('Freq', fontdict=font)

        mu = numpy.mean(clipped_data)
        sigma = numpy.std(clipped_data)

        axarr[i].text(.5, .75,
                      '{0}; $\mu={1}, \sigma={2}$'\
                          .format(channel_labels[i], mu, sigma),
                      horizontalalignment='center',
                      transform=axarr[i].transAxes,
                      fontdict=font)

        if max_y is not None:
            axarr[i].set_ylim(0, max_y)
        else:
            max_y = numpy.max(data)

        axarr[i].add_line(lines.Line2D((x_max, x_max), (0, max_y),
                                       linewidth=1, linestyle=':',  # ':'
                                       color='m'))  # alpha=0.5

    plt.xlim(0, xlim_max)

    return fig, 0, xlim_max, max_y


# ------------------------------------------------------------

def plot_browser(fig_spec, ymin=0, s=0, x_display_max=None, show_p=False):

    fig, xmin, xmax, ymax = fig_spec

    ax = plt.gca()

    if x_display_max is None:
        x_display_max = xmax

    slider_color = 'lightgoldenrodyellow'
    slider_note_ax = plt.axes([0.175, 0.04, 0.65, 0.02], axisbg=slider_color)
    slider_note_pos = Slider(slider_note_ax, 'Position', 0, xmax, valinit=s)

    # quarter rest unicode: u'\U0001D13D'  # doesn't display, missing font?

    slider_zoom_ax = plt.axes([0.175, 0.018, 0.65, 0.02], axisbg=slider_color)
    slider_zoom_pos = Slider(slider_zoom_ax, 'Zoom', xmin, xmax, valinit=x_display_max - s)

    def update_pos_zoom_slider(val):
        pos = slider_note_pos.val
        zoom = slider_zoom_pos.val
        ax.axis([pos, pos + zoom, ymin, ymax])
        fig.canvas.draw_idle()

    slider_note_pos.on_changed(update_pos_zoom_slider)

    '''
    def update_zoom_slider(val):
        pos = slider_note_pos.val
        zoom = slider_zoom_pos.val
        ax.axis([pos, pos+zoom, ymin, ymax])
        fig.canvas.draw_idle()
    '''

    slider_zoom_pos.on_changed(update_pos_zoom_slider)

    def onselect(xmin_sel, xmax_sel):
        indmin, indmax = numpy.searchsorted(range(xmin, xmax), (xmin_sel, xmax_sel))
        # indmax = min(len(x) - 1, indmax)
        print '[{0}, {1}] = {2}'.format(indmin, indmax, indmax - indmin)
        fig.canvas.draw()

    span = SpanSelector(ax, onselect, 'horizontal',
                        useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

    # call once manually to get zoom to update...
    update_pos_zoom_slider(None)

    if show_p:
        plt.show()


# ------------------------------------------------------------
# Extract house data
# ------------------------------------------------------------

def extract_house_data_jw2013(source_dir, dest_dir,
                              house_params=(house_1, house_2, house_3, house_6),
                              devices=('refrigerator', 'lighting', 'dishwasher', 'microwave', 'furnace'),
                              w_limit=10, verbose=True):
    """
    Extract house data from original REDD data
    Approximates Johnson & Willsky 2013 data
    :param source_dir: Directory from which to read original REDD data
    :param dest_dir: Directory to which to save extracted house data
    :param house_params: sequence of dictionaries, each specifying per house: house_num and device channel number(s)
    :param devices: sequence of device names to extract
    :param w_limit: (default 10) one-sided window size, in seconds, to include in median filter
    :param verbose:
    :return: None
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for house_spec in house_params:
        house_num = house_spec['house_num']
        if verbose:
            print '-------------------------------------'
            print 'processing house_{0}'.format(house_num)
        channel_data_list = list()
        for device in devices:
            if verbose:
                print 'processing house_{0} {1}'.format(house_num, device)
            # channel is file number of figures stream; channel num is same as device num
            channel = house_spec[device]
            if channel is None:
                print 'No figures to read for device {0}'.format(device)
                channel_data_list.append(None)
            else:
                if type(channel) is tuple:
                    if verbose:
                        print 'Combining {0} channels for device {1}'.format(len(channel), device)
                    ch_data_mfiltered_list = list()
                    for ch in channel:
                        ch_data = read_data(source_dir, house_num, ch, verbose=verbose)
                        ch_data_mfiltered = median_filter_seconds_precise(ch_data, w_limit, verbose=verbose)
                        ch_data_mfiltered_list.append(ch_data_mfiltered)
                    channel_data_list.append(aggregate_data_sum(ch_data_mfiltered_list, verbose=verbose))
                else:
                    ch_data = read_data(source_dir, house_num, channel, verbose=verbose)
                    channel_data_list.append(median_filter_seconds_precise(ch_data, w_limit, verbose=verbose))

        channel_sum = aggregate_data_sum(channel_data_list, verbose=verbose)

        house_dir = os.path.join(dest_dir, 'house_{0}'.format(house_num))
        if not os.path.exists(house_dir):
            os.makedirs(house_dir)

        obs_path = os.path.join(house_dir, 'all_summed.txt')
        numpy.savetxt(obs_path, channel_sum[:, 1])

        path = os.path.join(house_dir, 'timestamps.txt')
        numpy.savetxt(path, channel_sum[:, 0], fmt='%d')

        for device, channel_data in zip(devices, channel_data_list):
            if channel_data is None:
                print 'No figures to save for device {0}'.format(device)
            else:
                path = os.path.join(house_dir, '{0}.txt'.format(device))
                numpy.savetxt(path, channel_data[:, 1])

        if verbose:
            print 'Done processing house_{0}'.format(house_num)

    if verbose:
        print 'DONE'


# ------------------------------------------------------------

def extract_median_downsampled_house_data(source_dir, dest_dir,
                                          house_nums=(1, 2, 3, 6), step=20, verbose=False):
    """
    Extract house data from original REDD data
    Downsamples *all* channels INDIVIDUALLY,
    Intervals defined by step: number of indices
    :param source_dir: Directory from which to read original REDD data
    :param dest_dir: Directory to which to save extracted house data
    :param house_nums:
    :param step: Number of indices per interval used for median filter downsample
    :param verbose:
    :return: None
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for house_num in house_nums:
        if verbose:
            print '-------------------------------------'
            print 'processing house_{0}'.format(house_num)
        house_path = os.path.join(source_dir, 'house_{0}'.format(house_num))
        house_dir_glob = glob.glob(os.path.join(house_path, '*'))

        house_dest_dir = os.path.join(dest_dir, 'house_{0}'.format(house_num))
        if not os.path.exists(house_dest_dir):
            os.makedirs(house_dest_dir)

        for channel_path in house_dir_glob:
            bname = os.path.basename(channel_path)
            if len(bname) > 8 and bname[0:8] == 'channel_':
                if verbose: print 'Reading', channel_path
                channel_data = numpy.loadtxt(channel_path)

                channel_data_mf = median_filter_and_downsample_seconds(channel_data, step,
                                                                       verbose=verbose,
                                                                       test=True)

                channel_data_dest_path = os.path.join(house_dest_dir, bname)
                if verbose:
                    print 'Downsampled channel_data', channel_data_mf.shape
                    print 'Saving', channel_data_dest_path
                numpy.savetxt(channel_data_dest_path, channel_data_mf, fmt='%d %1.2f')

'''
extract_median_downsampled_house_data(source_dir=REDD_data_root,
                                      dest_dir=REDD_data_extracted_jw2013_downsampled_ROOT,
                                      house_nums=(2, 3, 6),
                                      step=20,
                                      verbose=True)
'''


# ------------------------------------------------------------

def extract_median_downsampled_house_data_seconds_batch(source_dir, dest_dir,
                                                        house_nums=((1, (1, 2)),
                                                                    (2, (1, 2)),
                                                                    (3, (1, 2)),
                                                                    (6, (1, 2))),
                                                        seconds=20,
                                                        verbose=True, test=True):
    """
    Extract house data from original REDD data
    house_nums specifies each house_num followed by list of channels to IGNORE (i.e., mains)
    Downsamples *all* channels TOGETHER using median_filter_and_downsample_seconds_batch()
    Intervals defined by seconds
    :param source_dir:
    :param dest_dir:
    :param house_nums: tuple: (<house_num>, (<ignore-list>...))
    :param seconds: Number of seconds in interval
    :param verbose:
    :param test:
    :return:
    """

    def get_channel_num(channel_filename):
        base_name = channel_filename.split('/')[-1]
        if len(base_name) >= 7 and base_name[:7] == 'channel':
            return base_name.split('_')[1].split('.')[0]
        else:
            return None

    if verbose:
        print 'Running extract_median_downsampled_house_data_seconds_batch()'

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for house_num, ignore_channels in house_nums:
        if verbose:
            print '-------------------------------------'
            print 'processing house_{0} (ignoring channels {1})'\
                .format(house_num, ignore_channels)
        house_source_path = os.path.join(source_dir, 'house_{0}'.format(house_num))
        house_source_dir_glob = glob.glob(os.path.join(house_source_path, '*'))

        channels = list()
        for elm in house_source_dir_glob:
            channel_num = get_channel_num(elm)
            if channel_num is not None and int(channel_num) not in ignore_channels:
                channels.append(channel_num)

        if verbose:
            print '>>> Loading all data'
        data_list = load_data_channel_batch(source_dir, house_num,
                                            channels=channels,
                                            verbose=verbose)

        if verbose:
            print '>>> Calling median_filter_and_downsample_seconds_batch()'
        data_list_mf = median_filter_and_downsample_seconds_batch(data_list, seconds,
                                                                  verbose=verbose, test=test)

        house_dest_dir = os.path.join(dest_dir, 'house_{0}'.format(house_num))
        if not os.path.exists(house_dest_dir):
            if verbose:
                print '>>> Creating house_dest_dir: {0}'.format(house_dest_dir)
            os.makedirs(house_dest_dir)

        if verbose:
            print '>>> Saving channels to {0}'.format(house_dest_dir)
        for channel, channel_data_mf in zip(channels, data_list_mf):
            channel_dest_path = os.path.join(house_dest_dir, 'channel_{0}.dat'.format(channel))
            if verbose:
                print '    Downsampled channel_{0}, shape: {1}'.format(channel, channel_data_mf.shape)
                print '    Saving {0}'.format(channel_dest_path)
            print channel_dest_path
            numpy.savetxt(channel_dest_path, channel_data_mf, fmt='%d %1.2f')

    print 'DONE'

'''
extract_median_downsampled_house_data_seconds_batch(REDD_DATA_jw2013_ROOT,
                                                    REDD_DATA_jw2013_downsampled_ROOT,
                                                    house_nums=((1, (1, 2)),
                                                                (2, (1, 2)),
                                                                (3, (1, 2)),
                                                                (6, (1, 2))),
                                                    seconds=20,
                                                    verbose=True, test=True)
'''


# ------------------------------------------------------------
# Read already-extracted house data
# NOTE: "already-extracted" here means data stored in
#          REDD_DATA_jw2013_extracted_ROOT and
#          REDD_DATA_jw2013_downsampled_ROOT
#          -- Use: read_extracted_house_data() and
#                  read_extracted_house_data_by_channel()
#       Contrast with reading from INTERVAL-extracted data
#          (which is generally house data that was then processed
#           from REDD_DATA_jw2013_extracted_ROOT and saved in
#           REDD_DATA_jw2013_downsampled_intervals_ROOT)
#          For this, use: read_extracted_interval_house_data()
# ------------------------------------------------------------

def read_extracted_house_data(root_dir, house, devices, verbose=False):

    if verbose:
        print 'Reading extracted house figures'

    data = list()
    labels = list()

    base_path = os.path.join(root_dir, 'house_{0}'.format(house['house_num']))

    path = os.path.join(base_path, 'timestamps.txt')
    if verbose:
        print 'Reading {0}'.format(path)
    data.append(numpy.loadtxt(path))

    path = os.path.join(base_path, 'all_summed.txt')
    if verbose:
        print 'Reading {0}'.format(path)
    data.append(numpy.loadtxt(path))
    labels.append('all_summed')

    for device in devices:
        if house[device] is not None:
            path = os.path.join(base_path, '{0}.txt'.format(device))
            if verbose:
                print 'Reading {0}'.format(path)
            data.append(numpy.loadtxt(path))
            labels.append(device)

    return data, labels


# ------------------------------------------------------------

def read_extracted_house_data_by_channel(root_dir, house_num, channels, verbose=False):
    """
    Read specified channels of extracted house data; each channel's data is labeled
    by the channel filename.
    NOTE: read_extracted_house_data_by_channel_batch(), below, allows specifying name of channel
    :param root_dir:
    :param house_num:
    :param channels:
    :param verbose:
    :return: list of data: [<aggregated_data>, ...], list of labels: ['all_summed', ...]
    """
    if verbose:
        print 'read_extracted_house_data_by_channel(): house_{0}, channels {1}'.format(house_num, channels)

    data = list()
    labels = list()

    base_path = os.path.join(root_dir, 'house_{0}'.format(house_num))

    for channel in channels:
        channel_filename = 'channel_{0}.dat'.format(channel)
        path = os.path.join(base_path, channel_filename)
        if verbose:
            print 'Loading {0}'.format(path)
        datum = numpy.loadtxt(path)
        print 'datum.shape', datum.shape
        data.append(datum)
        labels.append(channel_filename)

    data_agg = aggregate_data_sum(data, verbose=verbose)

    return [data_agg] + data, ['all_summed'] + labels


'''
data_list, ls = read_extracted_house_data_by_channel(REDD_data_extracted_jw2013_downsampled_ROOT,
                                                     house_num=1,
                                                     channels=(3, 4, 7, 8, 15, 16, 19, 20, 9, 17, 18),
                                                     verbose=True)

plot_browser(fig_spec=plot_data_values_list_stacked(data_list[1:],
                                                    labels=ls,
                                                    hour_marker_timestamps=data_list[0]))
'''


# ------------------------------------------------------------

def read_extracted_interval_house_data(source_root):
    """
    Reads channel_source_index.txt, obs.txt ('all_summed'), amplitudes.txt (all remaining)
    NOTE: generally source_root is likely under REDD_DATA_jw2013_downsampled_intervals_ROOT
    :param source_root:
    :return: channel_data_dict, channel_labels_dict
    """

    interval_start = None
    interval_end = None

    path_interval = os.path.join(source_root, 'interval.txt')
    with open(path_interval) as fin:
        line = fin.readlines()
        if len(line) != 1:
            print 'ERROR: read_extracted_interval_house_data():'
            print '       Unexpected interval.txt content:'
            print '{0}'.format(line)
        values = line[0].split()
        interval_start = int(values[0])
        interval_end = int(values[1])

    active_channels = list()
    channel_labels_dict = dict()

    channel_labels_dict['all_summed'] = 'all_summed'
    path_channel_source_index = os.path.join(source_root, 'channel_source_index.txt')
    with open(path_channel_source_index, 'r') as fin:
        for line in fin.readlines():
            values = line.split()
            channel_index = int(values[0])
            active_channels.append(channel_index)
            channel_labels_dict[channel_index] = line.strip()

    channel_data_dict = dict()

    path_obs = os.path.join(source_root, 'obs.txt')
    obs_array = numpy.loadtxt(path_obs)
    channel_data_dict['all_summed'] = obs_array

    path_amplitudes = os.path.join(source_root, 'amplitudes.txt')
    amplitudes = numpy.loadtxt(path_amplitudes)
    for i, channel_index in enumerate(active_channels):
        channel_data_dict[channel_index] = amplitudes[:, i]

    interval_spec = ((interval_start, interval_end), active_channels)

    return channel_data_dict, channel_labels_dict, interval_spec


def test_read_extracted_interval_house_data():
    channel_data_dict, channel_labels_dict, interval_spec = \
        read_extracted_interval_house_data\
            (os.path.join(REDD_DATA_jw2013_downsampled_intervals_ROOT,
                          'house_1_1200_6800'))

    print 'interval_spec:'
    print interval_spec

    (interval_start, interval_end), active_channels = interval_spec

    print '\nchannel_data_dict:'
    print 'all_summed:', channel_data_dict['all_summed']
    for channel_index in active_channels:
        print '{0} :'.format(channel_index), channel_data_dict[channel_index]
    print channel_data_dict

    print '\nchannel_labels_dict:'
    print 'all_summed:', channel_labels_dict['all_summed']
    for channel_index in active_channels:
        print '{0} :'.format(channel_index), channel_labels_dict[channel_index]

# test_read_extracted_interval_house_data()


# ------------------------------------------------------------
#
# ------------------------------------------------------------

def read_extracted_house_data_by_channel_batch \
                (data_root, house_num, channel_tuples,
                 do_median_filter_and_downsample_p=False,
                 return_dict=False,
                 verbose=False):
    """
    Read specified channels for extracted house data.
    :param data_root:
    :param house_num:
    :param channel_tuples: sequence of tuples: (<channel_number>, <channel_label>)
    :param do_median_filter_and_downsample_p: (default: False)
                If true, compute median filter downsample on read data;
                Default assumes already filtered and downsampled.
    :param return_dict: (default False) when true, returns dicts for
                all_data: key=<channel_num>, value=<channel_data>
                channel_labels: key=<channel_num>, value=<channel_label>
    :param verbose:
    :return:
    """

    print '>>> read_extracted_house_data_by_channel_batch(return_dict={0})'.format(return_dict)

    channels = [channel_num for channel_num, _ in channel_tuples]
    if return_dict:
        channel_labels = dict()
        for channel_num, label in channel_tuples:
            channel_labels[channel_num] = '{0} {1}'.format(channel_num, label)
    else:
        channel_labels = ['{0} {1}'.format(channel_num, label)
                          for channel_num, label in channel_tuples]

    data_store = load_data_channel_batch(data_root, house_num,
                                         channels=channels,
                                         return_dict=return_dict,
                                         verbose=verbose)
    if verbose:
        print 'len(data_store)', len(data_store),
        if return_dict:
            print ', data_store[sorted(data_store.keys())[0]].shape {0}'\
                .format(data_store[sorted(data_store.keys())[0]].shape)
        else:

            print ', data_store[0].shape', data_store[0].shape
    if do_median_filter_and_downsample_p:
        # TODO median_filter_and_downsample_seconds_batch also needs dict option
        data_store_mf = median_filter_and_downsample_seconds_batch \
            (data_store, 20, return_dict=return_dict, verbose=verbose, test=True)
    else:
        if verbose:
            print 'SKIPPING do_median_filter_and_downsample_p()'
        data_store_mf = data_store
    if verbose:
        print 'len(data_list_mf)', len(data_store_mf),
        if return_dict:
            print ', data_store_mf[sorted(data_store_mf.keys())[0]].shape {0}' \
                .format(data_store_mf[sorted(data_store_mf.keys())[0]].shape)
        else:
            print ', data_store_mf[0].shape', data_store_mf[0].shape

    data_agg = aggregate_data_sum\
        (data_store_mf, data_store_is_dict=return_dict, verbose=True)

    if return_dict:
        all_data = data_store_mf
        all_data['all_summed'] = data_agg
        all_labels = channel_labels
        all_labels['all_summed'] = 'all_summed'
    else:
        all_data = [data_agg] + data_store_mf
        all_labels = ['all_summed'] + channel_labels

    return all_data, all_labels


# ------------------------------------------------------------

def plot_extracted_house_data_histogram_batch(data_root,
                                              data_spec,
                                              clip=None,
                                              max_y=None,
                                              save_plot=False,
                                              save_path=None,
                                              show_plot=False,
                                              verbose=False
                                              ):

    # data_spec format:
    # ((<house_num>, (<channel_num>, <channel_label>)*)* )

    channel_data_all = list()
    channel_labels_all = list()

    for house_num, channel_tuples in data_spec:
        if channel_tuples is not None:
            channel_data, channel_labels = read_extracted_house_data_by_channel_batch \
                (data_root, house_num, channel_tuples, verbose=verbose)
            channel_labels_house_num = list()
            for label in channel_labels[1:]:
                channel_labels_house_num.append('{0}_{1}'.format(house_num, label))
            channel_data_all += channel_data[1:]
            channel_labels_all += channel_labels_house_num

    fig, xlim_min, xlim_max, max_y = plot_data_values_list_histograms_stacked \
        (data_list=channel_data_all, channel_labels=channel_labels_all,
         clip=clip,
         max_y=max_y,
         title='Channel Histograms')

    if show_plot:
        plt.show()


def plot_extracted_house_data_histogram(data_root, house_num, channel_tuples,
                                        clip=None,
                                        max_y=None,
                                        save_plot=False,
                                        save_path=None,
                                        show_plot=False,
                                        verbose=False):

    channel_data, channel_labels = read_extracted_house_data_by_channel_batch\
        (data_root, house_num, channel_tuples, verbose=verbose)

    fig, xlim_min, xlim_max, max_y = plot_data_values_list_histograms_stacked\
        (data_list=channel_data[1:], channel_labels=channel_labels[1:],
         clip=clip,
         max_y=max_y,
         title='House {0} channel histograms'.format(house_num))

    # plot_browser(fig_spec=(fig, xlim_min, xlim_max, max_y),
    #              s=0, x_display_max=xlim_max, show_p=False)

    if save_plot is not False and save_path is not None:
        path = os.path.join(save_path, 'house_{0}_histograms.pdf'.format(house_num))
        if verbose:
            print 'saving plot {0} ...'.format(path),
        plt.savefig(path, format='pdf')
        if verbose:
            print 'DONE.'

    if show_plot:
        plt.show()


# ------------------------------------------------------------


def plot_extracted_house_data(house_num, channel_data, channel_labels,
                              interval_spec=None,
                              using_interval=False,
                              data_dim=0,
                              save_plot=False,
                              save_path=None,
                              save_name_postfix=None,
                              show_plot=False,
                              verbose=False):
    """

    :param house_num:
    :param channel_data:
    :param channel_labels:
    :param interval_spec: (default None) optional interval tuple:
                          ((<start_index>, <end_index>), <active_channels>)
    :param save_plot:
    :param save_path:
    :param save_name_postfix:
    :param show_plot: (default False) Show (plt.show()) created plot when True
    :param verbose:
    :return:
    """

    def get_plot(channel_indices, title=None):
        return plot_data_values_dict_stacked \
            (data_dict=channel_data,
             labels_dict=channel_labels,
             channel_indices=channel_indices,
             data_dim=data_dim,
             hour_marker_timestamps=None,  # channel_data[0][:, 0],
             title=title)

    (interval_start, interval_end), active_channels = interval_spec
    # interval_start, interval_end = interval

    if verbose:
        print 'plotting interval [{0}, {1}] ...'.format(interval_start, interval_end),

    if using_interval:
        s = interval_start
        x_display_max = interval_end
    else:
        s = 0
        x_display_max = channel_data[channel_data.keys()[0]].shape[0]

    plot_browser(fig_spec=get_plot(channel_indices=active_channels,
                                   title='House {0} [{1}, {2}]'\
                                   .format(house_num, interval_start, interval_end)),
                 s=s, x_display_max=x_display_max, show_p=False)

    if verbose:
        print 'DONE.'

    if save_plot is not False and save_path is not None:
        if save_name_postfix is not None:
            filename = 'house_{0}_{1}-{2}_{3}.pdf'\
                .format(house_num, interval_start, interval_end, save_name_postfix)
        else:
            filename = 'house_{0}_{1}-{2}.pdf'.format(house_num, interval_start, interval_end)
        path = os.path.join(save_path, filename)
        if verbose:
            print 'saving plot {0} ...'.format(path),
        plt.savefig(path, format='pdf')
        if verbose:
            print 'DONE.'

    if show_plot:
        plt.show()


'''
def plot_extracted_house_data(house_num, channel_data, channel_labels,
                              interval_spec=None,
                              save_plot=False,
                              save_path=None,
                              show_plot=False,
                              verbose=False):
    """

    :param house_num:
    :param channel_data:
    :param channel_labels:
    :param interval_spec: (default None) optional interval tuple:
                          ((<start_index>, <end_index>), <active_channels>)
    :param save_plot:
    :param save_path:
    :param show_plot: (default False) Show (plt.show()) created plot when True
    :param verbose:
    :return:
    """

    def get_plot(title=None):
        return plot_data_values_list_stacked \
            ([d[:, 1] for d in channel_data],
             labels=channel_labels,
             hour_marker_timestamps=channel_data[0][:, 0],
             title=title)

    if interval_spec:
        (interval_start, interval_end), active_channels = interval_spec
        # interval_start, interval_end = interval
        if verbose:
            print 'plotting interval [{0}, {1}] ...'.format(interval_start, interval_end),
        plot_browser(fig_spec=get_plot(title='House {0} [{1}, {2}]'\
                                       .format(house_num, interval_start, interval_end)),
                     s=interval_start, x_display_max=interval_end, show_p=False)
        if verbose:
            print 'DONE.'

        if save_plot is not False and save_path is not None:
            path = os.path.join(save_path,
                                'house_{0}_{1}-{2}.pdf'\
                                .format(house_num, interval_start, interval_end))
            if verbose:
                print 'saving plot {0} ...'.format(path),
            plt.savefig(path, format='pdf')
            if verbose:
                print 'DONE.'

    else:
        plot_browser(fig_spec=get_plot(), show_p=False)

    if show_plot:
        plt.show()
'''


# ------------------------------------------------------------

def get_priors(priors, query, n=5000):
    """

    :param priors:
    :param query: <string> device type
    :param n: <integer> number of observations in sample
    :return:
    """
    for keys, mu, sigma, d in priors:
        if query in keys:
            alpha = dp_alpha.compute_alpha(d=d, n=n, alpha0=0.5)
            return mu, sigma, alpha
    print 'WARNING: get_priors(): Prior not found for query \'{0}\''.format(query)
    return None, None, None


def save_extracted_intervals(save_path, channel_source, channel_labels,
                             interval_spec, priors, verbose=False):

    # create save_path from dest_root and house_num
    # <dest_root>/house_<house_num>_<s>_<e>/
    #     sources/
    #         channel_<num>.txt
    #         ...
    #         labels.txt
    #     obs.txt   # aggregated (summed) data: 'all_summed'

    (interval_start, interval_end), active_channels = interval_spec

    channel_data = dict()
    for channel_idx in active_channels:
        channel_data[channel_idx] = channel_source[channel_idx]

    if verbose:
        print '---------- Saving {0}'.format(save_path)
        print 'len(channel_data)', len(channel_data)
        print 'channel_data[0][:,1] (all_summed): {0}'\
            .format(channel_source['all_summed'][interval_start:interval_end,1])
        print 'channel_labels', channel_labels
        print 'interval_spec:', interval_spec

    path_interval = os.path.join(save_path, 'interval.txt')
    with open(path_interval, 'w') as fout:
        fout.write('{0} {1}'.format(interval_start, interval_end))

    # save obs.txt == 'all_summed' == channel_data[0]
    obs_path = os.path.join(save_path, 'obs.txt')
    if verbose:
        print 'Saving all_summed to obs.txt [{0}, {1}]: {2}'\
            .format(interval_start, interval_end, obs_path)
    observations = channel_source['all_summed'][interval_start:interval_end, 1]
    numpy.savetxt(obs_path, observations, fmt='%1.2f')

    num_observations = observations.shape[0]

    path_amplitudes = \
        os.path.join(save_path, 'amplitudes.txt')

    amplitudes = numpy.zeros((interval_end - interval_start, len(active_channels)))
    for i, channel_index in enumerate(active_channels):
        amplitudes[:, i] = channel_data[channel_index][interval_start:interval_end, 1]
    with open(path_amplitudes, 'w') as fout:
        numpy.savetxt(fout, amplitudes, fmt='%1.2f')

    path_channel_source_index = \
        os.path.join(save_path, 'channel_source_index.txt')
    path_categorical_state_model_alpha = \
        os.path.join(save_path, 'categorical_state_model_alpha.txt')
    path_independent_normal_weights_mu = \
        os.path.join(save_path, 'independent_normal_weights_mu.txt')
    path_independent_normal_weights_sigma = \
        os.path.join(save_path, 'independent_normal_weights_sigma.txt')

    for channel_index in active_channels:
        with open(path_channel_source_index, 'a') as fout:
            fout.write('{0}\n'.format(channel_labels[channel_index]))

        _, channel_label = channel_labels[channel_index].split()
        # NOTE: The following version computes alpha based on FIXED n=5000
        mu, sigma, alpha = get_priors(priors, channel_label, n=5000)
        # NOTE: The following version computes alpha based on num_observations
        # mu, sigma, alpha = get_priors(priors, channel_label, num_observations)

        with open(path_categorical_state_model_alpha, 'a') as fout:
            fout.write('{0}\n'.format(alpha))
        with open(path_independent_normal_weights_mu, 'a') as fout:
            fout.write('{0}\n'.format(mu))
        with open(path_independent_normal_weights_sigma, 'a') as fout:
            fout.write('{0}\n'.format(sigma))

    '''
    # create sources/
    sources_path = os.path.join(save_path, 'sources')
    if not os.path.exists(sources_path):
        if verbose:
            print '>>> Creating sources_path: {0}'.format(sources_path)
        os.makedirs(sources_path)

    # save labels.txt
    labels_path = os.path.join(sources_path, 'labels.txt')
    with open(labels_path, 'w') as fout:
        for channel, channel_label_str in zip(channel_data[1:], channel_labels[1:]):
            num_str, label = channel_label_str.split()

            if verbose:
                print '    Saving channel_{0}.txt == {1}'.format(num_str, label)

            fout.write('{0} {1} {2}\n'.format(num_str, 'channel_{0}.txt'.format(num_str), label))

            # save channel data
            channel_path = os.path.join(sources_path, 'channel_{0}.txt'.format(num_str))
            numpy.savetxt(channel_path, channel[interval[0]:interval[1], 1], fmt='%1.2f')
    '''

    if verbose:
        print 'DONE.'


# ------------------------------------------------------------

def read_plot_save_intervals(data_root, house_num, channel_tuples,
                             do_median_filter_and_downsample_p=False,
                             intervals=None,
                             priors=None,
                             dest_root=None,
                             save_data=False,
                             save_plot=False,
                             save_local_test_plot=False,
                             show_plot=False,
                             verbose=False,
                             ):
    """
    Saving interval data will only
    :param data_root:
    :param house_num:
    :param channel_tuples:
    :param do_median_filter_and_downsample_p:
    :param intervals:
    :param priors:
    :param dest_root:
    :param save_data:
    :param save_plot:
    :param save_local_test_plot:
    :param show_plot:
    :param verbose:
    :return:
    """

    channel_data, channel_labels = read_extracted_house_data_by_channel_batch \
        (data_root, house_num, channel_tuples,
         return_dict=True,
         do_median_filter_and_downsample_p=do_median_filter_and_downsample_p,
         verbose=verbose)

    for interval_spec in intervals:
        # interval_spec: ((interval_start, interval_end), <active_channels>)*
        (interval_start, interval_end), active_channels = interval_spec

        save_path = None
        if save_data or save_plot:
            save_path = os.path.join(dest_root, 'house_{0}_{1}_{2}'
                                     .format(house_num, interval_start, interval_end))
            if not os.path.exists(save_path):
                if verbose:
                    print '>>> Creating save_path: {0}'.format(save_path)
                os.makedirs(save_path)

        if save_data:
            save_extracted_intervals(save_path, channel_data, channel_labels,
                                     interval_spec, priors=priors, verbose=verbose)

        if show_plot or save_plot:
            plot_extracted_house_data(house_num, channel_data, channel_labels,
                                      interval_spec=interval_spec,
                                      using_interval=True,
                                      data_dim=1,
                                      save_plot=save_plot,
                                      save_path=save_path,
                                      show_plot=False,
                                      verbose=verbose)
            if save_local_test_plot:
                # read local data
                local_channel_data, local_channel_labels, local_interval_spec = \
                    read_extracted_interval_house_data(save_path)
                plot_extracted_house_data(house_num, local_channel_data, local_channel_labels,
                                          interval_spec=local_interval_spec,
                                          data_dim=0,
                                          save_plot=save_plot,
                                          save_path=save_path,
                                          save_name_postfix='local',
                                          show_plot=False,
                                          verbose=verbose)

    if show_plot:
        plt.show()


# ------------------------------------------------------------
# House data scripts
# ------------------------------------------------------------

'''
# For testing
read_plot_save_intervals \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,  # REDD_DATA_jw2013_ROOT,
     house_num=1,
     channel_tuples=(
         (3, 'oven'),
         (4, 'oven'),
         (9, 'lighting'),
         (17, 'lighting'),
         (18, 'lighting'),
     ),
     do_median_filter_and_downsample_p=False,
     intervals=((1200, 6800),
                (1054, 6782),
                ),
     dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
     save_data=False,
     save_plot=False,
     show_plot=False,
     verbose=True
     )
'''


# ------------------------------------------------------------
# PRIORS

# d = alpha * log(1 + (n/alpha))

# format: ((<type_name>+), mu, sigma, alpha

PRIORS = (
    (('stove', 'oven'), 6, 90, 2),
    (('kitchen_outlets', 'outlets_unknown'), 10, 90, 5),
    (('microwave', 'dishwasher'), 15, 100, 5),
    (('refrigerator', 'electronics'), 50, 70, 3),
    (('lighting',), 30, 50, 8),
    (('washer_dryer',), 40, 300, 3),
    (('air_conditioning', 'electric_heat'), 30, 150, 3),
)


# ------------------------------------------------------------
# HOUSE 1

house_1_channels_labels = \
    ((3, 'oven'),
     (4, 'oven'),
     # (14, 'stove'),
     #
     ## (7, 'kitchen_outlets'),
     (8, 'kitchen_outlets'),
     (15, 'kitchen_outlets'),
     (16, 'kitchen_outlets'),
     #
     (9, 'lighting'),
     (17, 'lighting'),
     (18, 'lighting'),

     #
     (10, 'washer_dryer'),
     (20, 'washer_dryer')
     )

# Format: ((<interval_start>, <interval_end>), <active channels>)

house_1_interval_channel_short_spec = \
    (((1200, 6800), (3, 4, 8, 15, 16, 9, 17, 18)),)

house_1_interval_channel_spec = \
    (# 5600 ; ++  ; 14 (stove), 10 (washer_dryer), 20 (washer_dryer) == 0
     # ((1200, 6800), (3, 4, 7, 8, 15, 16, 9, 17, 18)),
     ((1200, 6800), (3, 4, 8, 15, 16, 9, 17, 18)),
     # 5200 ; ++  ; 3 (oven), 4 (oven), 14 (stove) == 0
     # ((22600, 27800), (7, 8, 15, 16, 9, 17, 18, 10, 20)),
     ((22600, 27800), (8, 15, 16, 9, 17, 18, 10, 20)),
     # 5000 ; +/  ; 14 (stove), 10 (washer_dryer) == 0
     # ((27300, 32300), (3, 4, 7, 8, 15, 16, 9, 17, 18, 20)),
     ((27300, 32300), (3, 4, 8, 15, 16, 9, 17, 18, 20)),
     # 3700 ; +   ; 3 (oven), 4 (oven), 14 (stove), 10 (washer_dryer), 20 (washer_dryer) == 0
     # ((48000, 51700), ()),
     # 3800 ; +++ ; 14 (stove), 16 (kitchen_outlets) == 0
     # ((55300, 59100), (3, 4, 7, 8, 15, 9, 17, 18, 10, 20)),
     ((55300, 59100), (3, 4, 8, 15, 9, 17, 18, 10, 20)),
     # 3900 ; ++  ; 3 (oven), 4 (oven), 14 (stove) == 0
     # ((80000, 83900), (7, 8, 15, 16, 9, 17, 18, 10, 20)),
     ((80000, 83900), (8, 15, 16, 9, 17, 18, 10, 20)),
     # 6100 ; +   ; 3 (oven), 4 (oven), 14 (stove) == 0
     # ((105500, 111600), ())
     # 4200 ; ++  ; 14 (stove), 10 (washer_dryer) == 0
     # ((112600, 116800), (3, 4, 7, 8, 15, 16, 9, 17, 18, 20)),
     ((112600, 116800), (3, 4, 8, 15, 16, 9, 17, 18, 20))
     )

house_1_ovens = \
    ((3, 'oven'),
     (4, 'oven'),
     (14, 'stove (NII)')
     )

house_1_outlets = \
    ((7, 'kitchen_outlets (NII-2)'),
     (8, 'kitchen_outlets'),
     (15, 'kitchen_outlets'),
     (16, 'kitchen_outlets')
     )

# not used in these experiments
house_1_kitchen_appliances = \
    ((5, 'refrigerator (NII)'),
     (6, 'dishwasher (NII)'),
     (11, 'microwave (NII)'))

house_1_lighting = \
    ((9, 'lighting'),
     (17, 'lighting'),
     (18, 'lighting'))

house_1_washer_dryer = \
    ((10, 'washer_dryer'),
     (19, 'washer_dryer (NII)'),
     (20, 'washer_dryer'))

house_1_heating_cooling = \
    ((13, 'electric_heat (NII)'),
     )

'''
plot_extracted_house_data_histogram(data_root=REDD_DATA_jw2013_downsampled_ROOT,
                                    house_num=1,
                                    channel_tuples=house_1_channels_labels_sub,
                                    clip=None,
                                    max_y=200,
                                    show_plot=True,
                                    verbose=True)
'''


def generate_house_1(verbose=False):
    print '>>> generate_house_1()'

    read_plot_save_intervals(data_root=REDD_DATA_jw2013_downsampled_ROOT,
                             house_num=1,
                             channel_tuples=house_1_channels_labels,
                             do_median_filter_and_downsample_p=False,
                             intervals=house_1_interval_channel_spec,
                             priors=PRIORS,
                             dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
                             save_data=True,
                             save_plot=True,
                             save_local_test_plot=True,
                             show_plot=False,
                             verbose=verbose,
                             )


'''
# HOUSE 1
read_plot_save_intervals \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,  # REDD_DATA_jw2013_ROOT,
     house_num=1,
     channel_tuples=(
         # (1, 'mains'),
         # (2, 'mains'),

         (3, 'oven'),
         (4, 'oven'),
         # (5, 'refrigerator'),
         # (6, 'dishwasher'),
         # (11, 'microwave'),
         (14, 'stove'),

         (7, 'kitchen_outlets'),
         (8, 'kitchen_outlets'),
         (15, 'kitchen_outlets'),
         (16, 'kitchen_outlets'),

         # (12, 'bathroom_gfi'),
         # (13, 'electric_heat'),

         (9, 'lighting'),
         (17, 'lighting'),
         (18, 'lighting'),

         (10, 'washer_dryer'),
         # (19, 'washer_dryer'),  # no value
         (20, 'washer_dryer')
     ),
     do_median_filter_and_downsample_p=False,
     intervals=((1200, 6800),      # 5600 ; ++  ; 14 (stove), 10 (washer_dryer), 20 (washer_dryer) == 0
                (22600, 27800),    # 5200 ; ++  ; 3 (oven), 4 (oven), 14 (stove) == 0
                (27300, 32300),    # 5000 ; +/  ; 14 (stove), 10 (washer_dryer) == 0
                # (48000, 51700),    # 3700 ; +   ; 3 (oven), 4 (oven), 14 (stove), 10 (washer_dryer), 20 (washer_dryer) == 0
                (55300, 59100),    # 3800 ; +++ ; 14 (stove), 16 (kitchen_outlets) == 0
                (80000, 83900),    # 3900 ; ++  ; 3 (oven), 4 (oven), 14 (stove) == 0
                # (105500, 111600),  # 6100 ; +   ; 3 (oven), 4 (oven), 14 (stove) == 0
                (112600, 116800),  # 4200 ; ++  ; 14 (stove), 10 (washer_dryer) == 0
                ),
     dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
     save_data=True,
     save_plot=True,
     show_plot=False,
     verbose=True
     )
'''


# ------------------------------------------------------------
# HOUSE 2

house_2_channels_labels = \
    ((3, 'kitchen_outlets'),
     (8, 'kitchen_outlets'),
     (4, 'lighting'),
     (5, 'stove'),
     (6, 'microwave'),
     # (7, 'washer_dryer'),  ##
     # (9, 'refrigerator'),  ##
     (10, 'dishwasher'),
     ## (11, 'disposal')
     )

# Format: ((<interval_start>, <interval_end>), <active channels>)
house_2_interval_channel_spec = \
    (# 6000 ; +   ; 5 (stove) == 0
     # ((700, 6700), (3, 8, 4, 6, 10, 11)),
     ((700, 6700), (3, 8, 4, 6, 10)),
     # 8200 ; +   ; 3 (kitchen_outlet) == 0
     # ((15000, 23200), (8, 4, 5, 6, 10, 11)),
     ((15000, 23200), (8, 4, 5, 6, 10)),
     # 5400 ; ++  ; 10 (dishwasher) == 0
     # ((36400, 41800), (3, 8, 4, 5, 6, 11)),
     ((36400, 41800), (3, 8, 4, 5, 6)),
    )

house_2_ovens = \
    ((5, 'stove'),)

house_2_outlets = \
    ((3, 'kitchen_outlets'),
     (8, 'kitchen_outlets'))

house_2_kitchen_appliances = \
    ((9, 'refigerator (NII)'),
     (6, 'microwave'),
     (10, 'dishwasher'),
     (11, 'disposal (NII-2)')
     )

house_2_lighting = \
    ((4, 'lighting'),)

house_2_washer_dryer = \
    ((7, 'washer_dryer (NII)'),
     )

house_2_heating_cooling = None


def generate_house_2(verbose=False):
    print '>>> generate_house_2()'

    read_plot_save_intervals(data_root=REDD_DATA_jw2013_downsampled_ROOT,
                             house_num=2,
                             channel_tuples=house_2_channels_labels,
                             do_median_filter_and_downsample_p=False,
                             intervals=house_2_interval_channel_spec,
                             priors=PRIORS,
                             dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
                             save_data=True,
                             save_plot=True,
                             save_local_test_plot=True,
                             show_plot=False,
                             verbose=verbose,
                             )

# generate_house_2(verbose=True)


'''
# HOUSE 2
read_plot_save_intervals \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     house_num=2,
     channel_tuples=((3, 'kitchen_outlets'),
                     (8, 'kitchen_outlets'),
                     (4, 'lighting'),
                     (5, 'stove'),
                     (6, 'microwave'),
                     # (7, 'washer_dryer'),  ##
                     # (9, 'refrigerator'),  ##
                     (10, 'dishwasher'),
                     (11, 'disposal')
                     ),
     do_median_filter_and_downsample_p=False,
     intervals=((700, 6700),     # 6000 ; +   ; 5 (stove) == 0
                (36400, 41800),  # 5400 ; ++  ; 10 (dishwasher) == 0
                (15000, 23200),  # 8200 ; +   ; 3 (kitchen_outlet) == 0
                ),
     dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
     save_data=True,
     save_plot=True,
     show_plot=False,
     verbose=True
     )
'''


# ------------------------------------------------------------
# HOUSE 3

# 3, 4, 6, 7, 16, 22, 13, 14, 11, 15, 17, 19
house_3_channels_labels = \
    ((3, 'outlets_unknown'),
     ## (4, 'outlets_unknown'),

     (6, 'electronics'),
     (7, 'refrigerator'),

     (16, 'microwave'),

     (22, 'kitchen_outlets'),

     (13, 'washer_dryer'),
     (14, 'washer_dryer'),

     (11, 'lighting'),
     ## (15, 'lighting'),
     (17, 'lighting'),
     (19, 'lighting')
     )

# Format: ((<interval_start>, <interval_end>), <active channels>)
house_3_interval_channel_spec = \
    (# 5000 ; +  ; 3 (outlets_unknown), 8 (disposal), 9 (dishwasher),  21 (k_outlets), 13, 14 (wash_dryer) == 0
     # ((3000, 8000), (4, 6, 7, 16, 22, 11, 15, 17, 19)),
     ((3000, 8000), (6, 7, 16, 22, 11, 17, 19)),
     # 5000 ; ++ ; 8 (disposal=1), 9 (dishwasher), 15 (microwave=1), 21 (kitchen_outlets) == 0
     # ((9000, 14000), (3, 4, 6, 7, 16, 22, 13, 14, 11, 15, 17, 19)),
     ((9000, 14000), (3, 6, 7, 16, 22, 13, 14, 11, 17, 19)),
     # 5000 ; +  ; 3 (outlets_unknown), 8 (disposal), 9 (dishwasher=1), 21, 22 (kitchen_outlets) == 0
     # ((19000, 24000), (4, 6, 7, 16, 22, 13, 14, 11, 15, 17, 19)),
     ((19000, 24000), (6, 7, 16, 22, 13, 14, 11, 17, 19)),
     # 5000 ; ++ ; 8 (disposal=1), 9 (dishwasher),  21 (kitchen_outlets) == 0
     # ((37000, 42000), (3, 4, 6, 7, 16, 22, 13, 14, 11, 15, 17, 19)),
     ((37000, 42000), (3, 6, 7, 16, 22, 13, 14, 11, 17, 19)),
     # 5000 ; +  ;  3 (outlets_unknown), 8 (disposal), 9 (dishwasher=1), 13, 14 (washer_dryer) == 0
     # ((42000, 47000), (4, 6, 7, 16, 22, 13, 14, 11, 15, 17, 19))
     ((42000, 47000), (6, 7, 16, 22, 13, 14, 11, 17, 19))
     )

house_3_ovens = None

house_3_outlets = \
    ((12, 'outlets_unknown (NII)'),
     (3, 'outlets_unknown'),
     (4, 'outlets_unknown (NII-2)'),
     (21, 'kitchen_outlets (NII)'),
     (22, 'kitchen_outlets'))

house_3_kitchen_appliances = \
    ((6, 'electronics'),
     (7, 'refrigerator'),
     (16, 'microwave'),
     (8, 'disposal (NII)'),
     (9, 'dishwaser (NII)')
     )

house_3_lighting = \
    ((5, 'lighting (NII)'),
     (11, 'lighting'),
     (15, 'lighting (NII-2)'),
     (17, 'lighting'),
     (19, 'lighting'))

house_3_washer_dryer = \
    ((13, 'washer_dryer'),
     (14, 'washer_dryer'))

house_3_heating_cooling = \
    ((10, 'furnace (NII)'),)


def generate_house_3(verbose=False):
    print '>>> generate_house_3()'

    read_plot_save_intervals(data_root=REDD_DATA_jw2013_downsampled_ROOT,
                             house_num=3,
                             channel_tuples=house_3_channels_labels,
                             do_median_filter_and_downsample_p=False,
                             intervals=house_3_interval_channel_spec,
                             priors=PRIORS,
                             dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
                             save_data=True,
                             save_plot=True,
                             save_local_test_plot=True,
                             show_plot=False,
                             verbose=verbose,
                             )

# generate_house_3(verbose=True)

'''
# HOUSE 3
read_plot_save_intervals \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     house_num=3,
     channel_tuples=(# (1, 'mains'),
                     # (2, 'mains'),

                     (3, 'outlets_unknown'),
                     (4, 'outlets_unknown'),
                     # (12, 'outlets_unknown'),  ## not very active

                     # (5, 'lighting'),   ## not very active
                     (6, 'electronics'),
                     (7, 'refrigerator'),
                     # (8, 'disposal'),
                     # (9, 'dishwaser'),
                     (16, 'microwave'),

                     # (21, 'kitchen_outlets'),
                     (22, 'kitchen_outlets'),

                     # (10, 'furance'),  ## infrequent

                     (13, 'washer_dryer'),
                     (14, 'washer_dryer'),

                     # (18, 'smoke_alarms'),  ## very low comparative wattage

                     # (20, 'bathroom_gfi'),  ## infrequent

                     (11, 'lighting'),
                     (15, 'lighting'),
                     (17, 'lighting'),
                     (19, 'lighting')
     ),
     do_median_filter_and_downsample_p=False,
     intervals=((3000, 8000),    # 5000 ; +  ; 3 (outlets_unknown), 8 (disposal), 9 (dishwasher),  21 (kitchen_outlets), 13, 14 (washer_dryer) == 0
                (9000, 14000),   # 5000 ; ++ ; 8 (disposal=1), 9 (dishwasher), 15 (microwave=1), 21 (kitchen_outlets) == 0
                (19000, 24000),  # 5000 ; +  ; 3 (outlets_unknown), 8 (disposal), 9 (dishwasher=1), 21, 22 (kitchen_outlets) == 0
                (37000, 42000),  # 5000 ; ++ ; 8 (disposal=1), 9 (dishwasher),  21 (kitchen_outlets) == 0
                (42000, 47000),  # 5000 ; +  ;  3 (outlets_unknown), 8 (disposal), 9 (dishwasher=1), 13, 14 (washer_dryer) == 0
                ),
     dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
     save_data=True,
     save_plot=True,
     show_plot=False,
     verbose=True
     )
'''


# ------------------------------------------------------------
# HOUSE 6

house_6_channels_labels = \
    ((8, 'refrigerator'),

     (10, 'outlets_unknown'),
     (11, 'outlets_unknown'),

     (12, 'electric_heat'),

     (14, 'lighting'),

     (15, 'air_conditioning'),
     (16, 'air_conditioning'),
     (17, 'air_conditioning')
     )

# Format: ((<interval_start>, <interval_end>), <active channels>)
house_6_interval_channel_spec = \
    (# 5000 ; +  ; 12 (electric_heat) == 0
     ((15000, 20000), (8, 10, 11, 14, 15, 16, 17)),
     # 5000 ; +  ; 3 (kitchen_outlets), 5 (stove) == 0
     ((29000, 36000), (8, 10, 11, 12, 14, 15, 16, 17)),
     # 5000 ; +  ; 3 (kitchen_outlets), 5 (stove) == 0
     ((36000, 41000), (8, 10, 11, 12, 14, 15, 16, 17)),
     # 5000 ; ++ ; 3 (kitchen_outlets) == 0
     ((46000, 51000), (8, 10, 11, 12, 14, 15, 16, 17)),
     # 5000 ;
     ((51000, 56000), (8, 10, 11, 12, 14, 15, 16, 17))
     )

house_6_ovens = \
    ((5, 'stove (NII)'),
     )

house_6_outlets = \
    ((10, 'outlets_unknown'),
     (11, 'outlets_unknown'),
     (3, 'kitchen_outlets (NII)'),
     (13, 'kitchen_outlets (NII)'),
     )

house_6_kitchen_appliances = \
    ((8, 'refrigerator'),
     (9, 'dishwaser (NII)'),
     )

house_6_lighting = \
    ((14, 'lighting'),)

house_6_washer_dryer = \
    ((4, 'washer_dryer (NII)'),
     )

house_6_heating_cooling = \
    ((12, 'electric_heat'),
     (15, 'air_conditioning'),
     (16, 'air_conditioning'),
     (17, 'air_conditioning'))


def generate_house_6(verbose=False):
    print '>>> generate_house_1()'

    read_plot_save_intervals(data_root=REDD_DATA_jw2013_downsampled_ROOT,
                             house_num=6,
                             channel_tuples=house_6_channels_labels,
                             do_median_filter_and_downsample_p=False,
                             intervals=house_6_interval_channel_spec,
                             priors=PRIORS,
                             dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
                             save_data=True,
                             save_plot=True,
                             save_local_test_plot=True,
                             show_plot=False,
                             verbose=verbose,
                             )

# generate_house_6(verbose=True)


'''
# HOUSE 6
read_plot_save_intervals \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     house_num=6,
     channel_tuples=(# (1, 'mains'),
                     # (2, 'mains'),

                     ##(3, 'kitchen_outlets'),
                     ##(13, 'kitchen_outlets'),
                     ##(5, 'stove'),
                     # (9, 'dishwaser'),
                     (8, 'refrigerator'),

                     # (6, 'electronics'),

                     # (4, 'washer_dryer'),

                     # (7, 'bathroom_gfi'),

                     (10, 'outlets_unknown'),
                     (11, 'outlets_unknown'),

                     (12, 'electric_heat'),

                     (14, 'lighting'),

                     (15, 'air_conditioning'),
                     (16, 'air_conditioning'),
                     (17, 'air_conditioning')
     ),
     do_median_filter_and_downsample_p=False,
     intervals=((15000, 20000),  # 5000 ; +  ; 12 (electric_heat) == 0
                (29000, 36000),  # 5000 ; +  ; 3 (kitchen_outlets), 5 (stove) == 0
                (36000, 41000),  # 5000 ; +  ; 3 (kitchen_outlets), 5 (stove) == 0
                (46000, 51000),  # 5000 ; ++ ; 3 (kitchen_outlets) == 0
                (51000, 56000),  # 5000 ;
                ),
     dest_root=REDD_DATA_jw2013_downsampled_intervals_ROOT,
     save_data=True,
     save_plot=True,
     show_plot=False,
     verbose=True
     )
'''


# ------------------------------------------------------------
# Manual analysis of device type: mu, sigma, alpha
# ------------------------------------------------------------

'''
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((2, house_2_channels_labels_sub), ),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )
'''


'''
# mu=6, sigma=90, alpha=2
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_ovens), (2, house_2_ovens), (3, house_3_ovens), (6, house_6_ovens)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )

# turn off:
#     house 1: 7_kitchen_outlets
#     house 3: 4_outlets_unknown
# mu=10, sigma=90, alpha=5
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_outlets), (2, house_2_outlets),
                (3, house_3_outlets), (6, house_6_outlets)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )

# mu=30, sigma=50, alpha=8
# turn off:
#     house 2: disposal
# microwave, dishwasher: mu=15, sigma=100, alpha=5
# refrigerator, electronics: mu=50, sigma=70, alpha=3
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_kitchen_appliances), (2, house_2_kitchen_appliances),
                (3, house_3_kitchen_appliances), (6, house_6_kitchen_appliances)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )

# turn off:
#     house 3: 15_lighting
# mu=30, sigma=50, alpha=8
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_lighting), (2, house_2_lighting),
                (3, house_3_lighting), (6, house_6_lighting)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )

# mu=40, sigma=300, alpha=3
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_washer_dryer), (2, house_2_washer_dryer),
                (3, house_3_washer_dryer), (6, house_6_washer_dryer)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )

# turn off:
#     house 1: 13_electric_heat
# mu=30, sigma=150, alpha=3
plot_extracted_house_data_histogram_batch \
    (data_root=REDD_DATA_jw2013_downsampled_ROOT,
     data_spec=((1, house_1_heating_cooling), (2, house_2_heating_cooling),
                (3, house_3_heating_cooling), (6, house_6_heating_cooling)),
     clip=None,
     max_y=1000,
     save_plot=False, save_path=None,
     show_plot=True,
     verbose=False
     )
'''

# ------------------------------------------------------------
# Find changepoints
# ------------------------------------------------------------

def find_changepoints(data, threshold=1, verbose=False):

    if verbose:
        print 'Finding changepoints'

    return numpy.concatenate(((False,), numpy.abs(numpy.diff(data[:, 1])) > threshold))


def test_find_changepoints(house='1', channel=5):
    d = read_data(REDD_DATA_jw2013_ROOT, house, channel, verbose=True)
    df = median_filter_seconds_precise(d, 10, verbose=True)
    cp = find_changepoints(df, verbose=True)

    diffs = numpy.abs(numpy.diff(df[:, 1]))
    plot_data_value_histogram(diffs)
    plt.ylim((0, 100))

    plt.show()

    print cp

    # plot_browser(fig_spec=plot_data_values(cp))

# test_find_changepoints()


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

'''
extract_house_data_jw2013(REDD_data_root, REDD_data_extracted_root,
                          house_params=(house_1, house_2, house_3, house_6),
                          devices=('refrigerator', 'lighting', 'dishwasher', 'microwave', 'furnace'),
                          w_limit=10, verbose=True)
'''

'''
data_list, ls = read_extracted_house_data(REDD_DATA_jw2013_extracted_ROOT, house=house_1,  # house_1 house_2 house_3 house_6
                                          devices=('refrigerator', 'lighting', 'dishwasher', 'microwave', 'furnace'),
                                          verbose=True)
plot_browser(fig_spec=plot_data_values_list_stacked(data_list[1:],
                                                    labels=ls,
                                                    hour_marker_timestamps=data_list[0]))
'''


def script(house, channel_list, w_limit=10):

    print 'start'

    dat = list()
    dat_filtered = list()
    for ch in channel_list:
        d = read_data(REDD_DATA_jw2013_ROOT, house, ch, verbose=True)
        dat.append(d)
        dat_filtered.append(median_filter_seconds_precise(d, w_limit, verbose=True))

    print 'aggregating {0} channels'.format(len(dat))
    dat_agg = aggregate_data_sum(dat_filtered)

    # dat_agg_m = median_filter(dat_agg, w_limit)

    plot_data_value_histogram(dat_agg[:, 1])

    data_list = [ d[:, 1] for d in [dat_agg] + dat ]

    plot_browser(fig_spec=plot_data_values_list_stacked(data_list))  # 30000, 80000


# script(house='1', channel_list=(5, 17, 18, 6, 11, 13))

# script(house='3', channel_list=(5, 11, 15, 17, 19))


'''
print datetime.datetime.utcfromtimestamp(1303132964).strftime('%Y-%m-%d %H:%M:%S')
print datetime.datetime.utcfromtimestamp(1303132967).strftime('%Y-%m-%d %H:%M:%S')
print datetime.datetime.utcfromtimestamp(1303132971).strftime('%Y-%m-%d %H:%M:%S')

print datetime.datetime.utcfromtimestamp(1303132971 - 1303132964).strftime('%Y-%m-%d %H:%M:%S')
'''


# ------------------------------------------------------------

# IF having trouble finding files: possibly due to mismatch
#   between REDD_DATA_ROOT and current working directory
# Run the following two lines to verify...
# print os.getcwd()
# print os.listdir(REDD_DATA_ROOT)

# generate_house_1(verbose=True)
# generate_house_2(verbose=True)
# generate_house_3(verbose=True)
# generate_house_6(verbose=True)
