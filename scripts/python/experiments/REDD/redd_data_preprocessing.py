import numpy
import math
import datetime
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, SpanSelector
import collections

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

# print os.listdir(REDD_DATA_ROOT)

## old paths...
# REDD_data_root = '../../figures/REDD/johnson_willsky'
# REDD_data_extracted_jw2013_root = '../../figures/REDD/extracted_jw2013'
# REDD_data_extracted_jw2013_downsampled_root = '../../figures/REDD/extracted_jw2013_downsampled'


# ------------------------------------------------------------
# REDD house figures
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


def load_data_channel_batch(data_root, house_num, channels, verbose=False):

    if verbose:
        print 'load_all_data(): {0}, house_{1}, channels={2}'.format(data_root, house_num, channels)

    house_path = os.path.join(data_root, 'house_{0}'.format(house_num))

    data = list()
    for channel_num in channels:
        path = os.path.join(house_path, 'channel_{0}.dat'.format(channel_num))
        datum = numpy.loadtxt(path)
        print 'channel {0}, datum.shape={1}'.format(channel_num, datum.shape)
        data.append(datum)

    '''
    for i in range(1, len(data)):
        print i, numpy.setdiff1d(data[0][:, 0], data[i][:, 0]), numpy.setdiff1d(data[i][:, 0], data[0][:, 0])
    '''

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

def median_filter_and_downsample_seconds_batch(data_list, seconds, verbose=False, test=False):
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


# ------------------------------------------------------------
# Aggregating
# ------------------------------------------------------------

def aggregate_data_sum(data_list, verbose=False):
    if verbose:
        print 'Aggregating by summing {0} channels'.format(len(data_list))
    new_data = numpy.copy(data_list[0])
    if len(data_list) > 1:
        for i in range(1, new_data.shape[0]):
            for datum in data_list:
                if datum is not None:
                    new_data[i, 1] += datum[i, 1]
    return new_data


# ------------------------------------------------------------
# Find hour boundaries
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
# Plotting
# ------------------------------------------------------------

def plot_data_value_histogram(data):
    plt.figure()
    actualmax = numpy.max(data)
    bins = numpy.linspace(0, actualmax, 100)
    n, bins, patches = plt.hist(data, bins, histtype='bar')  # normed=0,
    plt.setp(patches, 'facecolor', 'b', 'alpha', 0.5)
    plt.ylim((0, 100000))


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


'''
def plot_hour_markers(fig_spec, timestamps, s=None, e=None, new_fig=False):

    fig, xmin, xmax, ymax = fig_spec

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
            plt.plot((xp, xp), (0, ymax), 'r--')
        else:
            plt.plot((xp, xp), (0, ymax), 'y--')

    return fig, s, e, ymax
'''


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


def plot_data_values_list_stacked(data_list, s=None, e=None, labels=None,
                                  hour_marker_timestamps=None,
                                  title=None):
    font = {'family': 'serif',
            'color': 'darkblue',  # 'darkred'
            'weight': 'normal',
            'size': 10,
            }

    colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))
    fig, axarr = plt.subplots(len(data_list), sharex=True)

    # dynamically resize depending on size of
    num_channels = len(data_list)
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
# Extract house figures
# ------------------------------------------------------------

def extract_house_data_jw2013(source_dir, dest_dir,
                              house_params=(house_1, house_2, house_3, house_6),
                              devices=('refrigerator', 'lighting', 'dishwasher', 'microwave', 'furnace'),
                              w_limit=10, verbose=True):
    """
    Extract house data from original REDD data
    Follows Johnson & Willsky 2013
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
# Read extracted house figures
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
    NOTE: read_extracted_house_data_by_channel_batch allows specifying name of channel
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


def read_extracted_house_data_by_channel_batch(data_root, house_num, channel_tuples,
                                               do_median_filter_and_downsample_p=False,
                                               verbose=False):
    """
    Read specifying channels for extracted house data.
    :param data_root:
    :param house_num:
    :param channel_tuples: sequence of tuples: (<channel_number>, <channel_label>)
    :param do_median_filter_and_downsample_p: (default: False)
                If true, compute median filter downsample on read data;
                Default assumes already filtered and downsampled.
    :param verbose:
    :return:
    """

    channels = [cl[0] for cl in channel_tuples]
    channel_labels = ['{0} {1}'.format(cl[0], cl[1]) for cl in channel_tuples]

    data_list = load_data_channel_batch(data_root, house_num,
                                        channels=channels,
                                        verbose=verbose)
    if verbose:
        print 'len(data_list)', len(data_list), \
            ', data_list[0].shape', data_list[0].shape
    if do_median_filter_and_downsample_p:
        data_list_mf = median_filter_and_downsample_seconds_batch \
            (data_list, 20, verbose=verbose, test=True)
    else:
        data_list_mf = data_list
    if verbose:
        print 'len(data_list_mf)', len(data_list_mf), \
            ', data_list_mf[0].shape', data_list_mf[0].shape

    '''
    for elm in data_list_mf:
        print 'datum.shape', elm.shape
        for i in range(elm.shape[0]):
            if i % 1000 == 0:
                print i, elm[i]
    '''

    data_agg = aggregate_data_sum(data_list_mf, verbose=True)
    all_data = [data_agg] + data_list_mf
    channel_labels = ['all_summed'] + channel_labels

    return all_data, channel_labels


def plot_extracted_house_data(house_num, channel_data, channel_labels,
                              interval=None,
                              save_plot=False,
                              save_path=None,
                              show_plot=False,
                              verbose=False):
    """

    :param house_num:
    :param channel_data:
    :param channel_labels:
    :param interval: (default None) optional interval tuple: (<start_index>, <end_index>)
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

    if interval:
        s, e = interval
        if verbose:
            print 'plotting interval [{0}, {1}] ...'.format(s, e),
        plot_browser(fig_spec=get_plot(title='House {0} [{1}, {2}]'.format(house_num, s, e)),
                     s=s, x_display_max=e, show_p=False)
        if verbose:
            print 'DONE.'

        if save_plot is not None and save_path is not None:
            path = os.path.join(save_path, 'house_{0}_{1}-{2}.pdf'.format(house_num, s, e))
            if verbose:
                print 'saving plot {0} ...'.format(path),
            plt.savefig(path, format='pdf')
            if verbose:
                print 'DONE.'

    else:
        plot_browser(fig_spec=get_plot(), show_p=False)

    if show_plot:
        plt.show()


def save_extracted_intervals(save_path, channel_data, channel_labels, interval, verbose):

    # create save_path from dest_root and house_num
    # <dest_root>/house_<house_num>_<s>_<e>/
    #     sources/
    #         channel_<num>.txt
    #         ...
    #         labels.txt
    #     obs.txt   # aggregated (summed) data: 'all_summed'

    if verbose:
        print '---------- Saving {0}'.format(save_path)
        print 'len(channel_data)', len(channel_data)
        print 'channel_data[0][:,1] (all_summed)', channel_data[0][interval[0]:interval[1],1]
        print 'channel_labels', channel_labels
        print 'interval', interval

    # save obs.txt == 'all_summed' == channel_data[0]
    obs_path = os.path.join(save_path, 'obs.txt')
    if verbose:
        print 'Saving obs.txt [{0}, {1}]: {2}'.format(interval[0], interval[1], obs_path)
    numpy.savetxt(obs_path, channel_data[0][interval[0]:interval[1], 1], fmt='%1.2f')

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

    if verbose:
        print 'DONE.'


def read_plot_save_intervals(data_root, house_num, channel_tuples,
                             do_median_filter_and_downsample_p=False,
                             intervals=None,
                             dest_root=None,
                             save_data=False,
                             save_plot=False,
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
    :param dest_root:
    :param save_data:
    :param save_plot:
    :param show_plot:
    :param verbose:
    :return:
    """

    channel_data, channel_labels = read_extracted_house_data_by_channel_batch \
        (data_root, house_num, channel_tuples,
         do_median_filter_and_downsample_p=do_median_filter_and_downsample_p)

    if intervals is not None:

        for interval in intervals:

            save_path = None
            if save_data or save_plot:
                save_path = os.path.join(dest_root, 'house_{0}_{1}_{2}'.format(house_num, interval[0], interval[1]))
                if not os.path.exists(save_path):
                    if verbose:
                        print '>>> Creating save_path: {0}'.format(save_path)
                    os.makedirs(save_path)

            if save_data:
                save_extracted_intervals(save_path, channel_data, channel_labels, interval, verbose=verbose)

            if show_plot or save_plot:
                plot_extracted_house_data(house_num, channel_data, channel_labels,
                                          interval=interval,
                                          save_plot=save_plot,
                                          save_path=save_path,
                                          show_plot=False,
                                          verbose=verbose)

    else:
        if show_plot or save_plot:
            plot_extracted_house_data(house_num, channel_data, channel_labels,
                                      interval=None,
                                      save_plot=save_plot,
                                      save_path=dest_root,
                                      show_plot=False,
                                      verbose=verbose)

    if show_plot:
        plt.show()

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
