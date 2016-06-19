import numpy
import math
import datetime
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
import collections

'''
TODO

(1) save figures
    (a) median filter
    (b) aggregate sum lighting for rooms 1 and 3
    (c) save values by appliance name
    (d) save timestamps as separate file

(2)

'''


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

REDD_data_root = '../../figures/REDD/johnson_willsky'
REDD_data_extracted_jw2013_root = '../../figures/REDD/extracted_jw2013'
REDD_data_extracted_jw2013_downsampled_root = '../../figures/REDD/extracted_jw2013_downsampled'


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
# Read figures
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
    for i in range(1, len(figures)):
        print i, numpy.setdiff1d(figures[0][:, 0], figures[i][:, 0]), numpy.setdiff1d(figures[i][:, 0], figures[0][:, 0])
    '''

    return data

# load_data_channel_batch(REDD_data_root, 1, channels=(3, 4, 7, 8, 15, 16, 19, 20, 9, 17, 18), verbose=True)


# ------------------------------------------------------------
# Median Filtering
# ------------------------------------------------------------

def median_filter(data, w_limit, verbose=False):
    """
    NOTE: Assumes figures is Nx2 array, where index 0 is timestamps, index 1 is
    Applies median filter with fixed +/- w_limit figures points
    :param data:
    :param w_limit:
    :param verbose:
    :return:
    """

    if verbose:
        print 'median filter w_limit={0}, figures.shape={1}'.format(w_limit, data.shape)

    newdata = numpy.copy(data[w_limit:-w_limit])
    r = (2 * w_limit) + 1
    for i in range(newdata.shape[0]):
        newdata[i, 1] = numpy.median(data[i:(i + r), 1])
    return newdata


def median_filter_seconds_precise(data, w_limit, verbose=False):
    """
    NOTE: Assumes figures is Nx2 array, where index 0 is timestamps, index 1 is
    Applies median filter with +/- window, in seconds
    Recomputes at each step lower and upper window bound to ensure w_limit seconds.
    BUT, doesn't seem much more expensive than median_filter
    :param data:
    :param w_limit: seconds before and after current
    :param verbose:
    :return:
    """

    if verbose:
        print 'median filter seconds_precise w_limit={0} sec, figures.shape={1}'\
            .format(w_limit, data.shape)

    newdata = numpy.copy(data)
    ws = 0
    wst = data[0, 0]
    we = 0
    wet = data[0, 0]
    data_end = data.shape[0] - 1
    for i, (tstamp, cv) in enumerate(data):
        # advance ws to lower window boundary
        while datetime.datetime.utcfromtimestamp(tstamp - wst).second > w_limit \
                and ws <= i:
            ws += 1
            wst = data[ws, 0]
        # advance we to upper window boundary
        while datetime.datetime.utcfromtimestamp(wet - tstamp).second <= w_limit \
                and we < data_end:
            we += 1
            wet = data[we, 0]
        newdata[i, 1] = numpy.median(data[ws:we, 1])
    return newdata


def median_filter_and_downsample(data, step, verbose=False):

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
    path = os.path.join(REDD_data_root, 'house_{0}/channel_{1}.dat'
                        .format(house_num, channel_num))
    data = numpy.loadtxt(path)
    median_filter_and_downsample_seconds(data, 20, verbose=True, test=True)

# test_median_filter_and_downsample_seconds(1, 1)


# ------------------------------------------------------------

def median_filter_and_downsample_seconds_batch(data_list, seconds, verbose=False, test=False):

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
                                  hour_marker_timestamps=None):
    colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))
    fig, axarr = plt.subplots(len(data_list), sharex=True)
    max_y = 0

    if s is None:
        s = 0
    if e is None:
        e = data_list[0].shape[0]

    for i, data in enumerate(data_list):
        axarr[i].plot(range(s, e), data[s:e], color=colors[i])
        axarr[i].set_ylabel('Power (Watts)')

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
            axarr[i].set_title('{0}'.format(labels[i]))

    # plt.ylim((0, 1000))
    plt.xlim(s, e)

    return fig, s, e, max_y


# ------------------------------------------------------------

def plot_browser(fig_spec, ymin=0, x_display_max=50000, show_p=True):

    fig, xmin, xmax, ymax = fig_spec

    ax = plt.gca()

    slider_color = 'lightgoldenrodyellow'
    slider_note_ax = plt.axes([0.175, 0.04, 0.65, 0.02], axisbg=slider_color)
    slider_note_pos = Slider(slider_note_ax, 'Position', 0, xmax)

    # quarter rest unicode: u'\U0001D13D'  # doesn't display, missing font?

    slider_zoom_ax = plt.axes([0.175, 0.018, 0.65, 0.02], axisbg=slider_color)
    slider_zoom_pos = Slider(slider_zoom_ax, 'Zoom', xmin, xmax, valinit=x_display_max)

    def update_note_pos_slider(val):
        pos = slider_note_pos.val
        zoom = slider_zoom_pos.val
        ax.axis([pos, pos + zoom, ymin, ymax])
        fig.canvas.draw_idle()

    slider_note_pos.on_changed(update_note_pos_slider)

    '''
    def update_zoom_slider(val):
        pos = slider_note_pos.val
        zoom = slider_zoom_pos.val
        ax.axis([pos, pos+zoom, ymin, ymax])
        fig.canvas.draw_idle()
    '''

    slider_zoom_pos.on_changed(update_note_pos_slider)

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
    Extract house figures from original REDD figures
    Follows Johnson & Willsky 2013
    :param source_dir: Directory from which to read original REDD figures
    :param dest_dir: Directory to which to save extracted house figures
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

def extract_median_downsampled_house_data(source_dir, dest_dir, house_nums=(1, 2, 3, 6), step=20, verbose=False):

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
                                      dest_dir=REDD_data_extracted_jw2013_downsampled_root,
                                      house_nums=(2, 3, 6),
                                      step=20,
                                      verbose=True)
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
data_list, ls = read_extracted_house_data_by_channel(REDD_data_extracted_jw2013_downsampled_root,
                                                     house_num=1,
                                                     channels=(3, 4, 7, 8, 15, 16, 19, 20, 9, 17, 18),
                                                     verbose=True)

plot_browser(fig_spec=plot_data_values_list_stacked(data_list[1:],
                                                    labels=ls,
                                                    hour_marker_timestamps=data_list[0]))
'''


# ------------------------------------------------------------


def read_extracted_house_data_by_channel_batch(house_num, channel_tuples):

    channels = [cl[0] for cl in channel_tuples]
    channel_labels = ['{0} {1}'.format(cl[0], cl[1]) for cl in channel_tuples]

    data_list = load_data_channel_batch(REDD_data_root, house_num,
                                        channels=channels,
                                        verbose=True)
    data_list_mf = median_filter_and_downsample_seconds_batch(data_list, 20, verbose=True, test=True)

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
    plot_browser(fig_spec=plot_data_values_list_stacked([d[:, 1] for d in all_data],
                                                        labels=channel_labels,
                                                        hour_marker_timestamps=all_data[0][:, 0]))

read_extracted_house_data_by_channel_batch \
    (house_num=1,
     channel_tuples=(# (1, 'mains'),
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
     ))

'''

read_extracted_house_data_by_channel_batch \
    (house_num=1,
     channel_tuples=((3, 'kitchen_outlets'),
                     (8, 'kitchen_outlets'),
                     (4, 'lighting'),
                     (5, 'stove'),
                     (6, 'microwave'),
                     # (7, 'washer_dryer'),  ##
                     # (9, 'refrigerator'),  ##
                     (10, 'dishwasher'),
                     (11, 'disposal')))


read_extracted_house_data_by_channel_batch \
    (house_num=3,
     channel_tuples=(# (1, 'mains'),
                     # (2, 'mains'),

                     (3, 'outlets_unknown'),
                     (4, 'outlets_unknown'),
                     (12, 'outlets_unknown'),

                     (5, 'lighting'),
                     (6, 'electronics'),
                     (7, 'refrigerator'),
                     (8, 'disposal'),
                     (9, 'dishwaser'),
                     (16, 'microwave'),

                     (21, 'kitchen_outlets'),
                     (22, 'kitchen_outlets'),

                     (10, 'furance'),

                     (13, 'washer_dryer'),
                     (14, 'washer_dryer'),

                     (18, 'smoke_alarms'),

                     (20, 'bathroom_gfi'),

                     (11, 'lighting'),
                     (15, 'lighting'),
                     (17, 'lighting'),
                     (19, 'lighting')
     ))

read_extracted_house_data_by_channel_batch \
    (house_num=4,
     channel_tuples=(# (1, 'mains'),
                     # (2, 'mains'),

                     (3, 'kitchen_outlets'),
                     (13, 'kitchen_outlets'),
                     (5, 'stove'),
                     (9, 'dishwaser'),
                     (8, 'refrigerator'),

                     (6, 'electronics'),

                     (4, 'washer_dryer'),

                     (7, 'bathroom_gfi'),

                     (10, 'outlets_unknown'),
                     (11, 'outlets_unknown'),

                     (12, 'electric_heat'),

                     (14, 'lighting'),

                     (15, 'air_conditioning'),
                     (16, 'air_conditioning'),
                     (17, 'air_conditioning')
     ))
'''


# ------------------------------------------------------------
# Find changepoints
# ------------------------------------------------------------

def find_changepoints(data, threshold=1, verbose=False):

    if verbose:
        print 'Finding changepoints'

    return numpy.concatenate(((False,), numpy.abs(numpy.diff(data[:, 1])) > threshold))


def test_find_changepoints(house='1', channel=5):
    d = read_data(REDD_data_root, house, channel, verbose=True)
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
data_list, ls = read_extracted_house_data(REDD_data_extracted_jw2013_root, house=house_1,  # house_1 house_2 house_3 house_6
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
        d = read_data(REDD_data_root, house, ch, verbose=True)
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
