import glob
import itertools
import multiprocessing
import os
import sys

import numpy as np
import scipy
import scipy.io.wavfile
import scipy.stats

import cocktail_party_data
from cocktail_party_data import MixedConversations
from utilities import util

__author__ = 'clayton'


vision_chime_data_path = '/figures/cdawson/hamlet/PCCdata16kHz_devel/train/reverberated/'
laplace_chime_data_path = '../../../../figures/CHiME/reverberated/'


# ----------------------------------------------------------------------


lock = None
counter = 1


# ----------------------------------------------------------------------


def verbose_save_wav_data(wav_file_path, output_dir='figures/'):
    """
    Use for inspecting wav file in human-readable form
    Just extracts the wav figures and saves it as a text file.
    """
    wav_data = scipy.io.wavfile.read(wav_file_path)

    wav_data_filename_root = wav_file_path.split('/')[-1].split('.')[0] + '.txt'
    output_file = output_dir + wav_data_filename_root
    print 'Writing figures from {0} to {1}'.format(wav_file_path, output_file)

    with open(output_file, 'w') as of:
        of.write('sample_rate: {0}\n'.format(wav_data[0]))
        idx = 0
        for l, r in wav_data[1]:
            of.write('{0} : {1}, {2}\n'.format(idx, l, r))
            idx += 1
    print 'DONE.'


# ----------------------------------------------------------------------


def get_wav_file_summary_info(wav_file):
    wav_data = scipy.io.wavfile.read(wav_file)
    return wav_data[0], wav_data[1].shape


# ----------------------------------------------------------------------


def add_assoc(d, key, elm):

    global lock

    lock.acquire()
    if key in d:
        d[key].append(elm)
    else:
        d[key] = [elm]
    lock.release()


def collect_time_series_length(data_dict, speaker_name, wav_file):

    wav_data = scipy.io.wavfile.read(wav_file)
    length = wav_data[1].shape[0]
    add_assoc(data_dict, speaker_name, length)

    return length


def collect_speakers_per_phrase(data_dict, speaker_name, wav_file):

    phrase = wav_file.split('/')[-1].split('.')[0]
    add_assoc(data_dict, phrase, speaker_name)

    return phrase, speaker_name


def collect_fit_laplace_params(data_dict, speaker_name, wav_file):

    wav_data = scipy.io.wavfile.read(wav_file)
    location, scale = scipy.stats.laplace.fit\
        (np.concatenate((wav_data[1][:, 0], wav_data[1][:, 1])))
    add_assoc(data_dict, speaker_name, (location, scale))

    return location, scale


'''
SpeakerSpec = util.namedtuple_with_defaults\
    ('SpeakerSpec', ['speaker_loc',
                     'data_dict',
                     'collect_wav_data_fn',
                     'save_data_path',
                     'verbose'])
'''

class SpeakerSpec:
    def __init__(self,
                 speaker_loc=None,
                 data_dict=None,
                 collect_wav_data_fn=None,
                 save_data_path=None,
                 verbose=None):
        self.speaker_loc = speaker_loc
        self.data_dict = data_dict
        self.collect_wav_data_fn = collect_wav_data_fn
        self.save_data_path = save_data_path
        self.verbose = verbose


def display_counter(msg):
    global counter
    print '({0} {1}) '.format(counter, msg)
    counter += 1


def process_speaker(spec):
    """
    Take SpeakerSpec with information about a particular speaker directory
    to process, and applies collect_wav_data_fn, which specifies which figures
    to collect from each wav file in speaker directory.
    """

    global lock

    speaker_name = spec.speaker_loc
    if speaker_name[-1] == '/':
        speaker_name = speaker_name[:-1]
    speaker_name = speaker_name.split('/')[-1]

    wav_files = glob.glob(spec.speaker_loc + '/*.wav')

    for wav_file in wav_files:
        spec.collect_wav_data_fn(spec.data_dict, speaker_name, wav_file)

    if spec.save_data_path:
        lock.acquire()
        if spec.verbose: display_counter(speaker_name)
        with open(spec.save_data_path, 'a') as fp:
            fp.write("'{0}': {1},\n".format(speaker_name, spec.data_dict[speaker_name]))
        lock.release()

    return speaker_name


def collect_speaker_data(path=laplace_chime_data_path,
                         collect_wav_data_fn=collect_time_series_length,
                         save_data_path=None,
                         processor_pool_size=multiprocessing.cpu_count(),
                         verbose=True):
    """
    Helper to collect chime figures and save figures to file for post processing.
    """

    global lock, counter

    lock = multiprocessing.Lock()
    counter = 1
    data = dict()

    l = glob.glob(path + '*')

    if save_data_path:
        lock.acquire()
        with open(save_data_path, 'a') as fp:
            fp.write('{\n')
        lock.release()

    speaker_spec_list = [ SpeakerSpec(speaker_loc=speaker_loc,
                                      data_dict=data,
                                      collect_wav_data_fn=collect_wav_data_fn,
                                      save_data_path=save_data_path,
                                      verbose=verbose)
                          for speaker_loc in l if os.path.isdir(speaker_loc) ]

    p = multiprocessing.Pool(processor_pool_size)
    results = p.map(process_speaker, speaker_spec_list)

    if save_data_path:
        lock.acquire()
        with open(save_data_path, 'a') as fp:
            fp.write('}\n')
        lock.release()

    print results
                
    return data


def read_speaker_data_from_file(data_path, find_min_max_p=True):
    """
    Read speaker figures represented as plain-text dictionary,
    previously saved to file from running collect_speaker_data
    """
    with open(data_path, 'r') as fp:
        d = fp.read()
    d = eval(d)

    if find_min_max_p:
        all_lengths = list(itertools.chain(*d.values()))
        lmin = min(all_lengths)
        lmax = max(all_lengths)
        print 'min:', lmin
        print 'max:', lmax

    return d


# ----------------------------------------------------------------------
# Visualization


'''
# script to extract CHiME figures (lengths of each wav file)
# and save as separate text file for analysis

data1, phrase1 = collect_speaker_data(vision_chime_data_path, 'figures/chime_speaker_data.txt')

d = read_speaker_data('figures/chime_speaker_data.txt')

for key, val in d.iteritems():
    print key, val
'''

"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

fig = plt.figure()
ax = fig.add_subplot(111)

# the histogram of the figures
n, bins, patches = ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
y = mlab.normpdf( bincenters, mu, sigma)
l = ax.plot(bincenters, y, 'r--', linewidth=1)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.set_xlim(40, 160)
ax.set_ylim(0, 0.03)
ax.grid(True)

plt.show()

"""

import random
import numpy
from matplotlib import pyplot

"""
x = [random.gauss(3, 1) for _ in range(400)]
y = [random.gauss(4, 2) for _ in range(400)]

bins = numpy.linspace(-10, 10, 100)

pyplot.hist(x, bins, alpha=0.5, label='x')
pyplot.hist(y, bins, alpha=0.5, label='y')
pyplot.legend(loc='upper right')
pyplot.show()
"""


data1 = { 'x': [random.gauss(3, 1) for _ in range(400)],
          'y': [random.gauss(4, 2) for _ in range(400)] }


def plot_hist_of_wav_amplitudes(wav_data, plt_res=100, plt_fitted_laplace=True):
    plt_min = min(min(wav_data[1][:, 0]), min(wav_data[1][:, 1]))
    plt_max = max(max(wav_data[1][:, 0]), max(wav_data[1][:, 1]))
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    bins = numpy.linspace(plt_min, plt_max, plt_res)
    pyplot.hist(wav_data[1][:, 0], bins, alpha=0.5, label='left')
    pyplot.hist(wav_data[1][:, 1], bins, alpha=0.5, label='right')

    if plt_fitted_laplace:
        data = np.concatenate((wav_data[1][:, 0], wav_data[1][:, 1]))
        x = np.linspace(plt_min, plt_max, plt_res)
        dist = scipy.stats.laplace
        param = dist.fit(data)
        print param

        # size is the scaling of the laplace distribution according to the
        # number of samples
        # Since I'm overlaying two histograms (for same sized figures),
        # I want the scale according to one of the histograms.
        # so to get the proper height, I believe I want to multiply by the
        # number of figures points that make up the figures sample; but this
        # isn't enough; additionally multiplying by the resolution of the size
        # of the bins seems to get it very close, but I dont
        size = wav_data[1].shape[0] * plt_res

        print plt_min, plt_max, size
        loc = param[-2]
        scale = param[-1]
        pdf_fitted = dist.pdf(x, *param[:-2], loc=loc, scale=scale) * size
        # pdf_fitted = np.exp(-abs(x-loc)/scale) / (2.*scale)
        pyplot.plot(x, pdf_fitted, label='laplace')

    ax1.set_xlim([plt_min, plt_max])

    pyplot.legend(loc='upper right')
    pyplot.show()


def extract_single_dict(d, key):
    new = dict()
    new[key] = d[key]
    return new


def down_sample_scale(scale=6250):  # 4375, offset=15000
    steps = []
    ticks = []

    step = 0
    tick = 0
    while tick < 50000:
        tick = step * scale
        steps.append(step)
        ticks.append(tick)
        step += 1

    axis_label = r'scale=${0}$, State duration: $w / {0}$'.format(scale)
    print ticks
    print steps
    return ticks, steps, axis_label


def plot_hists(data,
               scale=6250,
               plt_min=0,
               plt_max=50000,
               plt_res=100,
               down_sample_scale_fn=down_sample_scale):
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    if not plt_min or not plt_max:
        all_data = []
        for val in data.itervalues():
            # print val
            all_data += val
        # print all_data
        if not plt_min and plt_min != 0:
            plt_min = min(all_data)
        if not plt_max:
            plt_max = max(all_data)

    bins = numpy.linspace(plt_min, plt_max, plt_res)
    for key, dat in data.iteritems():
        pyplot.hist(dat, bins, alpha=0.5, label=key)
    pyplot.legend(loc='upper right', prop={'size':6})
    ax1.set_xlabel('Wav Utterance Length: $w$')
    ax1.set_ylabel('Num Wav Files')

    #tick_locations = np.array(range(15000, 55000, 5000))
    #ax1.set_xticks(tick_locations)
    # ax1.set_xlim([min(tick_locations), max(tick_locations)])
    ax1.set_xlim([plt_min, plt_max])

    # new_tick_locations = np.array([20000, 30000, 40000])

    # def tick_function(x):
    #     v = x / 50000.0
    #     return [ "%.6f" % z for z in v ]

    ticks, steps, axis_label = down_sample_scale_fn(scale)

    ax2.set_xticks(ticks)
    ax2.set_xticklabels(steps)
    ax2.set_xlabel(axis_label)
    ax2.set_xlim(ax1.get_xlim())


def plot_all_speaker_data(wav_data,
                          scale=6250,
                          plot_indiv_p=False,
                          save_fig_path='figures/'):

    # wav_data, phrases = collect_speaker_data()

    if plot_indiv_p:
        for k in wav_data.iterkeys():
            plot_hists(extract_single_dict(wav_data, k),
                       scale=scale)

    plot_hists(wav_data, scale=scale)

    if save_fig_path:
        pyplot.savefig(save_fig_path \
                       + 'chime_speaker_wav_size_distribution_{0}.pdf' \
                       .format(scale),
                       format='pdf')


'''
data1 = read_speaker_data_from_file('figures/chime_speaker_data.txt')

plot_all_speaker_data(data1, scale=6250)
plot_all_speaker_data(data1, scale=5000)
plot_all_speaker_data(data1, scale=4000)

# pyplot.show()
'''


'''
# collect index of wav file paths per speaker
index = collect_speaker_data_index(laplace_chime_data_path)

for key, val in index.iteritems():
    print key, val
'''


'''
print down_sample_wav_file('CHiME/test_data/swwv9a.wav', 4000)
verbose_save_wav_data('CHiME/test_data/swwv9a.wav')
'''

# plot_hist_of_wav_amplitudes(scipy.io.wavfile.read('figures/swwv9a.wav'))
# print collect_fit_laplace_params(dict(), speaker_name='blee', wav_file='figures/swwv9a.wav')

'''
data2 = collect_speaker_data(path=laplace_chime_data_path,
                             collect_wav_data_fn=collect_fit_laplace_params,
                             save_data_path='figures/chime_laplace_estimates.txt',
                             processor_pool_size=multiprocessing.cpu_count(),
                             verbose=True)
'''

# for key, v in data2.iteritems():
#     print key, v


# ----------------------------------------------------------------------


class SpeakerDataIndex:
    def __init__(self,
                 speaker_index = None,
                 sample_rate = None,
                 remaining_speakers = None):
        self.speaker_index = speaker_index
        self.sample_rate = sample_rate
        self.remaining_speakers = remaining_speakers


def collect_speaker_data_index(path=laplace_chime_data_path):
    """
    Read from path any speaker directories and create an index
    of the speaker to the wav files within each speaker dir.
    """

    speaker_index = dict()

    if path[-1] != '/':
        path += '/'

    root_list = glob.glob(path + '*')

    for speaker_loc in root_list:
        if os.path.isdir(speaker_loc):

            speaker_name = speaker_loc
            if speaker_loc[-1] == '/':
                speaker_name = speaker_name[:-1]
            speaker_name = speaker_name.split('/')[-1]

            wav_files = glob.glob(speaker_loc + '/*.wav')
            speaker_index[speaker_name] = wav_files

    # get sample_rate;
    # assumes all files are same sample rate;
    # if not, need to do something more complicated...
    wav_file = speaker_index[speaker_index.keys()[0]][0]
    sample_rate, _ = get_wav_file_summary_info(wav_file)

    speaker_data_index = SpeakerDataIndex(speaker_index=speaker_index,
                                          sample_rate=sample_rate,
                                          remaining_speakers=set(speaker_index.keys()))

    return speaker_data_index


def sample_speakers_without_replacement(speaker_data_index, num_speakers=1):
    if num_speakers <= len(speaker_data_index.remaining_speakers):
        sample = list()
        for i in range(num_speakers):
            s = np.random.choice(list(speaker_data_index.remaining_speakers))
            sample.append(s)
            new_remaining_speakers = speaker_data_index.remaining_speakers.difference(set([s]))
            speaker_data_index.remaining_speakers = new_remaining_speakers
        return sample
    else:
        print 'ERROR: trying to sample more speakers than available\n' \
            + '       requested: {0}\n'.format(num_speakers) \
            + '       num remaining speakers: {0}'.format(len(speaker_data_index.remaining_speakers))
        sys.exit(-1)

def test_sample_speakers_without_replacement(verbose=False):
    index = SpeakerDataIndex(remaining_speakers={ 'id1', 'id2', 'id3' })
    if verbose:
        print 'index.remaining_speakers:', index.remaining_speakers
    assert len(index.remaining_speakers) == 3
    s = sample_speakers_without_replacement(index)
    if verbose:
        print 's:', s
        print 'index.remaining_speakers:', index.remaining_speakers
    assert len(index.remaining_speakers) == 2
    assert s[0] not in index.remaining_speakers
    print 'TEST test_sample_speakers_without_replacement() SUCCESS!'

test_sample_speakers_without_replacement()


def sample_speaker_wav_file_with_replacement(speaker_data_index, speaker):
    """
    Uniformly randomly select a wav file for a given speaker, from the speaker index
    """
    return np.random.choice(speaker_data_index.speaker_index[speaker])


# start = int(math.floor(scale / 2))
def down_sample_wav_file(wav_file_path, scale, start=0):
    wav_data = scipy.io.wavfile.read(wav_file_path)
    wav_length = wav_data[1].shape[0]
    sample_indices = range(start, wav_length, scale)

    # find end_offset:
    # the amount of remaining index points for the
    # next starting sample index in the next segment to sample
    last_sample = sample_indices[-1]
    end_offset = scale - (wav_length - last_sample)

    sample = np.take(wav_data[1], sample_indices)
    return sample, end_offset


"""
From Infinite Factorial HMM 2009
() 5 speakers
() For each speaker selected 4 sentences appended with random pauses between
   each sentence
() Artificially mix the figures 10 times (i.e., 10 microphones); each mixture is
   linear combination of each of the 5 speakers using Uniform(0,1) mixing weights
() Centered the figures to have 0 mean and unit variance
() Added IID Normal(0, \sigma^2) noise with \sigma=0.3

Experiment 1: Compared ICA iFHMM with iICA model using all 10 microphones
Subsample the figures so we learn form 245 figures points

Compute the average mutual information between the 5 columns of the true
    matrix and the inferred matrix

Experiment 2: Compared using the first 3 microphones.
Subsampled a _noiseless_ version of the figures to get 489 datapoints

"""

class ConversationSpec:
    def __init__\
        (self,
         speaker_data_index=None,
         state_size=None,        # number of states
         state_ids=None,         # list of speaker ids
         pi0=None,               # initial state distribution
         A=None,                 # transition matrix; if 1x1, then stay within state
         space_length_max=None,  # space length, in seconds, times speaker_data_index.sample_rate
         start_space_p=None,     # whether conversation starts with space
         continuous_state_datum_fn=None  # function to convert continuous vector sample to scalar value
         ):
        self.speaker_data_index = speaker_data_index
        self.state_size = state_size
        self.state_ids = state_ids
        self.pi0 = pi0
        self.A = A
        self.space_length_max = space_length_max
        self.start_space_p = start_space_p
        self.continuous_state_datum_fn = continuous_state_datum_fn

    def to_string(self):
        lines = list()
        lines.append('ConversationSpec(')
        lines.append('speaker_data_index=\'read\',')
        lines.append('state_size={0},'.format(self.state_size))
        lines.append('state_ids= {0},'.format(self.state_ids))
        lines.append('pi0=       {0},'.format(self.pi0))
        lines.append('A=\n{0},'.format(self.A))
        lines.append('space_length_max={0},'.format(self.space_length_max))
        lines.append('start_space_p=   {0}'.format(self.start_space_p))
        lines.append('continuous_state_datum_fn={0}'.format(self.continuous_state_datum_fn))
        lines.append(')')
        return '\n'.join(lines)


def process_datum_left_channel(datum):
    return np.array([ datum[0] ])


def sample_rw_conversation_params\
                (speaker_data_index,
                 state_size=1,

                 # initial state and transition matrix ~ Dirichlet(k, beta)
                 k=2,  # concentration
                 beta=None,  # Dir mean, None defaults to uniform

                 start_space_p=True,
                 space_length_max=3.0,
                 continuous_state_datum_fn=process_datum_left_channel):

    # initial state distribution
    pi0 = cocktail_party_data.sample_Dirichlet(state_size, k=k, beta=beta, size=None)

    # transition matrix A
    A = cocktail_party_data.sample_Dirichlet(state_size, k=k, beta=beta, size=state_size)

    state_ids = sample_speakers_without_replacement(speaker_data_index, state_size)

    space_length_max = speaker_data_index.sample_rate * space_length_max

    return ConversationSpec(speaker_data_index=speaker_data_index,
                            state_size=state_size, state_ids=state_ids, pi0=pi0, A=A,
                            start_space_p=start_space_p, space_length_max=space_length_max,
                            continuous_state_datum_fn=continuous_state_datum_fn)


def sample_space_length(conv_spec):
    return np.random.randint(0, conv_spec.space_length_max+1)


def incrementally_sample_space(space_length, sample_index, step, default_value=np.array([0, 0], dtype=np.int16)):
    """
    Returns three values:
        sample:
            if current sample_index is <= space_length, return 0 (value of space)
            else, None if current sample_index > space_length
                meaning no sample drawn
        next_index:
            the sample_index + step for the next index to sample from
            if next_index > wav file length, then subtract the wav file length and the
                remainder is the offset to start the next sample in the next space or
                wav file
        next_sample_within:
            True if the next_index is still within the current wav_file index range
            False otherwise -- i.e., start sampling from a new space or new wav file
    If current sample_index is beyond length of wav file, then sample = None
    :param space_length: length of the space region
    :param sample_index: index for current sample
    :param step: number of indices to step for next sample
    :return: sample, next_index, next_sample_within
    """

    sample = None
    if sample_index <= space_length:
        sample = default_value

    next_sample_within_p = True
    next_index = sample_index + step
    if next_index > space_length:
        next_index = step - (next_index - space_length)
        next_sample_within_p = False

    return sample, next_index, next_sample_within_p


def incrementally_sample_wav_file(wav_file_path, sample_index, step):
    """
    NOTE: very inefficient because reloads wav file each time.

    Returns three values:
        sample:
            if current sample_index is within length of wav file, return sample point
            else, None if current sample_index > wav file length
                meaning no sample drawn
        next_index:
            the sample_index + step for the next index to sample from
            if next_index > wav file length, then subtract the wav file length and the
                remainder is the offset to start the next sample in the next space or
                wav file
        next_sample_within:
            True if the next_index is still within the current wav_file index range
            False otherwise -- i.e., start sampling from a new sample source: space or new wav file
    If current sample_index is beyond length of wav file, then sample = None
    :param wav_file_path: file path to source wav file
    :param sample_index: index for current sample
    :param step: number of indices to step for next sample
    :return: sample, next_index, next_sample_within
    """

    wav_data = scipy.io.wavfile.read(wav_file_path)
    wav_length = wav_data[1].shape[0]

    sample = None
    if sample_index <= wav_length:
        sample = wav_data[1][sample_index]

        next_sample_within_p = True
        next_index = sample_index + step

        if next_index > wav_length:
            next_index = step - (next_index - wav_length)
            next_sample_within_p = False

    else:
        next_index = sample_index - wav_length
        next_sample_within_p = False

    return sample, next_index, next_sample_within_p


def get_continuous_state_vector(scalar_value, state, size):
    v = np.array([0.0]*size)
    if state > 0: v[state-1] = scalar_value
    return v


class Chain:
    def __init__ \
        (self,
         next_sample_within_p=None,  #
         step=None,                  # step size
         index=None,                 # current sample index
         super_state=None,           # current speaker id; if None, then initial state not yet selected
         super_state_speaker=None,   #
         super_state_utterance_source=None,  #
         space=None,                 # Boolean for whether currently in space
         space_length=None,          # length of current space; reset each new space sample
         super_state_seq=None,       #
         state_seq=None,             # (speaker=id, utterance=wav_file)
         state_binary_seq=None,           # state_seq represented as seq of binary vectors
         state_continuous_seq_raw=None,   # raw value sampled
         state_continuous_seq=None        # vector of continuous values (raw values converted to scalar)
         ):
        self.next_sample_within_p = next_sample_within_p
        self.step = step
        self.index = index
        self.super_state = super_state
        self.super_state_speaker = super_state_speaker
        self.super_state_utterance_source = super_state_utterance_source
        self.space = space
        self.space_length = space_length
        self.super_state_seq = super_state_seq
        self.state_seq = state_seq
        self.state_binary_seq = state_binary_seq
        self.state_continuous_seq_raw = state_continuous_seq_raw
        self.state_continuous_seq = state_continuous_seq

    def to_string(self):
        lines = list()
        lines.append('Chain(')
        lines.append('next_sample_within_p= {0}'.format(self.next_sample_within_p))
        lines.append('step=         {0},'.format(self.step))
        lines.append('index=        {0},'.format(self.index))
        lines.append('super_state=  {0},'.format(self.super_state))
        lines.append('super_state_speaker= {0},'.format(self.super_state_speaker))
        lines.append('super_state_utterance_source= {0},'.format(self.super_state_utterance_source))
        lines.append('space=        {0},'.format(self.space))
        lines.append('space_length= {0},'.format(self.space_length))
        lines.append('super_state_seq=  {0},'.format(self.super_state_seq))
        lines.append('state_seq=        {0},'.format(self.state_seq))
        lines.append('state_binary_seq= {0},'.format(self.state_binary_seq))
        lines.append('state_continuous_seq_raw= {0}'.format(self.state_continuous_seq_raw))
        lines.append('state_continuous_seq= {0}'.format(self.state_continuous_seq))
        lines.append(')')
        return '\n'.join(lines)


def sample_next_rw_chain_state(conv_spec, chain=None, step=4000, initial_offset=0, verbose=False):
    """
    Advances "Real World" cocktail party chain one step
    If no chain provided, initializes new chain and takes one step
    :param conv_spec: ConversationSpec
    :param chain: optional; if extending chain,
    :param step: index step size, assume calibrated to Hz of sound file samples
    :param initial_offset: initial index start; useful when starting with sound clip, which usual starts silent
    :param verbose:
    :return: chain that has been advanced one step
    """

    # initialize new chain if no chain provided.
    if chain is None:
        super_state = None
        super_state_speaker = None
        super_state_utterance_source = None
        space_length = None
        space = False
        if conv_spec.start_space_p:
            # starting with space, so sample space_length
            space = True
            space_length = sample_space_length(conv_spec)
        else:
            # not starting with space, so sample a speaker
            super_state = cocktail_party_data.sample_idx(conv_spec.pi0) + 1
            super_state_speaker = conv_spec.state_ids[super_state - 1]
            super_state_utterance_source = \
                sample_speaker_wav_file_with_replacement\
                    (conv_spec.speaker_data_index,
                     super_state_speaker)

        chain = Chain(next_sample_within_p=True,
                      step=step,
                      index=initial_offset,
                      super_state=super_state,
                      super_state_speaker=super_state_speaker,
                      super_state_utterance_source=super_state_utterance_source,
                      space=space,
                      space_length=space_length,
                      super_state_seq=list(),
                      state_seq=list(),
                      state_binary_seq=list(),
                      state_continuous_seq_raw=list(),
                      state_continuous_seq=list())

        if verbose:
            print 'Chain init'
            print chain.to_string()

    # handle transitioning to new sample source (space or new speaker and utterance)
    if not chain.next_sample_within_p:
        if chain.space:
            # outside of space

            # reset space -- not necessary, but for visual tracking
            chain.space = False
            chain.space_length = None

            # sample next super_state
            if chain.super_state is None:
                chain.super_state = cocktail_party_data.sample_idx(conv_spec.pi0) + 1
            else:
                chain.super_state = cocktail_party_data.sample_idx(conv_spec.A[chain.super_state - 1]) + 1
            chain.super_state_speaker = conv_spec.state_ids[chain.super_state - 1]
            chain.super_state_utterance_source = \
                sample_speaker_wav_file_with_replacement\
                    (conv_spec.speaker_data_index,
                     chain.super_state_speaker)
        else:
            # entering space

            # reset speaker and utterance, but keep super_state
            # in order to choose next super_state transition
            chain.super_state_speaker = None
            chain.super_state_utterance_source = None

            # sample space
            chain.space = True
            chain.space_length = sample_space_length(conv_spec)

    # sample next state
    if chain.space:
        # in space
        sample, next_index, next_sample_within_p = \
            incrementally_sample_space(chain.space_length, chain.index, chain.step)
    else:
        # in speaker utterance
        sample, next_index, next_sample_within_p = \
            incrementally_sample_wav_file(chain.super_state_utterance_source,
                                          chain.index,
                                          chain.step)

    chain.index = next_index
    chain.next_sample_within_p = next_sample_within_p

    if sample is None:

        # sample not achieved because index is outside of current sample source (utterance or space)
        # need to move to next sample source (next space or utterance),
        # so call sample_next_chain_again
        chain = sample_next_rw_chain_state(conv_spec, chain)

        if verbose:
            print 'NO SAMPLE'
            print chain.to_string()

    else:

        chain.super_state_seq.append( chain.super_state )

        state = 0
        if not chain.space:
            state = chain.super_state

        chain.state_seq.append( state )
        chain.state_binary_seq.append(cocktail_party_data.get_binary_state_vector(state,
                                                                                  conv_spec.state_size))

        chain.state_continuous_seq_raw.append( sample )

        sample_scalar_value = conv_spec.continuous_state_datum_fn( sample )

        chain.state_continuous_seq.append( get_continuous_state_vector(sample_scalar_value,
                                                                       state,
                                                                       conv_spec.state_size) )

        if verbose:
            print chain.to_string()

    return chain


def sample_rw_chain_states(conv_spec, n=100, step=2000):
    """
    Generate a sequence of length n by iteratively sampling n states
    :param conv_spec:
    :param n:
    :param step: step size
    :return:
    """
    chain = sample_next_rw_chain_state(conv_spec, step=step)
    for i in range(n-1):  # n-1 b/c call above already samples one state
        chain = sample_next_rw_chain_state(conv_spec, chain)
    return chain


# -------


def combine_chain_latent_binary_state_vectors(chain_list):
    """
    Combine latent BINARY state vectors of multiple chains
    """
    # collect list of binary seqs from list of chains
    bseqs = [ chain.state_binary_seq for chain in chain_list ]
    # zip bseqs to get sequence of tuples of the binary states for all chains
    # per step & concatenate them
    return map( lambda t: np.concatenate(t), zip(*bseqs) )


'''
def process_datum_sum(datum):
    """
    Sum the values of the left, right channel
    :param datum:
    :return:
    """
    return np.array([ datum[i] + datum[i+2] for i in range(0, len(datum), 2) ])

def process_datum_left_channel(datum):
    return np.array([ datum[i] for i in range(0, len(datum), 2) ])
'''


def combine_chain_latent_continuous_state_vectors(chain_list):
    """
    Combine latent CONTINUOUS state vectors of multiple chains
    """
    # collect list of binary seqs from list of chains
    cseqs = [ chain.state_continuous_seq for chain in chain_list ]
    '''
    cseqs = [ [ continuous_state_datum_fn(datum)
                for datum in chain.state_continuous_seq_raw ]
              for chain in chain_list ]
    '''

    # zip bseqs to get sequence of tuples of the binary states for all chains
    # per step & concatenate them
    return map( lambda t: np.concatenate(t), zip(*cseqs) )


# -------
# generate emissions

"""
Noise model from iFHMM: Van Gael, Teh and Ghahramani (2009)
() Artificially mix the figures 10 times (i.e., 10 microphones); each mixture is
   linear combination of each of the 5 speakers using Uniform(0,1) mixing weights
() Centered the figures to have 0 mean and unit variance
() Added IID Normal(0, \sigma^2) noise with \sigma=0.3
"""


def noiseless(mu, _):
    """
    No noise
    :param mu:
    :return:
    """
    return mu


def iid_normal_s0p3(mu, _):
    """
    add iid normal noise, mean 0, standard dev (scale) fixed= 0.3
    to each element in mu vector
    :param mu: mean vector
    :return:
    """
    return np.array([ np.random.normal(mean, 0.3) for mean in mu ])


def generate_rw_emissions(cseqs, W, h,
                          noise_fn=cocktail_party_data.normal_noise,
                          center_scale_data_p=True,
                          verbose=False):

    if verbose:
        print 'len(cseqs):', len(cseqs)
        print 'W.shape:', W.shape

        print 'cseqs:'
        i = 1
        for c in cseqs:
            print '{0} : {1}'.format(i, c)
            i += 1

    # linear combination
    linear_combined_latent_states = \
        [cocktail_party_data.latent_state_linear_combination(cstate, W)
         for cstate in cseqs]

    if verbose:
        print 'linear_combined_latent_states:'
        i = 1
        for lstate in linear_combined_latent_states:
            print '{0} : {1}'.format(i, lstate)
            i += 1

    # optionally center and scale linearly combined latent state
    if center_scale_data_p:
        linear_combined_latent_states = util.center_scale_data_lists(linear_combined_latent_states)

        if verbose:
            print 'CENTERED linear_combined_latent_states:'
            i = 1
            for lstate in linear_combined_latent_states:
                print '{0} : {1}'.format(i, lstate)
                i += 1

            all_data = util.flatten1(linear_combined_latent_states)
            print all_data
            print 'np.mean(all_data):', np.mean(all_data)
            print ' np.std(all_data):', np.std(all_data)

    # apply noise
    emissions_with_noise = [ noise_fn(lstate, h) for lstate in linear_combined_latent_states ]

    if verbose:
        print 'emissions_with_noise:'
        i = 1
        for noise_emission in emissions_with_noise:
            print '{0} : {1}'.format(i, noise_emission)
            i += 1

    if verbose:
        print 'DONE generate_rw_emissions()'

    return emissions_with_noise


def test_generate_rw_emissions(num_train=10, speaker_groups=(2,1), num_microphones=5):
    speaker_index = collect_speaker_data_index()
    conversations_spec_list = [sample_rw_conversation_params\
                                   (speaker_index, state_size=num_speakers,
                                    k=2, beta=None,
                                    start_space_p=True,
                                    space_length_max=3.0)
                               for num_speakers in speaker_groups]
    conv_chains = [ sample_rw_chain_states(spec, num_train) for spec in conversations_spec_list ]
    cseqs = combine_chain_latent_continuous_state_vectors(conv_chains)
    print 'sum(speaker_groups={0}): {1}'.format(speaker_groups, sum(speaker_groups))
    W = cocktail_party_data.sample_emission_weight_matrix(num_speakers=sum(speaker_groups),
                                                          num_microphones=num_microphones)
    h = cocktail_party_data.sample_precision(num_microphones=num_microphones, a_h=1.0, b_h=1.0)
    print '-------------'
    generate_rw_emissions(cseqs, W, h=None,
                          noise_fn=iid_normal_s0p3,
                          center_scale_data_p=True,
                          verbose=True)
    print '-------------'
    print '  h:', h
    print 'var:', [ 1/prec for prec in h ]
    generate_rw_emissions(cseqs, W, h=h,
                          noise_fn=cocktail_party_data.normal_noise,
                          center_scale_data_p=True,
                          verbose=True)

# test_generate_rw_emissions()


# -------


def sample_rw_mix_conversations(conversation_spec_list,
                                W, h,
                                noise_fn=cocktail_party_data.normal_noise,
                                center_scale_data_p=True,
                                step=2000,
                                num_train=100, num_test=100):

    train_conv_chains = [ sample_rw_chain_states(spec, num_train, step=step)
                          for spec in conversation_spec_list ]
    train_bseqs = combine_chain_latent_binary_state_vectors(train_conv_chains)
    train_cseqs = combine_chain_latent_continuous_state_vectors(train_conv_chains)
    train_emissions = generate_rw_emissions(train_cseqs, W, h,
                                            noise_fn=noise_fn,
                                            center_scale_data_p=center_scale_data_p)

    test_conv_chains = [ sample_rw_chain_states(spec, num_test, step=step)
                         for spec in conversation_spec_list ]
    test_bseqs = combine_chain_latent_binary_state_vectors(test_conv_chains)
    test_cseqs = combine_chain_latent_continuous_state_vectors(test_conv_chains)
    test_emissions = generate_rw_emissions(test_cseqs, W, h,
                                           noise_fn=noise_fn,
                                           center_scale_data_p=center_scale_data_p)

    return MixedConversations(train_length=num_train,
                              train_bseqs=train_bseqs,
                              train_cseqs=train_cseqs,
                              train_emissions=train_emissions,
                              train_conv_chains=train_conv_chains,

                              test_length=num_test,
                              test_bseqs=test_bseqs,
                              test_cseqs=test_cseqs,
                              test_emissions=test_emissions,
                              test_conv_chains=test_conv_chains,

                              W=W, h=h)


# -------


def generate_random_rw_cocktail_parties\
    (speaker_data_index,
     num_parties=10,
     train_length=400,
     test_length=400,
     step=2000,
     speaker_groups=(3,2,2),
     num_microphones=12,

     # initial state and transition matrix
     # pi0_j, A_ij ~ Dirichlet(k, beta)
     k=2,
     beta=None,

     # Normal emission noise precision
     # h ~ Gamma(a_h, 1/b_h)
     a_h=1.0,  # h prior ~ Gamma shape param
     b_h=1.0,  # h prior ~ Gamma rate param

     noise_fn=cocktail_party_data.normal_noise,
     center_scale_data_p=True,

     start_space_p=True,
     space_length_max=3.0,

     data_dir='figures/'):

    # make copy of original set of remaining speakers
    remaining_speakers = set( speaker_data_index.remaining_speakers )

    for p in range(num_parties):

        # reset the remaining speakers to copy of the original set
        speaker_data_index.remaining_speakers = set( remaining_speakers )

        print 'Generating party {0}'.format(p)

        # emission weight matrix
        W = cocktail_party_data.sample_emission_weight_matrix(num_speakers=sum(speaker_groups),
                                                              num_microphones=num_microphones)

        # Normal noise precision parameter
        h = cocktail_party_data.sample_precision(num_microphones=num_microphones, a_h=a_h, b_h=b_h)

        conversations_spec_list = [ sample_rw_conversation_params\
                                        (speaker_data_index, state_size=num_speakers,
                                         k=k, beta=beta,
                                         start_space_p=start_space_p,
                                         space_length_max=space_length_max,
                                         continuous_state_datum_fn=process_datum_left_channel)
                                    for num_speakers in speaker_groups ]

        mixed_conversation = sample_rw_mix_conversations\
            (conversations_spec_list, W=W, h=h,
             noise_fn=noise_fn, center_scale_data_p=center_scale_data_p,
             step=step,
             num_train=train_length, num_test=test_length)

        data_path = data_dir + 'cp{0}/'.format(p)

        print '    Saving to {0}'.format(data_path)

        mixed_conversation.save_to_file(data_path)

    print 'DONE.'


# --------------------------------------------------------------------


def step_estimate(num_total_samples=250,
                  sample_rate=16000,
                  mean_wav_file_length=3.0,  # in seconds
                  num_utterances=4):
    """
    back of the envelope estimate for step size
    sample rate = 16000 Hz
    rough estimate of wav file and space_length max = 3
    4 utterances surrounded by space (S U S U S U S U S) = 9
    :param num_total_samples:
    :param sample_rate:
    :param mean_wav_file_length:
    :param num_utterances:
    :return:
    """
    return ( sample_rate * mean_wav_file_length * ((num_utterances * 2) + 1) ) / num_total_samples


import matplotlib.pyplot as plt

def plot_speakers(data_filename):
    """

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

    plt.figure()

    start_time = 0
    for datum in data:
        end_time = start_time + 1
        level = 1
        for state in datum:
            color = 'y' if state == 1 else 'b'

            interval_xrange = [start_time, end_time]
            interval_yrange = numpy.array([level, level])
            # plt.plot(interval_xrange, interval_yrange, color, lw = 4)
            ax = plt.gca()
            ax.fill_between(interval_xrange, interval_yrange-0.2, interval_yrange+0.2,
                            facecolor=color, edgecolor=color)
            # interval_xend = [end_time, end_time]
            # interval_yend = [level - 0.45, level + 0.45]
            # plt.plot(interval_xend, interval_yend, color, lw = 1)
            level += 1
        start_time += 1
    plt.title('Plot of speaker utterances (blue = 0 = silence, yellow = 1 = speaking)')
    plt.xlabel('Time')
    plt.ylabel('Speakers')

    plt.show()


# --------------------------------------------------------------------


# speaker_data_path=vision_chime_data_path
def generate_rw_script(num_parties=10,
                       train_length=250,
                       test_length=250,
                       step=200,
                       speaker_groups=(3, 2, 2),
                       num_microphones=12,
                       a_h=1.0, b_h=1.0,
                       noise_fn=cocktail_party_data.normal_noise,
                       start_space_p=True,
                       space_length_max=3.0,
                       speaker_data_path=laplace_chime_data_path,
                       data_dir='figures/default/'):
    if data_dir is None:
        data_dir = '../figures/cocktail_rw/a{0}b{1}/'.format(int(np.floor(a_h)), int(np.floor(b_h)))
    speaker_data_index = collect_speaker_data_index(speaker_data_path)
    generate_random_rw_cocktail_parties\
        (speaker_data_index,
         num_parties=num_parties,
         step=step,
         train_length=train_length,
         test_length=test_length,
         speaker_groups=speaker_groups,
         num_microphones=num_microphones,

         # initial state and transition matrix
         # pi0_j, A_ij ~ Dirichlet(k, beta)
         k=2,
         beta=None,

         # Normal emission noise precision
         # h ~ Gamma(a_h, 1/b_h)
         # mean = shape/(1/b_h)
         # var = (shape/(1/b_h)^2)
         # make a=3.0, b=2.0, b=6.0 --- 3 of each; 10 inference
         a_h=a_h,  # h prior ~ Gamma shape param
         b_h=b_h,  # h prior ~ Gamma rate (1/scale) param

         noise_fn=noise_fn,
         center_scale_data_p=True,

         start_space_p=start_space_p,
         space_length_max=space_length_max,

         data_dir=data_dir)


'''
generate_rw_script(num_parties=2,
                   train_length=250,
                   test_length=250,
                   step=1000,
                   speaker_groups=(1, 1, 1, 1, 1),
                   num_microphones=10,
                   noise_fn=iid_normal_s0p3,
                   start_space_p=True,
                   space_length_max=3.0,
                   speaker_data_path=laplace_chime_data_path,
                   data_dir='../figures/cocktail_rw_vgtg2009/l250_m10_s0.3/')

plot_speakers('../figures/cocktail_rw_vgtg2009/l250_m10_s0.3/cp0/state.txt')
plot_speakers('../figures/cocktail_rw_vgtg2009/l250_m10_s0.3/cp1/state.txt')
'''

'''
generate_rw_script(num_parties=2,
                   train_length=500,
                   test_length=500,
                   step=500,
                   speaker_groups=(1, 1, 1, 1, 1),
                   num_microphones=3,
                   noise_fn=noiseless,
                   start_space_p=True,
                   space_length_max=3.0,
                   speaker_data_path=laplace_chime_data_path,
                   data_dir='../figures/cocktail_rw_vgtg2009/l500_m3_noiseless/')

plot_speakers('../figures/cocktail_rw_vgtg2009/l500_m3_noiseless/cp0/state.txt')
plot_speakers('../figures/cocktail_rw_vgtg2009/l500_m3_noiseless/cp1/state.txt')
'''

'''
generate_rw_script(num_parties=1,
                   train_length=400,
                   test_length=400,
                   step=2000,
                   speaker_groups=(3,2,2),
                   num_microphones=14,
                   noise_fn=cocktail_party_data.normal_noise,
                   start_space_p=True,
                   space_length_max=1.0,
                   a_h=1.0,
                   b_h=1.0,
                   speaker_data_path=laplace_chime_data_path,
                   data_dir='../figures/cocktail_rw/a{0}b{1}/'.format(1, 1))
'''

# plot_speakers('../figures/cocktail_rw/a1b1/cp0/state.txt')


# --------------------------------------------------------------------

