import scipy.io.wavfile
import glob
import sys
import os
import urllib
import zipfile
import audioop
import numpy
import math
import random
import collections
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Ensure <hamlet>/experiment/scripts/python/ in sys.path  (if possible)
# ----------------------------------------------------------------------

def seq_equal(s1, s2):
    if len(s1) != len(s2):
        return False
    for e1, e2 in zip(s1, s2):
        if e1 != e2:
            return False
    return True


def print_sys_path():
    for i, path in enumerate(sys.path):
        print i, path


def find_path_context(target_path_components):
    for path in sys.path:
        path_components = path.split('/')
        if 'hamlet' in path_components:
            if seq_equal(target_path_components,
                         path_components[-len(target_path_components):]):
                return True
    return False


def optional_add_relative_path(current, parent, relative_path, verbose=False):
    """
    If executing in current directory and parent path is not in sys.path,
    then add parent path.
    :return:
    """
    if not find_path_context(parent):
        if find_path_context(current):
            parent_path = os.path.realpath(os.path.join(os.getcwd(), relative_path))
            if verbose:
                print 'NOTICE: experiment_tools.py'
                print '    Executing from:     {0}'.format(os.getcwd())
                print '    Adding to sys.path: {0}'.format(parent_path)
            sys.path.insert(1, parent_path)


optional_add_relative_path\
    (current=('scripts', 'python', 'experiments', 'cocktail_party', 'experiment'),
     parent=('scripts', 'python'),
     relative_path='../../../',
     verbose=True)


from run import experiment_tools
from visualization import plot_binary_vector_time_series as plot_cp


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
DATA_ROOT = experiment_tools.DATA_ROOT
PARAMETERS_ROOT = experiment_tools.PARAMETERS_ROOT
RESULTS_ROOT = experiment_tools.RESULTS_ROOT


# ----------------------------------------------------------------------

SSC1_URL = 'http://laslab.org/SpeechSeparationChallenge'
# http://laslab.org/SpeechSeparationChallenge/
# 34 subjects, 500 utterances per subject


# ----------------------------------------------------------------------
# Right now paths are specific to laplace

SSC1_DATA_ROOT = '/Users/clayton/Documents/repository/data/Speech_Separation_Challenge_1/data'

CHIME_DATA_ROOT = '/Users/clayton/Documents/repository/data/CHiME/PCCdata16kHz/train/reverberated/'


# ----------------------------------------------------------------------
# Retrieve source SSC1 data from the web and unzip
# ----------------------------------------------------------------------

def retrieve_and_unzip_ssc1_files(dest_root=SSC1_DATA_ROOT):

    cwd = os.getcwd()
    os.chdir(dest_root)

    dest_zip_dir = 'data'
    if not os.path.exists(dest_zip_dir):
        os.makedirs(dest_zip_dir)

    for i in range(1, 35):
        url = '{0}/{1}.zip'.format(SSC1_URL, i)
        filename = '{0}.zip'.format(i)
        print i, url, filename
        ret = urllib.urlretrieve(url, filename)
        print '    ', ret
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(dest_zip_dir)
        zip_ref.close()
        print '    Done unzipping'
    os.chdir(cwd)

# retrieve_and_unzip_ssc1_files()


# ----------------------------------------------------------------------
# Analyze properties of source wav files
# ----------------------------------------------------------------------


def plot_wav_file_data_density(wav_file, resolution=100):
    sample_rate, data = scipy.io.wavfile.read(wav_file)
    data_min = numpy.min(data)
    data_max = numpy.max(data)
    bins = numpy.linspace(data_min, data_max, resolution)
    plt.hist(data, bins, alpha=0.5)
    plt.show()

# plot_wav_file_data_density(os.path.join(SSC1_DATA_ROOT, '1/bbaf2n.wav'))

# In general, the frequency distribution of values looks very Laplacian.


# ----------------------------------------------------------------------

def get_wav_file_summary_info(wav_file, verbose=False):
    """
    Collect
    scipy.io.wavfile.read() returns:
        sample rate (int: in samples/sec)
        data (if multichannel, will be a 2d array)
    :param wav_file: .wav file
    :param verbose: (default False) print summary statistics
    :return: sample_rate, data length, data min value, data max value
    """
    sample_rate, data = scipy.io.wavfile.read(wav_file)
    length = data.shape[0]

    if data.dtype == numpy.int8:
        b = 1
    elif data.dtype == numpy.int16:
        b = 2
    elif data.dtype == numpy.int32:
        b = 4
    else:
        b = None

    if verbose:
        print 'data dtype:', data.dtype
        print 'bytes:', b
        print 'sample rate:', sample_rate
        print 'length:', length
        print 'duration (seconds):', float(data.shape[0]) / float(sample_rate)
        print 'min value:', min(data)
        print 'max value:', max(data)
        if b is not None:
            print 'average:', audioop.avg(data, b)
            print 'root-mean-square (power):', audioop.rms(data, b)
            print 'num zero crossings:', audioop.cross(data, b)

    return sample_rate, length, min(data), max(data)


def test_get_wav_file_summary_info():
    stats = get_wav_file_summary_info(os.path.join(SSC1_DATA_ROOT, '1/bbaf2n.wav'),
                                      verbose=True)
    print stats

# test_get_wav_file_summary_info()


def get_wav_set_stats(data_root, verbose=False):
    print '{0}'.format(data_root)
    stats = numpy.array([get_wav_file_summary_info(filepath)
                         for filepath in glob.glob(os.path.join(data_root, '*.wav'))])
    sample_rates = set(stats[:, 0])
    all_data_mean_len = numpy.mean(stats[:, 1])
    all_data_min_len = numpy.min(stats[:, 1])
    all_data_max_len = numpy.max(stats[:, 1])
    all_val_min = numpy.min(stats[:, 2])
    all_val_max = numpy.max(stats[:, 3])

    if verbose:
        print 'sample_rates', sample_rates
        print 'all_data_mean_len', all_data_mean_len
        print 'all_data_min_len', all_data_min_len
        print 'all_data_max_len', all_data_max_len
        print 'all_val_min', all_val_min
        print 'all_val_max', all_val_max

    return sample_rates, all_data_mean_len, all_data_min_len, all_data_max_len, all_val_min, all_val_max


def get_stats_for_all_speakers(data_root, verbose=False):
    stats = [get_wav_set_stats(speaker_path)
             for speaker_path in glob.glob(os.path.join(data_root, '*'))]

    all_data_mean_len = numpy.mean([val[1] for val in stats])
    all_data_min_len = numpy.min([val[2] for val in stats])
    all_data_max_len = numpy.max([val[3] for val in stats])
    all_val_min = numpy.min([val[4] for val in stats])
    all_val_max = numpy.max([val[5] for val in stats])

    if verbose:
        print 'all_data_mean_len', all_data_mean_len
        print 'all_data_min_len', all_data_min_len
        print 'all_data_max_len', all_data_max_len
        print 'all_val_min', all_val_min
        print 'all_val_max', all_val_max


# get_stats_for_all_speakers(SSC1_DATA_ROOT, verbose=True)

'''
all_data_mean_len 45993.5882353
all_data_min_len 27751
all_data_max_len 72001
all_val_min -28344
all_val_max 28604
'''


# ----------------------------------------------------------------------

CP0_ROOT = os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT), 'cocktail_s16_m12/h10.0_nocs/cp0')

# plot_cp.plot_cp_data(path=os.path.join(CP0_ROOT, 'states.txt'))


# ----------------------------------------------------------------------
# Sample speakers and sentences
# ----------------------------------------------------------------------

def sample_dir(data_root, n, glob_pattern='*', replace=False):
    """
    A general helper for sampling with or without replacement from
    contents in a directory.
    NOTE: Be sure to be clear about whether sampling WITH *OR* WITHOUT replacement
    :param data_root: path to directory whose contents are being sampled
    :param n: size of sample
    :param glob_pattern: glob pattern for matching names in directory
    :param replace: flag to control whether to sample with replacement
    :return: tuple of tuples of filename and full path
    """
    population = glob.glob(os.path.join(data_root, glob_pattern))

    # sample indices without replacement
    indices = numpy.random.choice(len(population), n, replace=replace)

    paths = numpy.take(population, indices)

    return tuple([(os.path.basename(path), path) for path in paths ])


def test_sample_dir():
    print 'speaker samples:'
    for pair in sample_dir(SSC1_DATA_ROOT, 5, glob_pattern='*', replace=False):
        print pair

    print '-----------\nwav samples for speaker {0}:'.format(pair[0])
    for pair in sample_dir(pair[1], 5, glob_pattern='*.wav', replace=True):
        print pair

# test_sample_dir()


# ----------------------------------------------------------------------
# Generate conversation_spec
# ----------------------------------------------------------------------

'''
From iFHMM 2009:

Key to our presentation is that we want to illustrate that using nonparametric
methods, we can learn the number of speakers from a small amount of data.

5 speakers, 4 sentences each, appended with random pauses between each sentence
Artificially mix the data 10 times (i.e., 10 microphones: 5x10 mixture matrix)
    each mixture is linear combination of each of the 5 speakers using
        Uniform(0,1) mixing weights
    Centered the data to have zero mean and unit variance
    Added IID Normal(0, s^2) with s = 0.3

Experiments:
(1) using all 10 microphones
    subsample data to learn from 245 datapoints
(2) perform blind speach separation using only the first 3 microphones
    subsample noiseless version of data got get 489 datapoints

'''


# ----------------------------------------------------------------------

def array_to_tuple(a):
    if len(a.shape) == 1:
        return tuple(a)
    elif len(a.shape) == 2:
        rlist = list()
        for r in range(a.shape[0]):
            rlist.append(tuple(a[r, :]))
        return tuple(rlist)
    else:
        print '[ERROR] array_to_tuple(): array shape unsupported: {0}'.format(a.shape)
        sys.exit()


def test_array_to_tuple():
    a = numpy.array([[0.24924348, 0.15152109, 0.35768309, 0.24155234],
                     [0.34813172, 0.16677347, 0.16901248, 0.31608233],
                     [0.1474758, 0.22318591, 0.33383782, 0.29550047],
                     [0.66266516, 0.13494895, 0.17794056, 0.02444533]])
    print 'a\n', a
    print 'array_to_tuple(a)\n', array_to_tuple(a)

    b = numpy.array([0.09394473,  0.1122789 ,  0.47209671,  0.32167966])
    print 'b\n', b
    print 'array_to_tuple(b)\n', array_to_tuple(b)

# test_array_to_tuple()


# ----------------------------------------------------------------------

def sample_idx(pvec):
    """
    Sample index according to weights assigned to each index
    :param pvec: vector of weights
    :return: index (int)
    """
    r = random.random()
    return next((i for i, v in enumerate(numpy.cumsum(pvec)) if r <= v))


def sample_Dirichlet(dim=4, k=2.0, beta=None, size=None):
    """
    Wrapper to parameterize Dirichlet like Hamlet
    :param dim: number of dimensions of Dir vector (default 3)
    :param k: concentration (default 2.0)
    :param beta: mean vector (default uniform)
    :param size: number of samples
    :return: size-many samples from Dirichlet
    """
    if beta is None:
        beta = numpy.array([1.0 / dim for i in range(dim)])
    alpha = numpy.array([k * len(beta) * b for b in beta])
    return numpy.random.dirichlet(alpha, size=size)


def sample_conversation_params(state_size=4,
                               k=2.0,
                               beta=None):

    # initial state distribution
    pi0 = sample_Dirichlet(state_size, k=k, beta=beta, size=None)

    # transition matrix A
    A = sample_Dirichlet(state_size, k=k, beta=beta, size=state_size)

    return dict(pi0=pi0, A=A)


def sample_emission_weight_matrix(num_speakers, num_microphones, bias=1):
    return numpy.random.uniform(0, 1, size=( num_speakers + bias, num_microphones ))


def sample_emission_noise_matrix(obs_length, num_microphones, noise_sd=0.3):
    return numpy.random.normal(loc=0, scale=noise_sd, size=(obs_length, num_microphones))


def test_sample_emission_noise_matrix():
    m = sample_emission_noise_matrix(obs_length=200, num_microphones=12, noise_sd=0.3)
    print 'mean(m)', numpy.mean(m)
    print 'std(m)', numpy.std(m)

# test_sample_emission_noise_matrix()


class CocktailPartySpec(object):
    def __init__(self,

                 speaker_dir_root,

                 sample_rate=25000,  # sample pts per second

                 speaker_groups=(4, 4, 4, 4),
                 num_microphones=12,

                 initial_space_interval=46000,  # TODO based on average length of utterances
                 spacing_mu=0.25,     # mean
                 spacing_sigma=0.25,  # standard deviation

                 # initial state and transition matrix
                 # pi0, A_i* ~ Dirichlet(k, beta)
                 k=2,  # concentration
                 beta=None,  # Dir mean; None defaults to uniform

                 # hard-coded noise standard deviation for standard Normal, same for each microphone
                 noise_sd=0.3  # in terms of precision: math.pow(0.3, -2) = 11.1111...
                 ):

        self.speaker_dir_root = speaker_dir_root

        self.sample_rate = sample_rate
        self.speaker_groups = speaker_groups
        self.num_microphones = num_microphones

        # spacing parameters
        self.initial_space_interval = initial_space_interval
        self.spacing_mu = spacing_mu
        self.spacing_sigma = spacing_sigma

        self.k = k
        self.beta = beta

        '''
        if isinstance(noise_sd, (list, tuple)):
            if len(noise_sd) != num_microphones:
                print 'ERROR: length of noise_sd is {0} != num microphones {1}'\
                    .format(len(noise_sd), num_microphones)
                sys.exit()
            else:
                noise_sd = [float(noise_sd)] * num_microphones
        self.noise_sd = numpy.array(noise_sd)
        '''

        self.noise_sd = noise_sd

    def to_string(self):
        s = list()
        s.append('# CocktailPartySpec start')
        s.append('speaker_dir_root {0}'.format(self.speaker_dir_root))
        s.append('sample_rate {0}'.format(self.sample_rate))
        s.append('speaker_groups {0}'.format(self.speaker_groups))
        s.append('num_microphones {0}'.format(self.num_microphones))
        s.append('initial_space_interval {0}'.format(self.initial_space_interval))
        s.append('spacing_mu {0}'.format(self.spacing_mu))
        s.append('spacing_sigma {0}'.format(self.spacing_sigma))
        s.append('k {0}'.format(self.k))
        s.append('beta {0}'.format(self.beta))
        s.append('noise_sd {0}'.format(self.noise_sd))
        s.append('# CocktailPartySpec end')

        return '\n'.join(s)

    def show(self):
        print '<CocktailPartySpec>:'
        print '    speaker_dir_root', self.speaker_dir_root
        print '    sample_rate', self.sample_rate
        print '    speaker_groups', self.speaker_groups
        print '    num_microphones', self.num_microphones
        print '    initial_space_interval', self.initial_space_interval
        print '    spacing_mu', self.spacing_mu
        print '    spacing_sigma', self.spacing_sigma
        print '    k', self.k
        print '    beta', self.beta
        print '    noise_sd', self.noise_sd


def sample_speakers_for_conversations(cp_spec):
    speakers = sample_dir(cp_spec.speaker_dir_root, sum(cp_spec.speaker_groups),
                          glob_pattern='*', replace=False)
    i = 0
    speaker_groups = list()
    for group in cp_spec.speaker_groups:
        end = i + group
        speaker_groups.append(speakers[i:end])
        i = end
    return speaker_groups


def test_sample_speakers_for_conversations():
    cp_spec = CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                                speaker_groups=(4, 2, 3, 4))
    speaker_groups = sample_speakers_for_conversations(cp_spec)
    for i, g in enumerate(speaker_groups):
        print '{0} ------'.format(i)
        for s in g:
            print s

# test_sample_speakers_for_conversations()


def sample_duration(mu, sigma, sample_rate, offset=0):
    """
    A bit of a hack:
    (1) Does rejection sampling until get a positive sample from
        normal distribution
        with mu, sigma assumed to be on seconds scale
    (2) Multiply by sample_rate (to convert form seconds to unit of samples)
    (3) Take floor and convert to int
    (4) Add offset
    :param mu: float (in seconds)
    :param sigma: float (in seconds)
    :param sample_rate: int
    :param offset: int
    :return: int representing number of data-pts in sampled duration
    """
    s = numpy.random.normal(mu, sigma)
    while s < 0:
        s = numpy.random.normal(mu, sigma)
    return int(math.floor(s * sample_rate)) + offset


# ----------------------------------------------------------------------

class ConversationSpec(object):
    """
    Class representing conversation_spec consisting of some number of
    turn-taking speakers

    self.cp_spec : CocktailPartySpec : includes the following slots:
        speaker_dir_root : root directory of all speakers data
        sample_rate : sample rate of speaker data
        speaker_groups : tuple of ints representing number of speaker per sub-conversation_spec
        num_microphones
        spacing_mu : mean of space duration Gaussian
        spacing_sigma : varaince of space duration Gaussian
        k : concentration hyperparam of Dirichlet
        beta : Dirichlet mean; None defaults to uniform
        noise_sd : per-microphone standard Normal scale in standard-deviation
    self.speaker_group : seq of tuples representing speaker source parameters
        (<speaker_number_string>, <path_to_speaker_data_source_file>)
    self.num_speakers : number of speakers in conversation_spec
    self.conv_dynamics_spec : a dictionary specifying
        pi0 : initial state distribution
        A : row-stochastic transition matrix
    self.duration : total length of conversation_spec in seconds
        (relative to cp_spec.sample_rate)
    self.conversation_spec : a list of tuples representing speakers and space
        speaker: (speaker_idx, speaker_name, sentence_name, sentence_length, sentence_data)
        space: (None, None, None, length, None)
    """

    def __init__(self, cp_spec, speaker_group, conv_dynamics_spec, duration, sample_spec_p=False):
        self.cp_spec = cp_spec
        self.speaker_group = speaker_group
        self.num_speakers = len(self.speaker_group)
        self.conv_dynamics_spec = conv_dynamics_spec
        self.duration = duration
        self.conversation_spec = None
        if sample_spec_p:
            self.sample_spec()

    def to_string(self):
        s = list()
        s.append('# ConversationSpec start')
        s.append('speaker_group {0}'.format(self.speaker_group))
        s.append('num_speakers {0}'.format(self.num_speakers))
        s.append('conv_dynamics_spec start')
        s.append('pi0 {0}'.format(array_to_tuple(self.conv_dynamics_spec['pi0'])))
        s.append('A {0}'.format(array_to_tuple(self.conv_dynamics_spec['A'])))
        s.append('conv_dynamics_spec end')
        s.append('duration {0}'.format(self.duration))
        if self.conversation_spec:
            s.append('conversation_spec start')
            for spec in self.conversation_spec:
                if spec[4] is not None:
                    s.append('({0}, {1}, {2}, {3}, {4})'\
                             .format(spec[0], spec[1], spec[2], spec[3], None))
                    # array_to_tuple(spec[4])  # better to reload from source
                else:
                    s.append('{0}'.format(spec))
            s.append('conversation_spec end')
        s.append('# ConversationSpec end')

        return '\n'.join(s)

    def save(self, dest_path, save_cp_spec_p=False):
        with open(dest_path, 'w') as fout:
            pass

    def sample_spec(self, verbose=False):

        if verbose:
            print '----------'
            self.cp_spec.show()
            print 'speaker_group:'
            for g in self.speaker_group:
                print '    {0}'.format(g)
            print 'conv_dynamics_spec:', self.conv_dynamics_spec
            print 'duration:', self.duration

        total_duration = self.cp_spec.sample_rate * self.duration
        conversation_spec = list()

        def sample_uniform(conv):
            space_length = random.randint(1, self.cp_spec.initial_space_interval)
            conv.append((None, None, None, space_length, None))
            return space_length

        def sample_speaker_sentence(s_idx, conv):
            speaker_name, speaker_path = self.speaker_group[s_idx]

            # sample utterance
            utterance_sample = sample_dir(speaker_path, n=1, glob_pattern='*.wav', replace=True)
            sentence_name, wav_path = utterance_sample[0]
            sample_rate, sentence_data = scipy.io.wavfile.read(wav_path)
            sentence_length = sentence_data.shape[0]

            # save utterance
            conv.append((s_idx, speaker_name, sentence_name, sentence_length, sentence_data))

            # sample space
            space_length = sample_duration(self.cp_spec.spacing_mu,
                                           self.cp_spec.spacing_sigma,
                                           self.cp_spec.sample_rate)

            # save space
            conv.append((None, None, None, space_length, None))

            return sentence_data.shape[0] + space_length

        # sample from Markov chain
        speaker_idx = sample_idx(self.conv_dynamics_spec['pi0'])
        length = sample_uniform(conversation_spec)
        idx = length
        length = sample_speaker_sentence(speaker_idx, conversation_spec)
        idx += length
        while idx < total_duration:
            speaker_idx = sample_idx(self.conv_dynamics_spec['A'][speaker_idx, :])
            length = sample_speaker_sentence(speaker_idx, conversation_spec)
            idx += length

        if verbose:
            states = list()
            for c in conversation_spec:
                if c[0] is not None:
                    states.append(c[0])
                print c
            print 'total conversation_spec length:', idx
            print 'speaker frequencies', collections.Counter(states)

        self.conversation_spec = conversation_spec

        return conversation_spec

    def sample_data(self, sample_step_size=2000, include_summary_states_p=False):  # 1000, 500

        if self.conversation_spec is None:
            print '[ERROR] ConversationSpec.sample_data(): self.conversation_spec is None.'
            print '        Need to call ConversationSpec.sample_spec() first.'
            sys.exit()

        # conversation_spec data; rows=sample, cols=channel
        data = list()
        # conversation_spec state; rows=sample, cols=boolean state
        states = list()

        # start half way into step size to avoid initial 0's in sampled audio
        idx = int(math.floor(sample_step_size / 2.0))

        # extract the lengths of each conversation_spec component
        # and compute cumulative sums for absolute index boundaries
        seg_endpt_indices = numpy.cumsum([c[3] for c in self.conversation_spec])

        # since every segment's local indices are 0-based,
        # need this to adjust idx, which is absolute (i.e., is cumulative sum-based)
        seg_startpt_idx = 0

        def get_sample_from_conversation_segment(seg, idx=None):
            """
            Get sample from conversation_spec segment
            Segment types:
                speaker : (speaker_idx, speaker_name, sentence_name, sentence_length, sentence_data)
                space   : (None, None, None, length, None)
            When speaker:
                if idx is None, uniform random sample from sentence_data
                else return idx of sentence_data
            When space:
                return 0
            :param seg: conversation_spec segment
            :param idx: int or None
            :return: data_sample, state_sample
            """
            data_sample = numpy.zeros(self.num_speakers)

            if include_summary_states_p:
                state_sample = numpy.zeros(self.num_speakers + 1, dtype=int)
            else:
                state_sample = numpy.zeros(self.num_speakers, dtype=int)

            if seg[0] is None:
                # space
                return data_sample, state_sample
            else:
                # speaker
                state_sample[seg[0]] = 1
                if include_summary_states_p:  # only do this if we're including summary state
                    state_sample[self.num_speakers] = 1
                if idx is None:
                    data_sample[seg[0]] = numpy.random.choice(seg[4])
                else:
                    data_sample[seg[0]] = seg[4][idx]
                return data_sample, state_sample

        for seg_endpt_idx, seg in zip(seg_endpt_indices, self.conversation_spec):
            at_least_one = False
            while idx < seg_endpt_idx:
                at_least_one = True
                seg_local_idx = idx - seg_startpt_idx

                # get_sample
                data_sample, state_sample = get_sample_from_conversation_segment(seg, seg_local_idx)
                data.append(data_sample)
                states.append(state_sample)

                idx += sample_step_size

            # handle edge case where idx takes step over current seg
            if at_least_one is False:
                # get_sample
                data_sample, state_sample = get_sample_from_conversation_segment(seg, None)
                data.append(data_sample)
                states.append(state_sample)

            seg_startpt_idx = seg_endpt_idx  # update seg_startpt_idx to current seg_endpt_idx

        data = numpy.vstack(data)
        states = numpy.vstack(states)

        return data, states


def test_conversation_spec():
    cp_spec = CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                                sample_rate=25000,
                                speaker_groups=(4,),  # (4, 4, 4, 4),
                                num_microphones=12,
                                spacing_mu=0.2,  # mean
                                spacing_sigma=0.25,
                                k=2,
                                beta=None,
                                noise_sd=10.0)
    speaker_groups = sample_speakers_for_conversations(cp_spec)
    speaker_group = speaker_groups[0]  # since we're only sampling the first
    conversation_params = [sample_conversation_params
                           (state_size=num_speakers,
                            k=cp_spec.k, beta=cp_spec.beta)
                           for num_speakers in cp_spec.speaker_groups]
    conv_dynamics_spec = conversation_params[0]  # since we're only sampling the first

    print '----- Creating ConversationSpec'
    conv_spec = ConversationSpec(cp_spec, speaker_group, conv_dynamics_spec, 40)

    print '----- Sampling conversation_spec'
    conv_spec.sample_spec(verbose=True)

    # test aspects of sampling from conv_spec
    print '----- '
    print 'num segments:', len(conv_spec.conversation_spec)
    seg_indices = [c[3] for c in conv_spec.conversation_spec]
    cum_seg_indices = numpy.cumsum(seg_indices)
    print 'num seg_indices:', len(seg_indices)
    print 'num cum_seg_indices:', len(cum_seg_indices)
    print 'seg_indices:', seg_indices
    print 'cum_seg_indices:', cum_seg_indices

    print '----- Sample data and states'
    data, states = conv_spec.sample_data(sample_step_size=2000)
    print 'data.shape', data.shape
    print 'states.shape', states.shape

    numpy.savetxt('test_data_raw.txt', data)
    numpy.savetxt('test_states.txt', states, fmt='%d')

    print data
    print states

    plot_cp.plot_cp_data(path='test_states.txt')

# test_conversation_spec()


# ----------------------------------------------------------------------

def mix_latent_state_linear_combination(latent_state, W):
    """
    Calculates the latent state linear combination by weight matrix W
    :param latent_state: latent state matrix dim=(seq_length, num_speakers)
              -- could be binary or real-valued
    :param W: weight matrix dim=(num_speakers + 1, num_microphones)
              -- assumes last column is bias term
    :return: mixed matrix dim=(seq_length, num_microphones)
    """
    # add bias term as additional column of 1's appended to right of latent_state matrix
    lstate_with_bias = numpy.hstack((latent_state, numpy.ones((latent_state.shape[0], 1))))
    return numpy.dot(lstate_with_bias, W)


def test_mix_latent_state_linear_combination():
    print 'test_mix_latent_state_linear_combination()'
    latent_state = numpy.array([[1, 2],
                                [3, 4],
                                [5, 6],
                                [7, 8]])
    W = sample_emission_weight_matrix(num_speakers=latent_state.shape[1],
                                      num_microphones=10)
    print '---------- latent_state (4, 2)'
    print latent_state.shape
    print latent_state
    print '---------- W (3, 10)'
    print W.shape
    print W
    m = mix_latent_state_linear_combination(latent_state, W)
    print '---------- mixed (4, 10)'
    print m.shape
    print m

# test_mix_latent_state_linear_combination()


# ----------------------------------------------------------------------

def scale_to_center_unitvar(arr):
    """
    scale (standardize) by columns:
        per column, subtract the mean and divide by the stdev each
        value in the column
    :param arr:
    :return:
    """
    cs_arr = numpy.zeros(arr.shape)

    for i in range(arr.shape[1]):
        col_mu = numpy.mean(arr[:, i])
        col_sigma = numpy.std(arr[:, i])
        if col_sigma > 0.00001:
            # handle near-zero standard-deviation
            cs_arr[:, i] = (arr[:, i] - col_mu) / col_sigma
        else:
            cs_arr[:, i] = (arr[:, i] - col_mu)

    return cs_arr


def test_scale_to_center_unitvar():
    a = numpy.array([[1, 2, 3],
                     [4, 5, 6],
                     [5, 6, 12]])
    print 'a\n', a
    print 'mean(a)', numpy.mean(a)
    print 'std(a)', numpy.std(a)
    for i in range(a.shape[1]):
        print '  col {0} mean: {1}'.format(i, numpy.mean(a[:, i]))
        print '  col {0} std: {1}'.format(i, numpy.std(a[:, i]))

    print '----- scale:'
    scaled_a = scale_to_center_unitvar(a)
    print 'scaled_a\n', scaled_a
    print 'mean(scaled_a)', numpy.mean(scaled_a)
    print 'std(scaled_a)', numpy.std(scaled_a)
    for i in range(scaled_a.shape[1]):
        print '  col {0} mean: {1}'.format(i, numpy.mean(scaled_a[:, i]))
        print '  col {0} std: {1}'.format(i, numpy.std(scaled_a[:, i]))

# test_scale_to_center_unitvar()


# ----------------------------------------------------------------------

class CocktailParty(object):

    def __init__(self, cp_spec, train_duration, test_duration, sample_spec_p=False):
        self.cp_spec = cp_spec
        self.train_duration = train_duration
        self.test_duration = test_duration

        # linear combination matrix; dimension=(num_speakers + 1, num_microphones)
        self.W = None

        self.speaker_groups = None
        self.conversation_params = None
        self.train_conversation_specs = None
        self.test_conversation_specs = None

        if sample_spec_p:
            self.sample_spec()

        self.sample_step_size = None

        # during generation, keep track of simple stats
        self.generation_stats_string = None

        # latent state; dimension=(seq_length, num_speakers)
        self.train_data_raw = None
        self.train_states = None
        self.test_data_raw = None
        self.test_states = None

        # scale the raw date to be centered with unit variance, column-wise
        self.train_data_scaled = None
        self.test_data_scaled = None

        # data linearly mixed by self.W into microphone channels;
        # dimension=(seq_length, num_microphones)
        self.train_data_mixed = None
        self.test_data_mixed = None

        # emission (obs) noise;
        # dimension=(seq_length, num_microphones)
        self.train_noise = None
        self.test_noise = None

        # obs = sum of mixed and noise;
        # dimension=(seq_length, num_microphones)
        self.train_obs = None
        self.test_obs = None

    def sample_spec(self):
        # get emission weight matrix
        self.W = sample_emission_weight_matrix(num_speakers=sum(self.cp_spec.speaker_groups),
                                               num_microphones=self.cp_spec.num_microphones)

        # sample speakers for conversations
        self.speaker_groups = sample_speakers_for_conversations(self.cp_spec)

        # sample the parameters for each sub-conversation_spec
        self.conversation_params = [sample_conversation_params
                                    (state_size=num_speakers,
                                     k=self.cp_spec.k, beta=self.cp_spec.beta)
                                    for num_speakers in self.cp_spec.speaker_groups]

        # sample conversation_spec chains
        self.train_conversation_specs \
            = [ConversationSpec(self.cp_spec, speaker_group, conv_spec, self.train_duration,
                                sample_spec_p=True)
               for speaker_group, conv_spec
               in zip(self.speaker_groups, self.conversation_params)]
        self.test_conversation_specs \
            = [ConversationSpec(self.cp_spec, speaker_group, conv_spec, self.test_duration,
                                sample_spec_p=True)
               for speaker_group, conv_spec
               in zip(self.speaker_groups, self.conversation_params)]

    def sample_data(self, sample_step_size=2000, verbose=False):
        if self.W is None or self.speaker_groups is None or self.conversation_params is None \
                or self.train_conversation_specs is None or self.test_conversation_specs is None:
            print '[ERROR] CocktailParty.sample_data(): Need to run CocktailParty.sample_spec() first'
            if self.W is None:
                print '        Missing self.W'
            if self.speaker_groups is None :
                print '        Missing self.speaker_groups'
            if self.conversation_params is None:
                print '        Missing self.conversation_params'
            if self.train_conversation_specs is None:
                print '        Missing self.train_conversation_specs'
            if self.test_conversation_specs is None:
                print '        Missing self.test_conversation_specs'
            sys.exit()

        self.sample_step_size = sample_step_size

        # initialize the generation_stats_string: create empty list
        self.generation_stats_string = list()

        def sample_and_combine_subconversations(conversation_specs):
            # sample data from each sub_conversation
            sub_conversations = list()
            min_len = sys.maxint
            num_speakers = 0
            for conv_spec in conversation_specs:
                data, states = conv_spec.sample_data(sample_step_size=sample_step_size,
                                                     include_summary_states_p=False)
                sub_conversations.append((data, states))
                if data.shape[0] < min_len:
                    min_len = data.shape[0]
                num_speakers += data.shape[1]
            # combine sub_conversations
            all_data = numpy.zeros((min_len, num_speakers))
            all_states = numpy.zeros((min_len, num_speakers), dtype=int)
            col_idx = 0
            for data, states in sub_conversations:
                all_data[:, col_idx:col_idx+data.shape[1]] = data[0:min_len, :]
                all_states[:, col_idx:col_idx + states.shape[1]] = states[0:min_len, :]
                col_idx += data.shape[1]
            return all_data, all_states

        def make_stats_string(title, train, train_label, test, test_label, verbose):
            string_list \
                = ((title,
                    ('mean({0})'.format(train_label), numpy.mean(train)),
                    ('std({0})'.format(train_label), numpy.std(train)),
                    ('mean({0})'.format(test_label), numpy.mean(test)),
                    ('std({0})'.format(test_label), numpy.std(test))))
            string_list = map(lambda s: '{0}'.format(s), string_list)
            string_list = '\n'.join(string_list)
            self.generation_stats_string.append(string_list)
            if verbose:
                print string_list

        # sample and combine data and states for train and test
        self.train_data_raw, self.train_states \
            = sample_and_combine_subconversations(self.train_conversation_specs)
        self.test_data_raw, self.test_states \
            = sample_and_combine_subconversations(self.test_conversation_specs)

        make_stats_string('----- raw',
                          self.train_data_raw, 'self.train_data_raw',
                          self.test_data_raw, 'self.test_data_raw',
                          verbose=verbose)

        # center and scale to unit variance
        self.train_data_scaled = scale_to_center_unitvar(self.train_data_raw)
        self.test_data_scaled = scale_to_center_unitvar(self.test_data_raw)

        make_stats_string('----- after scaling',
                          self.train_data_scaled, 'self.train_data_scaled',
                          self.test_data_scaled, 'self.test_data_scaled',
                          verbose=verbose)

        # mix data
        self.train_data_mixed = mix_latent_state_linear_combination(self.train_data_scaled, self.W)
        self.test_data_mixed = mix_latent_state_linear_combination(self.test_data_scaled, self.W)

        make_stats_string('----- after mixing',
                          self.train_data_mixed, 'self.train_data_mixed',
                          self.test_data_mixed, 'self.test_data_mixed',
                          verbose=verbose)

        # sample emission noise
        self.train_noise = sample_emission_noise_matrix \
            (self.train_data_mixed.shape[0], self.cp_spec.num_microphones, self.cp_spec.noise_sd)
        self.test_noise = sample_emission_noise_matrix \
            (self.test_data_mixed.shape[0], self.cp_spec.num_microphones, self.cp_spec.noise_sd)

        make_stats_string('----- noise',
                          self.train_noise, 'self.train_noise',
                          self.test_noise, 'self.test_noise',
                          verbose=verbose)

        # obs
        self.train_obs = self.train_data_mixed + self.train_noise
        self.test_obs = self.test_data_mixed + self.test_noise

        make_stats_string('----- final obs',
                          self.train_obs, 'self.train_obs',
                          self.test_obs, 'self.test_obs',
                          verbose=verbose)

    def save(self, dest_root='.'):
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)
        params_dir = os.path.join(dest_root, 'model_params')
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        # save model generation statistics
        with open(os.path.join(params_dir, 'generation_stats.txt'), 'w') as fout:
            for s in self.generation_stats_string:
                fout.write('{0}\n'.format(s))

        # model params cp specs and convesations
        with open(os.path.join(params_dir, 'model.txt'), 'w') as fout:
            fout.write('###### CocktailParty start\n')

            fout.write('train_duration {0}\n'.format(self.train_duration))
            fout.write('test_duration {0}\n'.format(self.test_duration))
            fout.write('sample_step_size {0}\n'.format(self.sample_step_size))

            # self.cp_spec
            fout.write('### cp_spec start\n')
            fout.write('{0}\n'.format(self.cp_spec.to_string()))
            fout.write('### cp_spec end\n')

            # self.train_conversation_specs
            fout.write('### train_conversation_specs start\n')
            for conv_spec in self.train_conversation_specs:
                fout.write('{0}\n'.format(conv_spec.to_string()))
            fout.write('### train_conversation_specs end\n')

            # self.test_conversation_specs
            fout.write('### test_conversation_specs start\n')
            for conv_spec in self.test_conversation_specs:
                fout.write('{0}\n'.format(conv_spec.to_string()))
            fout.write('### test_conversation_specs end\n')

            fout.write('###### CocktailParty end\n')

        numpy.savetxt(os.path.join(params_dir, '1_train_obs_raw.txt'), self.train_data_raw, fmt='%f')
        numpy.savetxt(os.path.join(params_dir, '1_test_obs_raw.txt'), self.test_data_raw, fmt='%f')

        numpy.savetxt(os.path.join(params_dir, '2_train_data_scaled.txt'), self.train_data_scaled, fmt='%f')
        numpy.savetxt(os.path.join(params_dir, '2_test_data_scaled.txt'), self.test_data_scaled, fmt='%f')

        numpy.savetxt(os.path.join(params_dir, '3_train_data_mixed.txt'), self.train_data_mixed, fmt='%f')
        numpy.savetxt(os.path.join(params_dir, '3_test_data_mixed.txt'), self.test_data_mixed, fmt='%f')

        numpy.savetxt(os.path.join(params_dir, '4_train_noise.txt'), self.train_noise, fmt='%f')
        numpy.savetxt(os.path.join(params_dir, '4_test_noise.txt'), self.test_noise, fmt='%f')

        numpy.savetxt(os.path.join(dest_root, 'obs.txt'), self.train_obs, fmt='%f')
        numpy.savetxt(os.path.join(dest_root, 'states.txt'), self.train_states, fmt='%d')

        numpy.savetxt(os.path.join(dest_root, 'test_obs.txt'), self.test_obs, fmt='%f')
        numpy.savetxt(os.path.join(dest_root, 'test_states.txt'), self.test_states, fmt='%d')

        numpy.savetxt(os.path.join(dest_root, 'weights.txt'), self.W, fmt='%f')


def test_cocktail_party_generation():
    cp_spec = CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                                sample_rate=25000,
                                speaker_groups=(4, 4, 4, 4),
                                num_microphones=12,
                                spacing_mu=0.2,  # mean
                                spacing_sigma=0.25,
                                k=2,
                                beta=None,
                                noise_sd=0.3)
    print '----- Generating Cocktail Party'
    cp = CocktailParty(cp_spec=cp_spec,
                       train_duration=40,
                       test_duration=40,
                       sample_spec_p=True)
    print '----- Sampling cp data'
    cp.sample_data(sample_step_size=2000, verbose=True)

    print '----- Saving data'
    cp.save('cp0')

    # plot_cp.plot_cp_data(path='cp0/states.txt', show_p=False)
    # plot_cp.plot_cp_data(path='cp0/test_states.txt', show_p=True)

# test_cocktail_party_generation()


# ----------------------------------------------------------------------

def generate_random_cocktail_parties(cp_spec,
                                     num_parties=10,
                                     party_num_offset=0,
                                     train_duration=40,  # in seconds
                                     test_duration=40,
                                     dest_root='.'):

    dest_dir = os.path.join(dest_root, 'n{0}'.format(cp_spec.noise_sd))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for party in range(party_num_offset, num_parties + party_num_offset):
        print 'Generating party {0} of {1}'.format(party, num_parties)

        party_dir = os.path.join(dest_dir, 'cp{0}'.format(party))
        if not os.path.exists(party_dir):
            os.makedirs(party_dir)

        cp = CocktailParty(cp_spec=cp_spec,
                           train_duration=train_duration,
                           test_duration=test_duration,
                           sample_spec_p=True)

        cp.sample_data(sample_step_size=2000)

        cp.save(party_dir)

    print 'DONE.'


def test_generate_random_cocktail_parties():
    cp_spec = CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                                sample_rate=25000,
                                speaker_groups=(4, 4, 4, 4),
                                num_microphones=12,
                                spacing_mu=0.2,  # mean
                                spacing_sigma=0.25,
                                k=2,
                                beta=None,
                                noise_sd=0.3)
    generate_random_cocktail_parties(cp_spec=cp_spec,
                                     num_parties=2,
                                     party_num_offset=0,
                                     train_duration=40,
                                     test_duration=40,
                                     dest_root='cocktail_SSC1_s16_m12')

# test_generate_random_cocktail_parties()


# ----------------------------------------------------------------------
# SCRIPTS
# ----------------------------------------------------------------------

CP_SSC1_ROOT = os.path.join(os.path.join(HAMLET_ROOT, DATA_ROOT), 'cocktail_SSC1_s16_m12')

print os.listdir(os.path.join(HAMLET_ROOT, DATA_ROOT))


def generate_noise_0p3(num_parties=10):
    cp_spec = CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                                sample_rate=25000,
                                speaker_groups=(4, 4, 4, 4),
                                num_microphones=12,
                                spacing_mu=0.2,  # mean
                                spacing_sigma=0.25,
                                k=2,
                                beta=None,
                                noise_sd=0.3)
    generate_random_cocktail_parties(cp_spec=cp_spec,
                                     num_parties=num_parties,
                                     party_num_offset=0,
                                     train_duration=40,
                                     test_duration=40,
                                     dest_root=CP_SSC1_ROOT)

generate_noise_0p3(num_parties=10)
