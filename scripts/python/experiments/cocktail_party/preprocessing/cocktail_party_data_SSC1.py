import scipy.io.wavfile
import glob
import sys
import os
import urllib
import zipfile
import audioop
import wave
import numpy
import math
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
# Hamlet Paths
# ----------------------------------------------------------------------

HAMLET_ROOT = '../../../../../../'
DATA_ROOT = experiment_tools.DATA_ROOT
PARAMETERS_ROOT = experiment_tools.PARAMETERS_ROOT
RESULTS_ROOT = experiment_tools.RESULTS_ROOT


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

SSC1_URL = 'http://laslab.org/SpeechSeparationChallenge'
# http://laslab.org/SpeechSeparationChallenge/
# 34 subjects, 500 utterances per subject

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
    stats = numpy.array([get_wav_file_summary_info(filepath)
                         for filepath in glob.glob(os.path.join(data_root, '*.wav'))])
    sample_rates = set(stats[:, 0])
    all_data_min_len = numpy.min(stats[:, 1])
    all_data_max_len = numpy.max(stats[:, 1])
    all_val_min = numpy.min(stats[:, 2])
    all_val_max = numpy.max(stats[:, 3])

    if verbose:
        print 'sample_rates', sample_rates
        print 'all_data_min_len', all_data_min_len
        print 'all_data_max_len', all_data_max_len
        print 'all_val_min', all_val_min
        print 'all_val_max', all_val_max

    return sample_rates, all_data_min_len, all_data_max_len, all_val_min, all_val_max


def get_stats_for_all_speakers(data_root, verbose=False):
    stats = [get_wav_set_stats(speaker_path)
             for speaker_path in glob.glob(os.path.join(data_root, '*'))]

    all_data_min_len = numpy.min(stats[:, 1])
    all_data_max_len = numpy.max(stats[:, 1])
    all_val_min = numpy.min(stats[:, 2])
    all_val_max = numpy.max(stats[:, 3])

    if verbose:
        print 'all_data_min_len', all_data_min_len
        print 'all_data_max_len', all_data_max_len
        print 'all_val_min', all_val_min
        print 'all_val_max', all_val_max


# get_stats_for_all_speakers(SSC1_DATA_ROOT, True)


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
# Generate conversation
# ----------------------------------------------------------------------

'''
Need to generate the following data:
obs.txt : real-valued multivariate observations
test_obs.txt : test data generated from same process
states.txt : binary latent state: 1=speaking, 0=silent
weights.txt : mixing weights
emissions/
    linear_combined_latent_state_centered_scaled.txt
    linear_combined_latent_state :
    noise.txt : noise applied to each microphone channel

statistics_human.txt : human-readable statistics of generated data files, per microphone/channel
    weights
    linear_combined_latent_state
    noise
    obs
    test_obs

model.params : parameters of generating process
'''


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

    return dict(state_size=state_size, pi0=pi0, A=A)


def sample_emission_weight_matrix(num_speakers, num_microphones, bias=1):
    return numpy.random.uniform(0, 1, size=( num_speakers + bias, num_microphones ))


class CocktailPartySpec(object):
    def __init__(self,

                 speaker_dir_root,

                 sample_rate=25000,  # sample pts per second

                 speaker_groups=(4, 4, 4, 4),
                 num_microphones=12,

                 spacing_mu=0.25,     # mean
                 spacing_sigma=0.25,  # variance

                 # initial state and transition matrix
                 # pi0, A_i* ~ Dirichlet(k, beta)
                 k=2,  # concentration
                 beta=None,  # Dir mean; None defaults to uniform

                 # hard-coded h precision, same for each microphone
                 h=10.0
                 ):

        self.speaker_dir_root = speaker_dir_root

        self.sample_rate = sample_rate
        self.speaker_groups = speaker_groups
        self.num_microphones = num_microphones

        # spacing parameters
        self.spacing_mu = spacing_mu
        self.spacing_sigma = spacing_sigma

        self.k = k
        self.beta = beta

        # TODO: Possibly sample noise precision from prior Gamma?

        if isinstance(h, (list, tuple)):
            if len(h) != num_microphones:
                print 'ERROR: length of h is {0} != num microphones {1}'\
                    .format(len(h), num_microphones)
                sys.exit()
            else:
                h = [float(h)] * num_microphones
        self.h = numpy.array(h)


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


def sample_duration(mu, sigma, sample_rate, offset=1):
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
    :return:
    """
    s = numpy.random.normal(mu, sigma)
    while s < 0:
        s = numpy.random.normal(mu, sigma)
    return int(math.floor(s * sample_rate)) + offset


def sample_conversation(cp_spec, speaker_group, conv_spec, duration):
    total_duration = cp_spec.sample_rate * duration

    # NOTE: the additional row added after the speaker groups represents ANY speaker
    data = numpy.zeros((len(speaker_group) + 1, total_duration), dtype=numpy.int8)

    idx = 0
    speaker_idx =
    while idx < total_duration:
        pass

    # TODO
    pass


def generate_random_cocktail_parties(cp_spec,
                                     num_parties=10,
                                     party_num_offset=0,
                                     train_duration=15,  # in seconds
                                     test_duration=15,
                                     save_statistics_p=True,
                                     dest_dir=None):
    for party in range(party_num_offset, num_parties + party_num_offset):
        print 'Generating party {0}'.format(party)

        # get emission weight matrix
        W = sample_emission_weight_matrix(num_speakers=sum(cp_spec.speaker_groups),
                                          num_microphones=cp_spec.num_microphones)

        # sample speakers for conversations
        speaker_groups = sample_speakers_for_conversations(cp_spec)

        # sample the parameters for each sub-conversation
        conversation_params = [sample_conversation_params
                               (state_size=num_speakers,
                                k=cp_spec.k, beta=cp_spec.beta)
                               for num_speakers in cp_spec.speaker_groups]

        # sample conversation chains
        train_conversation_chains = [sample_conversation(cp_spec, speaker_group, conv_spec, train_duration)
                                     for speaker_group, conv_spec
                                     in zip(speaker_groups, conversation_params)]

        # TODO
        # NOTE: Bias term will be added to the END of latent states -- see original latent_state_linear_combination


CocktailPartySpec(speaker_dir_root=SSC1_DATA_ROOT,
                  sample_rate=25000,
                  speaker_groups=(4, 4, 4, 4),
                  num_microphones=12,
                  # TODO spacing param
                  k=2,
                  beta=None,
                  h=10.0)

# ----------------------------------------------------------------------
# OLD
# ----------------------------------------------------------------------

def show_info(aname, a):
    """
    Simple helper for debugging
    :param aname: name of the array being shown
    :param a:
    :return:
    """
    print "Array", aname
    print "shape:", a.shape
    print "dtype:", a.dtype
    print "min, max:", a.min(), a.max()
    print


'''
def get_wav_file_summary_info(wav_file):
    """
    Returns the sample rate (int: in samples/sec) and data (2d: data, channel (usually 2))
    :param wav_file:
    :return: int, (int, int)
    """
    wav_data = scipy.io.wavfile.read(wav_file)
    return wav_data[0], wav_data[1].shape
'''


def get_stats_wav_files(root_path):
    """
    Collect simple stats of wav files in root_path
    :param root_path:
    :return:
    """
    sample_rates = set()
    data_lengths = list()
    for fpath in glob.glob(os.path.join(root_path, '*.wav')):
        sample_rate, (data_len, channels) = get_wav_file_summary_info(fpath)
        sample_rates.add(sample_rate)
        data_lengths.append(data_len)

    return


# ----------------------------------------------------------------------

def run():

    for i, id_src in enumerate(glob.glob(os.path.join(SSC1_DATA_ROOT, '*'))):
        wave_paths = glob.glob(os.path.join(id_src, '*.wav'))
        print i, id_src, len(wave_paths)

    '''
    for f in os.listdir(CHIME_DATA_ROOT):
        if len(f) >= 2 and 'id' == f[:2]:
            print f
    '''

# run()


def run2():
    f1 = os.path.join(SSC1_DATA_ROOT, '1/bbaf2n.wav')
    f2 = os.path.join(SSC1_DATA_ROOT, '1/bbaf5a.wav')

    for f in (f1, f2):
        print f, get_wav_file_summary_info(f)

# run2()


def blee():
    wav_filename = '1/bbaf2n.wav'
    wav_path = os.path.join(SSC1_DATA_ROOT, wav_filename)
    wav_data = scipy.io.wavfile.read(wav_path)
    data = wav_data[1]  # [:, 0]
    show_info(wav_filename, data)

# blee()

# ----------------------------------------------------------------------

'''
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

def downsample_wav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=1):
    """
    http://stackoverflow.com/questions/30619740/python-downsampling-wav-audio-file
    :param src:
    :param dst:
    :param inrate:
    :param outrate:
    :param inchannels:
    :param outchannels:
    :return:
    """
    if not os.path.exists(src):
        print 'Source not found!'
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print 'Failed to open files!'
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print 'Failed to downsample wav'
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print 'Failed to write wav'
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print 'Failed to close wav files'
        return False

    return True
