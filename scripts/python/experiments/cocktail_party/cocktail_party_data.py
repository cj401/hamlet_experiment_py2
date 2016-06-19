import os
import random

import numpy as np
import numpy.random
import prettytable  # This requires that prettytable has been installed!

from utilities import util

__author__ = 'clayton'


def collect_frequencies(arr):
    c = util.Counter()
    for v in arr:
        c[v] += 1
    return c


def sample_idx(pvec):
    r = random.random()
    return next((i for i, v in enumerate(np.cumsum(pvec)) if r <= v))


def sample_idx_test(pvec, reps=100000):
    f = collect_frequencies([sample_idx(pvec) for i in range(reps)])
    f.normalize()
    return [abs(f[idx] - pvec[idx]) for idx in range(len(pvec))]


# --------------------------------------------------------------------


def sample_Dirichlet(dim=3, k=2, beta=None, size=None):
    """
    Wrapper to parameterize Dirichlet like Hamlet
    :param dim: number of dimensions of Dir vector (default 3)
    :param k: concentration (default 2)
    :param beta: mean vector (default uniform)
    :param size: number of samples
    :return: size-many samples from Dirichlet
    """
    if beta is None:
        beta = np.array([1.0 / dim for i in range(dim)])
    alpha = np.array([k * len(beta) * b for b in beta])
    return numpy.random.dirichlet(alpha, size=size)


# --------------------------------------------------------------------


def sample_conversation_params(state_size=3,
                               spacing_omega=0.8,

                               # initial state and transition matrix ~ Dirichlet(k, beta)
                               k=2,        # concentration
                               beta=None,  # Dir mean; None defaults to uniform

                               # super state duration ~ Poisson(omega)
                               # omega ~ Gamma(a_omega, b_omega)
                               a_omega = 3.0,  # shape
                               b_omega = 1.0   # rate
                               ):

    # initial state distribution
    pi0 = sample_Dirichlet(state_size, k=k, beta=beta, size=None)

    # transition matrix A
    A = sample_Dirichlet(state_size, k=k, beta=beta, size=state_size)

    # super state duration ~ Poisson(omega)
    omegas = np.array( [ spacing_omega ] + \
                       list(numpy.random.gamma(a_omega, b_omega, size=state_size)) )

    return dict(state_size=state_size, pi0=pi0, A=A, omegas=omegas)


# --------------------------------------------------------------------

# can kill this section
# this was useful for understanding the generative process in batch form,
# but more useful to precisely control the generation through
# the sequential iteration method below

def batch_sample_markov_state_sequence(A, pi0, length):
    state = sample_idx(pi0)
    seq = [state]
    for i in range(1, length):
        state = sample_idx(A[state, :])
        seq.append(state)
    return np.array(seq)


def batch_add_spaces_to_state_sequence(state_seq):
    new_seq = np.array([0] * ((2 * len(state_seq)) - 1))
    for s, idx in zip(state_seq, range(len(state_seq))):
        new_seq[2 * idx] = s + 1  # s+1 in order to move every state id up one, so s=0 is space
    return new_seq


def batch_sample_state_durations(state_seq, omegas, poisson_adj=1):
    return np.array([numpy.random.poisson(omegas[state]) + poisson_adj
                     for state in state_seq])


def batch_expand_state_durations(state_seq, durations):
    # util.flatten1( )
    return np.array([[state] * D for state, D in zip(state_seq, durations)])


def batch_gen_conversation(spec, length=10, verbose=True):
    pi0 = spec['pi0']
    A = spec['A']
    omegas = spec['omegas']
    seq = batch_sample_markov_state_sequence(A, pi0, length)
    if verbose: print 'seq:   ', seq
    seq_s = batch_add_spaces_to_state_sequence(seq)
    if verbose: print 'seq_s: ', seq_s
    D = batch_sample_state_durations(seq_s, omegas)
    if verbose: print 'D:     ', D
    seq_e = batch_expand_state_durations(seq_s, D)
    if verbose: print 'seq_e: ', seq_e
    if verbose: print 'fseq_e:', util.flatten1(seq_e)
    return seq_e, seq, seq_s, D


# --------------------------------------------------------------------


def sample_duration(omega, poisson_adj=1):
    return numpy.random.poisson(omega) + poisson_adj


def print_chain(chain):
    print 'super_state:', chain['super_state']
    print 'D:          ', chain['D']
    print 'state:      ', chain['state']
    print 'space:      ', chain['space']
    print 'super_state_seq: ', chain['super_state_seq']
    print 'super_state_D:   ', chain['super_state_D']
    print 'state_seq:       ', chain['state_seq']
    print 'state_binary_seq:', chain['state_binary_seq']


def lol(ll):
    if ll is None:
        return None
    else:
        return [ list(l) for l in ll ]


def chain_to_string(chain):
    lines = list()
    lines.append('dict(')
    lines.append('\'super_state\'={0},'.format(chain['super_state']))
    lines.append('\'D\'={0},'.format(chain['D']))
    lines.append('\'state\'={0},'.format(chain['state']))
    lines.append('\'space\'={0},'.format(chain['space']))
    lines.append('\'super_state_seq\'={0},'.format(list(chain['super_state_seq'])))
    lines.append('\'super_state_D\'={0},'.format(list(chain['super_state_D'])))
    lines.append('\'state_seq\'={0},'.format(list(chain['state_seq'])))
    lines.append('\'state_binary_seq\'={0}'.format(lol(chain['state_binary_seq'])))
    lines.append(')\n')
    return '\n'.join(lines)


def get_binary_state_vector(state, size):
    v = np.array([0]*size)
    if state > 0: v[state-1] = 1
    return v


def sample_next_hsmm_state(conv_spec, chain=None, verbose=False):
    if not chain:
        super_state = sample_idx(conv_spec['pi0']) + 1
        D = sample_duration(conv_spec['omegas'][super_state])
        state = super_state
        super_state_seq = [ super_state ]
        super_state_D = [ D ]
        state_seq = [ super_state ]
        state_binary_seq = [ get_binary_state_vector(state, conv_spec['state_size']) ]
        space = False
        chain = dict(super_state=super_state, state=state, D=D-1,
                     state_seq=state_seq, state_binary_seq=state_binary_seq,
                     super_state_seq=super_state_seq,
                     super_state_D=super_state_D,
                     space=space)
    else:
        if ( chain['D'] <= 0 ):  # leq just to be safe
            if chain['space']:   # just finished space, start new super_state
                chain['space'] = False
                chain['super_state'] = sample_idx(conv_spec['A'][chain['super_state']-1])+1
                chain['state'] = chain['super_state']
                chain['D'] = sample_duration(conv_spec['omegas'][chain['super_state']])
            else:                # just finished super_state, start new space
                chain['space'] = True
                chain['state'] = 0
                chain['D'] = sample_duration(conv_spec['omegas'][0])
            chain['super_state_seq'].append(chain['state'])
            chain['super_state_D'].append(chain['D'])

        chain['D'] -= 1
        chain['state_seq'].append(chain['state'])
        chain['state_binary_seq'].append(get_binary_state_vector(chain['state'],
                                                                 conv_spec['state_size']))

    if verbose: print_chain(chain)

    return chain


def sample_hsmm_states(conv_spec, n=100):
    """
    Generate a sequence of length n by iteratively sampling n states
    :param conv_spec:
    :param n:
    :return:
    """
    chain = sample_next_hsmm_state(conv_spec)
    for i in range(n-1):
        chain = sample_next_hsmm_state(conv_spec, chain)
    return chain


def combine_chain_latent_state_vectors(chain_list):
    """
    Combine latent state vectors from multiple chains into single latent state vector
    :param chain_list:
    :return:
    """
    # collect list of binary seqs from list of chains
    bseqs = [ chain['state_binary_seq'] for chain in chain_list ]
    # zip bseqs to get sequence of tuples of the binary states for all chains
    # per step & concatenate them
    return map( lambda t: np.concatenate(t), zip(*bseqs) )


# -------


def sample_emission_weight_matrix(num_speakers, num_microphones, bias=1):
    return numpy.random.uniform(0, 1, size=( num_speakers + bias, num_microphones ))


def sample_precision( num_microphones, a_h=1.0, b_h=1.0):
    """
    NOTE THAT b_h IS *RATE*, not scale!!!
    :param num_microphones:
    :param a_h: Gamma shape
    :param b_h: Gamma RATE (not scale!!) : scale = 1/b_h
    :return:
    """
    return np.random.gamma(shape=a_h, scale=1/b_h, size=num_microphones)


def latent_state_linear_combination(latent_state, W):
    """
    Calculates the latent state linear combination by weight matrix W
    :param latent_state: latent state vector -- could be binary or real-valued
    :param W: weight matrix -- assumes last column is bias term
    :return:
    """
    # add bias term as additional bit added to end of latent_state
    #print 'binary_state    ', binary_state
    lstate_with_bias = np.concatenate([latent_state, np.array([1.0])])
    #print 'bstate_with_bias', bstate_with_bias
    return np.dot(lstate_with_bias, W)


def normal_noise_precision(mu, h):
    """
    Sample from N(mu, \sqrt(1/h))
    :param mu:
    :param h:
    :return:
    """
    # print 'mu.shape', mu.shape
    # print 'h.shape', h.shape
    return numpy.random.normal(mu, np.sqrt(1/h))


# l^T * W = c

def generate_normal_emissions(bseqs, W, h, center_scale_data_p=True):

    # linear combination
    linear_combined_latent_states_original = \
        np.array([ list(latent_state_linear_combination(bstate, W))
                   for bstate in bseqs])

    # print 'linear_combined_latent_states_original.shape', linear_combined_latent_states_original.shape
    # print linear_combined_latent_states_original
    # sys.exit(0)

    # optionally center and scale linearly combined latent state
    linear_combined_latent_states_centered_scaled = np.copy(linear_combined_latent_states_original)

    # W_centered_scaled = np.copy(W)
    if center_scale_data_p:
        linear_combined_latent_states_centered_scaled, data_mean, data_std = \
            util.center_scale_data_array(linear_combined_latent_states_original)
        #W_centered_scaled, data_mean, data_std \
        #    = util.center_scale_data_array(W, -data_mean, 1/data_std)
    else:
        print "NOT CENTERING/SCALING"
        arr_eql = np.array_equal(linear_combined_latent_states_original,
                                 linear_combined_latent_states_centered_scaled)
        print "    lcls equal to lclscs: {0}".format(arr_eql)

    # sample noise
    noise = normal_noise_precision(np.zeros(h.shape),
                                   np.array([list(h)] * linear_combined_latent_states_centered_scaled.shape[0]))

    # add noise
    emissions_with_noise = linear_combined_latent_states_centered_scaled + noise

    return (emissions_with_noise,
            noise,
            linear_combined_latent_states_centered_scaled,
            linear_combined_latent_states_original,
            #W_centered_scaled
            )

def test_generate_normal_emissions():

    def print_results(linear_combined_latent_states_original,
                      linear_combined_latent_states_centered_scaled,
                      noise,
                      emissions_with_noise,
                      W_centered_scaled=None):

        if W_centered_scaled:
            print 'W_centered_scaled:\n{0}'.format(W_centered_scaled)
            W_eql = np.array_equal(W, W_centered_scaled)
            print 'W and W_cs are equal: {0}'.format(W_eql)

        print 'linear_combined_latent_states_original:\n{0}'.format(linear_combined_latent_states_original)
        print 'linear_combined_latent_states_centered_scaled:\n{0}'.format(linear_combined_latent_states_centered_scaled)

        arr_eql = np.array_equal(linear_combined_latent_states_centered_scaled,
                                 linear_combined_latent_states_original)
        print 'lcls and lclscs are equal: {0}'.format(arr_eql)

        print 'noise:\n{0}'.format(noise)
        print 'emissions_with_noise:\n{0}'.format(emissions_with_noise)

    # state sequence of 6 latent state vectors
    bseqs = [ np.array([0, 1, 1, 0]),
              np.array([1, 1, 0, 0]),
              np.array([1, 1, 1, 0]),
              np.array([0, 0, 0, 1]),
              np.array([0, 0, 1, 1]),
              np.array([1, 0, 1, 1])]
    # Weight matrix, 4-state latent vector x 3 microphones
    # last row is bias
    W = np.array([[0.25, 0.35, 0.40],
                  [0.20, 0.30, 0.50],
                  [0.30, 0.50, 0.20],
                  [0.10, 0.70, 0.20],
                  [0.35, 0.50, 0.15]])
    h = np.array([1.0, 1.0, 1.5])

    print 'bseqs: {0}'.format(bseqs)
    print 'W: {0}'.format(W)
    print 'h: {0}'.format(h)

    W_centered_scaled = None

    print '----------------------------------------'

    emissions_with_noise, \
    noise, \
    linear_combined_latent_states_centered_scaled, \
    linear_combined_latent_states_original \
        = generate_normal_emissions(bseqs, W, h, center_scale_data_p=False)

    print_results(linear_combined_latent_states_original,
                  linear_combined_latent_states_centered_scaled,
                  noise,
                  emissions_with_noise,
                  W_centered_scaled)

    print '----------------------------------------'

    emissions_with_noise, \
    noise, \
    linear_combined_latent_states_centered_scaled, \
    linear_combined_latent_states_original, \
        = generate_normal_emissions(bseqs, W, h, center_scale_data_p=True)

    print_results(linear_combined_latent_states_original,
                  linear_combined_latent_states_centered_scaled,
                  noise,
                  emissions_with_noise,
                  W_centered_scaled)

    '''
    print '----------------------------------------'

    linear_combined_latent_states_original_cs_directly = \
        np.array([ list(latent_state_linear_combination(bstate, W_centered_scaled))
                   for bstate in bseqs])

    print 'linear_combined_latent_states_centered_scaled:\n{0}'\
        .format(linear_combined_latent_states_centered_scaled)
    print 'linear_combined_latent_states_original_cs_directly:\n{0}'\
        .format(linear_combined_latent_states_original_cs_directly)
    arr_diff = linear_combined_latent_states_original_cs_directly \
               - linear_combined_latent_states_centered_scaled
    print 'arr_diff:\n{0}'.format(arr_diff)
    '''

# test_generate_normal_emissions()


'''
return [ normal_noise(latent_state_linear_combination(bstate, W), h)
         for bstate in bseqs ]
'''


# -------

class MixedConversations:
    def __init__(self,
                 train_length=None,
                 train_bseqs=None,
                 train_cseqs=None,
                 train_linear_combined_latent_states_original=None,
                 train_linear_combined_latent_states_centered_scaled=None,
                 train_noise=None,
                 train_emissions=None,
                 train_conv_chains=None,

                 test_length=None,
                 test_bseqs=None,
                 test_cseqs=None,
                 test_linear_combined_latent_states_original=None,
                 test_linear_combined_latent_states_centered_scaled=None,
                 test_noise=None,
                 test_emissions=None,
                 test_conv_chains=None,

                 W=None,
                 h=None
                 ):

        self.train_length = train_length
        self.train_bseqs = train_bseqs
        self.train_cseqs = train_cseqs
        self.train_linear_combined_latent_states_original = train_linear_combined_latent_states_original
        self.train_linear_combined_latent_states_centered_scaled = train_linear_combined_latent_states_centered_scaled
        self.train_noise = train_noise
        self.train_emissions = train_emissions
        self.train_conv_chains = train_conv_chains

        self.test_length = test_length
        self.test_bseqs = test_bseqs
        self.test_cseqs = test_cseqs
        self.test_linear_combined_latent_states_original = test_linear_combined_latent_states_original
        self.test_linear_combined_latent_states_centered_scaled = test_linear_combined_latent_states_centered_scaled
        self.test_noise = test_noise
        self.test_emissions = test_emissions
        self.test_conv_chains = test_conv_chains

        self.W = W
        self.h = h

    def to_string(self):
        lines = list()
        lines.append('MixedConversations(')

        # train figures
        lines.append('train_length={0},'.format(self.train_length))
        lines.append('train_conv_chains=[')
        for conv_chain in self.train_conv_chains:
            lines.append(chain_to_string(conv_chain))  # conv_chain.to_string()
            lines.append(',')
        lines.append('],')
        lines.append('train_bseqs={0},'.format(lol(self.train_bseqs)))
        if self.train_cseqs:
            lines.append('train_cseqs={0},'.format(lol(self.train_cseqs)))

        # train emissions
        lines.append('train_linear_combined_latent_states_original={0},'
                     .format(list(self.train_linear_combined_latent_states_original)))
        lines.append('train_linear_combined_latent_states_centered_scaled={0},'
                     .format(list(self.train_linear_combined_latent_states_centered_scaled)))
        lines.append('train_noise={0},'.format(list(self.train_noise)))
        lines.append('train_emissions={0},'.format(list(self.train_emissions)))

        # test figures
        lines.append('test_length={0},'.format(self.test_length))
        lines.append('test_conv_chains=[')
        for conv_chain in self.test_conv_chains:
            lines.append(chain_to_string(conv_chain))  # conv_chain.to_string()
            lines.append(',')
        lines.append('],')
        lines.append('test_bseqs={0},'.format(lol(self.test_bseqs)))
        if self.test_cseqs:
            lines.append('test_cseqs={0},'.format(lol(self.test_cseqs)))

        # test emissions
        lines.append('test_linear_combined_latent_states_original={0},'
                     .format(list(self.test_linear_combined_latent_states_original)))
        lines.append('test_linear_combined_latent_states_centered_scaled={0},'
                     .format(list(self.test_linear_combined_latent_states_centered_scaled)))
        lines.append('test_noise={0},'.format(list(self.test_noise)))
        lines.append('test_emissions={0},'.format(list(self.test_emissions)))

        lines.append('W={0}'.format(lol(self.W)))
        lines.append('h={0}'.format(list(self.h)))

        lines.append(')')
        return '\n'.join(lines)

    def save_to_file(self, path='test/'):
        if not os.path.exists(path):
            os.makedirs(path)

        emissions_path = path + 'emissions/'
        if not os.path.exists(emissions_path):
            os.makedirs(emissions_path)

        print 'Saving MixedConversations to path \'{0}\''.format(path)

        with open(path + 'model.params', 'w') as fout:
            fout.write(self.to_string())

        with open(path + 'states.txt', 'w') as fout:
            for binary_state in self.train_bseqs:
                for bit in binary_state:
                    fout.write(' {0}'.format(bit))
                fout.write('\n')

        if self.train_cseqs:
            with open(path + 'states_continuous.txt', 'w') as fout:
                for continuous_state in self.train_cseqs:
                    for val in continuous_state:
                        fout.write(' {0}'.format(val))
                    fout.write('\n')

        with open(path + 'obs.txt', 'w') as fout:
            for observation in self.train_emissions:
                for obs in observation:
                    fout.write(' {0}'.format(obs))
                fout.write('\n')

        np.savetxt(emissions_path + 'linear_combined_latent_state.txt',
                   self.train_linear_combined_latent_states_original,
                   delimiter=' ')

        np.savetxt(emissions_path + 'linear_combined_latent_state_centered_scaled.txt',
                   self.train_linear_combined_latent_states_centered_scaled,
                   delimiter=' ')

        np.savetxt(emissions_path + 'noise.txt',
                   self.train_noise,
                   delimiter=' ')

        with open(path + 'test_obs.txt', 'w') as fout:
            for observation in self.test_emissions:
                for obs in observation:
                    fout.write(' {0}'.format(obs))
                fout.write('\n')

        with open(path + 'weights.txt', 'w') as fout:
            for w_row in self.W:
                for w in w_row:
                    fout.write(' {0}'.format(w))
                fout.write('\n')

        print 'DONE.'


# --------------------------------------------------------------------


def sample_mix_conversations(conversations,
                             W, h,
                             center_scale_data_p=True,
                             num_train=100, num_test=100):

    train_conv_chains = [ sample_hsmm_states(spec, num_train) for spec in conversations ]
    train_bseqs = combine_chain_latent_state_vectors(train_conv_chains)
    train_emissions_with_noise, \
        train_noise, \
        train_linear_combined_latent_states_centered_scaled, \
        train_linear_combined_latent_states_original \
            = generate_normal_emissions(train_bseqs, W, h, center_scale_data_p=center_scale_data_p)

    test_conv_chains = [ sample_hsmm_states(spec, num_test) for spec in conversations ]
    test_bseqs = combine_chain_latent_state_vectors(test_conv_chains)
    test_emissions_with_noise, \
        test_noise, \
        test_linear_combined_latent_states_centered_scaled, \
        test_linear_combined_latent_states_original \
            = generate_normal_emissions(test_bseqs, W, h, center_scale_data_p=center_scale_data_p)

    return MixedConversations(train_length=num_train,
                              train_bseqs=train_bseqs,
                              train_linear_combined_latent_states_original=train_linear_combined_latent_states_original,
                              train_linear_combined_latent_states_centered_scaled=train_linear_combined_latent_states_centered_scaled,
                              train_noise=train_noise,
                              train_emissions=train_emissions_with_noise,
                              train_conv_chains=train_conv_chains,

                              test_length=num_test,
                              test_bseqs=test_bseqs,
                              test_linear_combined_latent_states_original=test_linear_combined_latent_states_original,
                              test_linear_combined_latent_states_centered_scaled=test_linear_combined_latent_states_centered_scaled,
                              test_noise=test_noise,
                              test_emissions=test_emissions_with_noise,
                              test_conv_chains=test_conv_chains,

                              W=W, h=h)


# -------

'''
def mixed_convs_to_string(mconvs):
    lines = list()
    lines.append('dict=(')
    lines.append('\'train_length\'={0},'.format(mconvs['train_length']))
    lines.append('\'train_conv_chains\'=[')
    for conv_chain in mconvs['train_conv_chains']:
        lines.append(chain_to_string(conv_chain))
        lines.append(',')
    lines.append('],')
    lines.append('\'train_bseqs\'={0},'.format(lol(mconvs['train_bseqs'])))
    lines.append('\'train_emissions\'={0},'.format(lol(mconvs['train_emissions'])))

    lines.append('\'test_length\'={0},'.format(mconvs['test_length']))
    lines.append('\'test_conv_chains=[')
    for conv_chain in mconvs['test_conv_chains']:
        lines.append(chain_to_string(conv_chain))
        lines.append(',')
    lines.append('],')
    lines.append('\'test_bseqs\'={0},'.format(lol(mconvs['test_bseqs'])))
    lines.append('\'test_emissions\'={0},'.format(lol(mconvs['test_emissions'])))

    lines.append('\'W\'={0},'.format(lol(mconvs['W'])))
    lines.append('\'h\'={0}'.format(list(mconvs['h'])))

    lines.append(')\n')
    return '\n'.join(lines)


def save_mixed_convs(mconvs, path='test/'):
    if not os.path.exists(path):
        os.makedirs(path)
    print 'Saving mconvs to path \'{0}\''.format(path)

    with open(path + 'model.params', 'w') as fout:
        fout.write(mixed_convs_to_string(mconvs))

    with open(path + 'states.txt', 'w') as fout:
        for binary_state in mconvs['train_bseqs']:
            for bit in binary_state:
                fout.write(' {0}'.format(bit))
            fout.write('\n')
    with open(path + 'obs.txt', 'w') as fout:
        for observation in mconvs['train_emissions']:
            for obs in observation:
                fout.write(' {0}'.format(obs))
            fout.write('\n')

    # could also save test_bseqs as test_states,
    # but decided not needed, just as we're not in
    # hdp_hmm_let synth figures

    with open(path + 'test_obs.txt', 'w') as fout:
        for observation in mconvs['test_emissions']:
            for obs in observation:
                fout.write(' {0}'.format(obs))
            fout.write('\n')

    with open(path + 'weights.txt', 'w') as fout:
        for observation in mconvs['W']:
            for obs in observation:
                fout.write(' {0}'.format(obs))
            fout.write('\n')
    print 'DONE.'
'''


# --------------------------------------------------------------------


def statistics_of_generated_data(data_path='../figures/cocktail/h0.5/cp0/',
                                 save_path=None,
                                 pprint=True):
    """
    Reports var, std, precision statistics of generated figures
    Column == microphone channel
    :param data_path:
    :return:
    """

    def get_stats(data, filename, decimal_precision=3):
        dfmtr = '{{0:.{0}f}}'.format(decimal_precision)
        header = [ '', 'Total' ] + [ '{0}'.format(i).ljust(decimal_precision)
                                     for i in range(data.shape[1]) ]
        table = prettytable.PrettyTable(header)
        for h in header:
            table.align[h] = 'r'

        data_var = np.var(data)
        data_var_by_column = np.var(data, axis=0)
        data_precision = 1/data_var
        data_precision_by_column = np.array([ 1/v for v in data_var_by_column ])
        data_std = np.std(data)
        data_std_by_column = np.std(data, axis=0)
        data_mean = np.mean(data)
        data_mean_by_column = np.mean(data, axis=0)
        data_median = np.median(data)
        data_median_by_column = np.median(data, axis=0)
        data_min = np.min(data)
        data_min_by_column = np.min(data, axis=0)
        data_max = np.max(data)
        data_max_by_column = np.max(data, axis=0)

        table.add_row(['precision', dfmtr.format(data_precision)]
                      + [ dfmtr.format(v) for v in data_precision_by_column])
        table.add_row(['var', dfmtr.format(data_var)]
                      + [ dfmtr.format(v) for v in data_var_by_column])
        table.add_row(['std', dfmtr.format(data_std)]
                      + [ dfmtr.format(v) for v in data_std_by_column])
        table.add_row(['mean', dfmtr.format(data_mean)]
                      + [ dfmtr.format(v) for v in data_mean_by_column ])
        table.add_row(['median', dfmtr.format(data_median)]
                      + [ dfmtr.format(v) for v in data_median_by_column ])
        table.add_row(['min', dfmtr.format(data_min)]
                      + [ dfmtr.format(v) for v in data_min_by_column ])
        table.add_row(['max', dfmtr.format(data_max)]
                      + [ dfmtr.format(v) for v in data_max_by_column ])

        stats_dict = dict(filename=filename,
                          precision_all=data_precision, precision_by_column=data_precision_by_column,
                          var_all=data_var, var_by_column=data_var_by_column,
                          std_all=data_std, std_by_column=data_std_by_column,
                          mean_all=data_mean, mean_by_column=data_mean_by_column,
                          median_all=data_median, median_by_colum=data_median_by_column,
                          min_all=data_min, min_by_column=data_min_by_column,
                          max_all=data_max, max_by_column=data_max_by_column)

        if pprint:
            print "\nStatistics for {0} from '{1}'".format(stats_dict['filename'], data_path)
            print table

        return table, stats_dict

    emissions_path = data_path + 'emissions/'

    weights = np.loadtxt(data_path + 'weights.txt')
    weights_table, weights_stats_dict = get_stats(weights, 'weights.txt')

    lcls_table, lcls_stats_dict = None, None
    if os.path.isfile(emissions_path + 'linear_combined_latent_state.txt'):
        lcls = np.loadtxt(emissions_path + 'linear_combined_latent_state.txt')
        lcls_table, lcls_stats_dict = get_stats(lcls, 'linear_combined_latent_state.txt')

    lclscs_table, lclscs_stats_dict = None, None
    if os.path.isfile(emissions_path + 'linear_combined_latent_state_centered_scaled.txt'):
        lclscs = np.loadtxt(emissions_path + 'linear_combined_latent_state_centered_scaled.txt')
        lclscs_table, lclscs_stats_dict = get_stats(lclscs, 'linear_combined_latent_state_centered_scaled.txt')

    noise_table, noise_stats_dict = None, None
    if os.path.isfile(emissions_path + 'noise.txt'):
        noise = np.loadtxt(emissions_path + 'noise.txt')
        noise_table, noise_stats_dict = get_stats(noise, 'noise.txt')

    obs_table, obs_stats_dict = None, None
    if os.path.isfile(data_path + 'obs.txt'):
        obs = np.loadtxt(data_path + 'obs.txt')
        obs_table, obs_stats_dict = get_stats(obs, 'obs.txt')

    obs_test_table, obs_test_stats_dict = None, None
    if os.path.isfile(data_path + 'test_obs.txt'):
        obs_test = np.loadtxt(data_path + 'test_obs.txt')
        obs_test_table, obs_test_stats_dict = get_stats(obs_test, 'test_obs.txt')

    if save_path:
        with open(save_path + 'statistics.txt', 'w') as fout:
            d = dict(lcls=lcls_stats_dict, lclscs=lclscs_stats_dict, noise=noise_stats_dict,
                     obs=obs_stats_dict, obs_test=obs_test_stats_dict, weights=weights_stats_dict)
            fout.write('dict=({0})'.format(d))
        with open(save_path + 'statistics_human.txt', 'w') as fout:
            fout.write("\nStatistics for {0} from '{1}'\n".format(weights_stats_dict['filename'], data_path))
            fout.write("{0}\n".format(weights_table))
            if lcls_stats_dict:
                fout.write("\nStatistics for {0} from '{1}'\n".format(lcls_stats_dict['filename'], data_path))
                fout.write("{0}\n".format(lcls_table))
            if lclscs_stats_dict:
                fout.write("\nStatistics for {0} from '{1}'\n".format(lclscs_stats_dict['filename'], data_path))
                fout.write("{0}\n".format(lclscs_table))
            if noise_stats_dict:
                fout.write("\nStatistics for {0} from '{1}'\n".format(noise_stats_dict['filename'], data_path))
                fout.write("{0}\n".format(noise_table))
            if obs_stats_dict:
                fout.write("\nStatistics for {0} from '{1}'\n".format(obs_stats_dict['filename'], data_path))
                fout.write("{0}\n".format(obs_table))
            if obs_test_stats_dict:
                fout.write("\nStatistics for {0} from '{1}'\n".format(obs_test_stats_dict['filename'], data_path))
                fout.write("{0}\n".format(obs_test_table))

'''
statistics_of_generated_data(data_path='../figures/cocktail/h0.5/cp0/',
                             save_path='../figures/cocktail/h0.5/cp0/')
'''

'''
statistics_of_generated_data(data_path='../figures/cocktail/a1b1_orig/cp0/',
                             save_path='../figures/cocktail/a1b1_orig/cp0/')
'''


# --------------------------------------------------------------------


def generate_random_cocktail_parties\
                (num_parties=10,
                 part_num_offset=0,
                 train_length=400,
                 test_length=400,
                 speaker_groups=(3, 2, 2),  # list of num_speaker per conv
                 num_microphones=12,

                 spacing_omega=0.8,

                 # initial state and transition matrix
                 # pi0_j, A_ij ~ Dirichlet(k, beta)
                 k=2,  # concentration
                 beta=None,  # Dir mean; None defaults to uniform

                 # super state duration
                 # D ~ Poisson(omega)
                 # omega ~ Gamma(a_omega, b_omega)
                 a_omega=3.0,  # shape
                 b_omega=1.0,  # rate

                 # Normal emission noise precision
                 # h ~ Gamma(a_h, b_h)
                 # mean = shape/scale
                 # var = (shape/scale^2)
                 # make a=3.0, b=2.0, b=6.0 --- 3 of each; 10 inference
                 a_h=1.0,  # h prior ~ Gamma shape param
                 b_h=1.0,  # h prior ~ Gamma scale param

                 # optionally hard-code h precision (same precision for each microphone)
                 h=None,

                 center_scale_data_p=True,

                 save_statistics_p=True,
                 data_dir='figures/'):

    # timestamp = util.get_timestamp()

    h_generate = False
    if h is None:
        h_generate = True
    else:
        if isinstance(h, (list, tuple)):
            if len(h) != num_microphones:
                print "length of specified h list/tuple is {0} != num_microphones {1}"\
                    .format(len(h), num_microphones)
        else:
            print h
            h = [ float(h) ] * num_microphones
        h = np.array(h)
        print "Manually specifying precision vector h={0}".format(h)

    for p in range(part_num_offset, num_parties + part_num_offset):

        print 'Generating party {0}'.format(p)

        # emission weight matrix
        W = sample_emission_weight_matrix(num_speakers=sum(speaker_groups),
                                          num_microphones=num_microphones)

        if h_generate:
            # Normal noise precision parameter
            h = sample_precision(num_microphones=num_microphones, a_h=a_h, b_h=b_h)

        # Sample parameters for each conversation
        conversations = [ sample_conversation_params\
                          (state_size=num_speakers, spacing_omega=spacing_omega,
                           k=k, beta=beta, a_omega=a_omega, b_omega=b_omega)
                          for num_speakers in speaker_groups ]

        mixed_conversation = sample_mix_conversations\
            (conversations, W=W, h=h,
             center_scale_data_p=center_scale_data_p,
             num_train=train_length, num_test=test_length)

        data_path = data_dir + 'cp{0}/'.format(p)

        print '    Saving to {0}'.format(data_path)

        mixed_conversation.save_to_file(path=data_path)

        if save_statistics_p:
            print 'Statistics {0}'.format(data_path)
            statistics_of_generated_data(data_path=data_path, save_path=data_path, pprint=True)
            print '============================================='

    print 'DONE.'


# --------------------------------------------------------------------


conv1 = dict(state_size=3,
             pi0=np.array([0.2, 0.1, 0.7]),
             A=np.array([[0.5, 0.2, 0.3],
                         [0.1, 0.5, 0.4],
                         [0.2, 0.3, 0.5]]),
             omegas=np.array([1.0, 4.817171822149042, 5.5601723775863947, 2.5607154594305161]),
             # mu=np.array([ 0.30937712,  1.59461785, -3.88594726]),
             # h=np.array([ 2.55156268,  1.11594631,  0.43530385])
             )

conv2 = dict(state_size=3,
             pi0=np.array([0.54764902, 0.14471936, 0.30763162]),
             A=np.array([[0.40228973, 0.09041656, 0.50729371],
                         [0.26140704, 0.69234323, 0.04624973],
                         [0.34456736, 0.27026315, 0.38516949]]),
             omegas=np.array([0.8, 2.5654997295488142, 3.914977736772633, 2.952841666362703]))

conv3 = dict(state_size=2,
             pi0=np.array([0.55201095, 0.44798905]),
             A=np.array([[0.429508, 0.570492],
                         [0.79389941, 0.20610059]]),
             omegas=np.array([0.8, 5.1679804, 1.57983888]))

conv4 = dict(state_size=2,
             pi0=np.array([0.36188255, 0.63811745]),
             A=np.array([[0.18148085, 0.81851915],
                         [0.45375021, 0.54624979]]),
             omegas=np.array([0.8, 1.58805976, 1.60533681]))

def conv_to_string(conv):
    lines = list()
    lines.append('dict(')
    lines.append('state_size={0},'.format(conv['state_size']))
    lines.append('pi0={0},'.format(list(conv['pi0'])))
    lines.append('A={0},'.format(list(conv['A'])))
    lines.append('omegas={0},'.format(list(conv['omegas'])))
    lines.append(')\n')
    return ''.join(lines)


# 7 speakers (plust last row as bias), 12 microphones
W = np.array([[0.44730875, 0.5445322, 0.66902356, 0.89881059, 0.59440348,
               0.79898535, 0.41624634, 0.06374723, 0.58569319, 0.05732156,
               0.52392386, 0.85074085],
              [0.60205858, 0.06609026, 0.26333553, 0.07591228, 0.59104334,
               0.91236355, 0.66675098, 0.51560989, 0.76581559, 0.04037277,
               0.31198102, 0.5965149],
              [0.14799197, 0.27793634, 0.51347782, 0.04405109, 0.13051354,
               0.77605942, 0.35441366, 0.64046593, 0.79168121, 0.23638308,
               0.50950684, 0.29149228],
              [0.28766934, 0.48714957, 0.72787208, 0.17440725, 0.84341386,
               0.68504428, 0.68842865, 0.51234954, 0.21939325, 0.56321382,
               0.95093912, 0.12644179],
              [0.29329641, 0.29844203, 0.43860911, 0.0501127, 0.98255521,
               0.66569858, 0.16561445, 0.07713743, 0.62682286, 0.35478735,
               0.64969518, 0.21736584],
              [0.4394461, 0.01742136, 0.83725971, 0.25769161, 0.26099183,
               0.1254114, 0.08710049, 0.33415062, 0.09656484, 0.70485597,
               0.01256662, 0.65997307],
              [0.36744144, 0.24460989, 0.59132815, 0.43615986, 0.69732223,
               0.48142677, 0.13667896, 0.99046782, 0.52529154, 0.63944954,
               0.42638539, 0.42357734],
              [0.97607545, 0.96483876, 0.27654596, 0.43097701, 0.98587226,
               0.12642987, 0.18158992, 0.73133826, 0.75053224, 0.53184757,
               0.02929165, 0.76506032]])

h = np.array([0.89909425, 1.35087469, 1.01604911, 1.52605192, 1.4458019,
              2.94293135, 0.12031849, 0.33655889, 1.62353987, 0.50928693,
              2.83470432, 2.2805566])

bin_state = np.array([0, 0, 1, 1, 0, 0, 1])


'''
sigma2_0 = 2.0
mu = np.random.normal(loc=0, scale=sigma2_0, size=num_speakers)
a_h = 1.0
b_h = 1.0
h = np.random.gamma(shape=a_h, scale=b_h, size=num_speakers)
'''

# mu = calculate_means(bin_state, W)
# print mu

# emissions = normal_emission(mu, h)
# print emissions

# --------------------------------------------------------------------

# script

# mconvs = sample_mix_conversations([conv1, conv3, conv4], W, h, n=300)
# save_mixed_convs(mconvs, path='../figures/cocktail_party/')


# noise_a1b1
def generate_noise(a_h=1.0, b_h=1.0, h=None,
                   speaker_groups=(3, 2, 2),
                   num_microphones=12,
                   num_parties=10, part_num_offset=0,
                   center_scale_data_p=False,
                   data_dir=None):
    if data_dir is None:
        cs = ''
        if not center_scale_data_p:
            cs='nocs'
        if h is None:
            data_dir = '../figures/cocktail/a{0}b{1}_{2}/'\
                .format(int(np.floor(a_h)), int(np.floor(b_h)), cs)
        else:
            data_dir = '../figures/cocktail/h{0}_{1}/'.format(h, cs)
    generate_random_cocktail_parties \
        (num_parties=num_parties,
         part_num_offset=part_num_offset,
         train_length=400,
         test_length=400,
         speaker_groups=speaker_groups,  # list of num_speaker per conv
         num_microphones=num_microphones,

         spacing_omega=0.8,

         # initial state and transition matrix
         # pi0_j, A_ij ~ Dirichlet(k, beta)
         k=2,  # concentration
         beta=None,  # Dir mean; None defaults to uniform

         # super state duration
         # D ~ Poisson(omega)
         # omega ~ Gamma(a_omega, b_omega)
         a_omega=3.0,  # shape
         b_omega=1.0,  # rate

         # Normal emission noise precision
         # h ~ Gamma(a_h, b_h)
         a_h=a_h,  # h prior ~ Gamma shape param
         b_h=b_h,  # h prior ~ Gamma rate param

         # optionally manually specify value used for all precision in precision vector h
         h=h,

         center_scale_data_p=center_scale_data_p,

         data_dir=data_dir)


# generate_noise_a3b2()
# generate_noise_a3b6()

# generate_noise(a_h=1.0, b_h=1.0, num_parties=3)
# generate_noise(a_h=6.0, b_h=10.0, num_parties=3)
# generate_noise(a_h=20.0, b_h=76.0, num_parties=3)

# generate non-centered/scaled figures
# generate_noise(a_h=1.0, b_h=1.0, center_scale_data_p=False, num_parties=3)


'''
for h in [ 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 ]:
    generate_noise(h=h, num_parties=3)
'''

'''
for h in [ 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 ]:
    generate_noise(h=h, center_scale_data_p=False, num_parties=3, part_num_offset=3)
'''

'''
# generating additional noise level datasets, using offset...
for h in [ 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 ]:
    generate_noise(h=h, center_scale_data_p=False, num_parties=4, part_num_offset=6)
'''

'''
for h in [ 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 ]:  # [ 0.5 ]:
    generate_noise(h=h,
                   center_scale_data_p=False,
                   speaker_groups=(4, 4, 4, 4),
                   num_microphones=12,
                   num_parties=10,
                   data_dir='../figures/cocktail_s16_m12/h{0}_nocs/'.format(h))
'''

'''
for h in [ 2.0 ]:  # [ 0.5 ]:
    generate_noise(h=h,
                   center_scale_data_p=False,
                   speaker_groups=(5, 5, 4),
                   num_microphones=12,
                   num_parties=10,
                   data_dir='../figures/cocktail_s14_m12/h{0}_nocs/'.format(h))
'''


# --------------------------------------------------------------------

