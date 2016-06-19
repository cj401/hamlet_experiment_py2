import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import cocktail_party_data

__author__ = 'clayton'


def gamma_to_variance_properties(a_h, b_h):
    """
    A not super helpful helper to get an idea of affect of parameters of
    Gamma prior distribution over precision, on variance
    :param a_h:
    :param b_h:
    :return:
    """
    m_h = float(a_h) / float(b_h)
    mode_h = float(b_h) / (float(a_h) + 1.0)
    var_h = float(m_h) / float(b_h)

    # These come from Inverse Gamma distribution, since the Gamma
    # is prior over the _precision_, and here we want to see the
    # effects on the variance, where variance = 1/precision
    if a_h > 1:
        exp_noise_variance_mean = b_h / (a_h - 1)  # 1.0 / m_h
    if a_h > 2:
        exp_noise_variance_var = b_h**2 / ((a_h - 1)*(a_h - 2))  # 1.0 / var_h

    print "Gamma shape a_h= {0}".format(a_h)
    print "Gamma  rate a_h= {0}".format(b_h)
    print "Gamma mean = {0}".format(m_h)
    print "Gamma mode = {0}".format(mode_h)
    print "Gamma  var = {0}".format(var_h)
    if a_h <= 1:
        print "Moment not defined: expected Variance mean = infinity"
    else:
        print "expected Variance mean = {0}".format(exp_noise_variance_mean)
    if a_h <= 2:
        print "Moment not defined: expected Variance var = infinity"
    else:
        print "expected Variance  var = {0}".format(exp_noise_variance_var)


'''
# http://matplotlib.org/examples/mplot3d/surface3d_demo.html
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2, Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''


# prec_sample = sample_precision(1000, a_h=1.0, b_h=1.0)

def estimate_expected_precision_mean_stdev(a_h=1.0, b_h=1.0, num_samples=100000,verbose=False):

    # calculate the empirical mean and std of precision samples from Gamma(a_h, 1/b_h)
    precision_samples = cocktail_party_data.sample_precision(num_samples, a_h=a_h, b_h=b_h)
    mu = np.mean(precision_samples)
    std = np.std(precision_samples)

    # calculate the empirical sigma (variance) mean and std based on precision samples
    variance_samples = np.array([ 1/x for x in precision_samples ])
    var_mu = np.mean(variance_samples)
    var_std = np.std(variance_samples)

    if verbose:

        analytic_precision_mu   = float(a_h) / float(b_h)
        analytic_precision_std  = np.sqrt( analytic_precision_mu / float(b_h))
        analytic_precision_mode = 'undefined'
        if a_h >= 1.0:
            analytic_precision_mode = ( float(a_h) - 1.0 ) / float(b_h)

        analytic_variance_mean = 'infinity'
        if a_h > 1:
            analytic_variance_mean = b_h / ( a_h - 1.0 )
        analytic_variance_std = 'infinity'
        if a_h > 2:
            analytic_variance_std = np.sqrt( b_h**2 / ( ((a_h - 1)**2) * (a_h - 2)) )
        analytic_variance_mode = b_h / ( a_h + 1.0 )

        print 'Gamma(a_h={0}, b_h={1}):\n'.format(a_h, b_h) \
              + '    [precision: mu={0}, stdev={1} ; a_mu={2}, a_std={3}, a_mode={4}]\n' \
                  .format(mu, std, analytic_precision_mu, analytic_precision_std, analytic_precision_mode) \
              + '    [ variance: mu={0}, stdev={1} ; a_mu={2}, a_std={3}, a_mode={4}]' \
                  .format(var_mu, var_std, analytic_variance_mean, analytic_variance_std, analytic_variance_mode)

    return mu, std, var_mu, var_std


def plot_precision_and_variance():
    a_h_levels = list()
    for a_h in np.arange(4, 8.5, 0.5):
        x = np.linspace(2.0, 12.1, 100)
        a_h_level = dict(label='{0}'.format(a_h),
                         x=x,
                         mu=list(),
                         std=list(),
                         var_mu=list(),
                         var_std=list())
        for b_h in x:
            mu, std, var_mu, var_std = estimate_expected_precision_mean_stdev(a_h, b_h, verbose=False)
            a_h_level['mu'].append(mu)
            a_h_level['std'].append(std)
            a_h_level['var_mu'].append(var_mu)
            a_h_level['var_std'].append(var_std)
        a_h_levels.append(a_h_level)

    def plot_a_h_levels(ax, a_h_levels, stat):

        for a_h_level in a_h_levels:
            ax.plot(a_h_level['x'], a_h_level[stat], label=a_h_level['label'])

        ax.set_xlabel('b_h (rate)')
        ax.set_ylabel(stat)
        ax.legend(loc='upper left')

    def plot2vars(ax, a_h_levels):

        color_cycle = ['r', 'g', 'b', 'y', 'c', 'm']

        ax.set_color_cycle(color_cycle)
        for a_h_level in a_h_levels:
            ax.plot(a_h_level['x'], a_h_level['var_mu'], label=a_h_level['label'])

        ax.set_color_cycle(color_cycle)
        for a_h_level in a_h_levels:
            ax.plot(a_h_level['x'], a_h_level['var_std'], linestyle='--')

        ax.set_xlabel(r'$b_h$ (rate)')
        ax.set_ylabel(r'Noise Variance $\mathbb{E}[\mu]$ and $\mathbb{E}[\sigma]$')
        ax.set_title(r'Noise Variance $\mathbb{E}[\mu]$ and $\mathbb{E}[\sigma]$'
                     + r' as a function of Gamma($a_h$, $\frac{1}{b_h}$)')
        ax.legend(loc='upper left', title=r'$a_h$ (scale)')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')

    for stat, ax in [ ('mu', ax1), ('std', ax2), ('var_mu', ax3), ('var_std', ax4) ]:
        plot_a_h_levels(ax, a_h_levels, stat)

    fig = plt.figure()
    ax = fig.gca()
    plot2vars(ax, a_h_levels)

    plt.show()

# plot_precision_and_variance()


def get_variance_a_b_given_mean_var(mu, sigma):
    """
    The mean and standard dev (std) of variance sample from Gamma(a_h, 1/b_h) is
    mu (mean) = b_h / ( a_h - 1.0 )
    sigma (std) = \sqrt{ b_h^2 / ( ((a_h - 1)^2) * (a_h - 2)) }

    This function finds a_h and b_h given a desired mu and sigma:
    a_h = \mu^2 / \sigma^2
    b_h = (\mu^3 / \sigma^2) - \mu

    :param mu:
    :param sigma:
    :return: a_h, b_h
    """
    return (mu**2 / sigma**2), (mu**3 / sigma**2) - mu


def test_analytic_a_b(mu, sigma):
    print 'mu={0}, sigma={1}'.format(mu, sigma)
    a_h, b_h = get_variance_a_b_given_mean_var(mu, sigma)
    print 'a_h={0}, b_h={1}'.format(a_h, b_h)
    estimate_expected_precision_mean_stdev(a_h=a_h, b_h=b_h, verbose=True)
    print '------------------------------------------'

'''
test_analytic_a_b(1.5, 0.5)
test_analytic_a_b(2.0, 0.65)
test_analytic_a_b(3.0, 0.8)
test_analytic_a_b(4.0, 0.8)
'''

'''
test_analytic_a_b(1.5, 0.74)
test_analytic_a_b(2.0, 0.83)
test_analytic_a_b(3.0, 0.9)
test_analytic_a_b(4.0, 1.0)
'''

'''
estimate_expected_precision_mean_stdev(a_h=1.0, b_h=1.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=4.5, b_h=5.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=6.0, b_h=10.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=11.0, b_h=30.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=20.0, b_h=76.0, verbose=True)
'''

'''
estimate_expected_precision_mean_stdev(a_h=4.1, b_h=4.6, verbose=True)
estimate_expected_precision_mean_stdev(a_h=4.5, b_h=5.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=6.0, b_h=10.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=11.0, b_h=30.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=16.0, b_h=60.0, verbose=True)
# (4.1, 4.6), (6.0, 10.0), (11.0, 30.0), (16.0, 60.0)

print '1.5 4.1 / 4.6   |', 4.1 / 4.6
print '1.5 4.5 / 5.0   |', 4.5 / 5.0
print '2.0 6.0 / 10.0  |', 6.0 / 10.0
print '3.0 11.0 / 30.0 |', 11.0 / 30.0
print '4.0 16.0 / 60.0 |', 16.0 / 60.0
'''


'''
estimate_expected_precision_mean_stdev(a_h=1.0, b_h=1.0, verbose=True)
print
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=2.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=3.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=4.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=5.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=6.0, verbose=True)
print
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=1.0/2.0, verbose=True)
estimate_expected_precision_mean_stdev(a_h=3.0, b_h=1.0/6.0, verbose=True)
'''


def plot_variance_histograms(ab_list, num_samples=10000, num_bins=100, plt_max=None):
    fig = plt.figure()
    ax = fig.gca()
    samples = list()
    all_data = list()
    for a_h, b_h in ab_list:
        precision_samples = cocktail_party_data.sample_precision(num_samples, a_h=a_h, b_h=b_h)
        variance_samples = np.array([ 1/x for x in precision_samples ])
        samples.append((a_h, b_h, variance_samples))
        if a_h != 1.0 and b_h != 1.0:
            all_data += list(variance_samples)

    if not plt_max:
        plt_max = max(all_data)

    bins = np.linspace(0, plt_max, num_bins)

    for a_h, b_h, s in samples:
        mu = np.mean(s)
        std = np.std(s)
        vmax = np.max(s)
        num_gt_max = len(s[np.where( s > plt_max )])
        plt.hist(s, bins, alpha=0.5,
                 label=r'$a_h={0}$, $b_h={1}$'.format(a_h, b_h) \
                       + r', $\widehat{{\mu}}={0:.2f}$'.format(mu) \
                       + r', $\widehat{{\sigma}}={0:.2f}$'.format(std) \
                       + r', $n>{0}={1}$'.format(plt_max, num_gt_max) \
                       + r', max=${0:.2f}$'.format(vmax)
                 )

    ax.set_xlabel('Variance ($\sigma$)')
    ax.set_ylabel('Variance frequency')
    ax.set_title(r'{0} samples of $\sigma$ ~ $Gamma(a_h,$ $\frac{{1}}{{b_h}})$'
                 .format(num_samples))

    ax.legend(loc='upper right', prop={'size':10})


# preferred params: (6.0, 10.0), (20.0, 76.0)
'''
plot_variance_histograms([(1.0, 1.0), (4.5, 5.0), (6.0, 10.0), (11.0, 30.0), (20.0, 76.0)],
                         plt_max=12)

plot_variance_histograms([(1.0, 1.0), (4.1, 4.6), (6.0, 10.0), (11.0, 30.0), (16.0, 60.0)],
                         plt_max=12)
plt.show()
'''


def calc_precision_surf(a_min=0.25, a_max=6.0, b_min=0.25, b_max=6.0, step=0.25, num_samples=10000):
    X = np.arange(a_min, a_max, step)
    Y = np.arange(b_min, b_max, step)
    x_len = len(X)
    y_len = len(Y)
    mus = np.zeros((y_len, x_len))
    stds = np.zeros((y_len, x_len))
    for y, yi in zip(Y,range(y_len)):
        for x, xi in zip(X, range(x_len)):
            mu, std, var_mu, var_std = estimate_expected_precision_mean_stdev(x, y, num_samples)
            mus[yi][xi] = mu
            stds[yi][xi] = std
            #print mu, std
    return mus, stds, X, Y


def calc_variance_surf(a_min=0.25, a_max=6.0, b_min=0.25, b_max=6.0, step=0.25, num_samples=10000):
    X = np.arange(a_min, a_max, step)
    Y = np.arange(b_min, b_max, step)
    x_len = len(X)
    y_len = len(Y)
    mus = np.zeros((y_len, x_len))
    stds = np.zeros((y_len, x_len))
    for y, yi in zip(Y,range(y_len)):
        for x, xi in zip(X, range(x_len)):
            #print x,y
            samples = cocktail_party_data.sample_precision(num_samples, a_h=x, b_h=y)

            samples = np.array([ 1/p for p in samples ])

            mu = np.mean(samples)
            mus[yi][xi] = mu
            std = np.std(samples)
            stds[yi][xi] = std
            #print mu, std
    return mus, stds, X, Y


def plot_surf(Z, X=np.arange(-5, 5, 0.25), Y=np.arange(-5, 5, 0.25), z_min=0.0, z_max=6.0, title=''):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           norm=colors.LogNorm(0.01, 4),  # (vmin=np.amin(Z), vmax=np.amax(Z)),
                           cmap=cm.ScalarMappable().get_cmap(),  # norm=colors.LogNorm(), cmap=cm.coolwarm  cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(z_min, z_max)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel('a_h (shape)')
    ax.set_ylabel('b_h (rate)')
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=5)


def generate_precision_plots():
    mu, std, X, Y = calc_precision_surf(a_min=0.05, a_max=6.0, b_min=0.8, b_max=6.0, step=0.25)
    # mu, std, X, Y = calc_variance_surf(a_min=0.05, a_max=1.0, b_min=0.05, b_max=1.0, step=0.25)

    print np.amin(mu), np.amax(mu)
    print np.amin(std), np.amax(std)
    plot_surf(mu, X, Y, title=r'precision $\mu$', z_max=np.amax(mu))
    plot_surf(std, X, Y, title=r'precision $\sigma$', z_max=np.amax(std),)

    plt.show()
