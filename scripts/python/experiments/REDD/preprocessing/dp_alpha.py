import numpy
import scipy.optimize


def get_objective_fn(d, n):
    """
    Returns the objective function for computing the value of
    the relationship between
        alpha (concentration),
        d (expected number of latent states)
        n (number of observations)
    in a Dirichlet Process:
        d = alpha * log(1 + (n / alpha))
    :param d: expected number of latent states
    :param n: number of observations
    :return: Function of alpha
    """
    def objective(alpha):
        return float(d) - (alpha * numpy.log(float(n) / alpha))
    return objective


def get_objective_fprime1(n):
    """
    Return first derivative of objective fn
    :param n: number of observations
    :return: Function of alpha ; first derivative
    """
    def objective(alpha):
        return 1. - numpy.log(n / alpha)
    return objective


def fprime2(alpha):
    """
    Second derivative of objective fn
    :param alpha: concentration parameter
    :return: 1/alpha
    """
    return 1. / alpha


def compute_alpha(d, n, alpha0=0.5, maxiter=5000, verbose=False):
    """
    Approximates the value of the Dirichlet Process parameter alpha
     according to this relationship:
        d = alpha * log(1 + (n / alpha))
     where
        alpha (concentration),
        d (expected number of latent states)
        n (number of observations)
    Uses numpy.optimize.newton to approximate the value of alpha.
    Uses three functions:
        f(alpha):       d - (alpha * log(1 + (n / alpha)))   objective fn
        fprime1(alpha): 1 - log(n / alpha)                   1st derivative fn
        fprime2(alpha): 1 / alpha                            2nd derivative fn
    :param d: expected number of latent states
    :param n: number of observations
    :param alpha0: should not be greater or equal to 2*d
    :param maxiter: maximum number of iterations of newton to run
    :param verbose: (default=False)
    :return:
    """
    fn = get_objective_fn(d, n)
    fprime1 = get_objective_fprime1(n)
    if verbose:
        print 'Starting...',
    alpha = scipy.optimize.newton(func=fn, x0=alpha0, fprime=fprime1, fprime2=fprime2,
                                  maxiter=maxiter)
    if verbose:
        print 'Done!  alpha=', alpha
    return alpha


# ------------------------------------------------------------

def test(n=5000):

    # NOTE: this first version fails when alpha0 >= 4.0
    for alpha0 in (0.5, 0.6, 1.0, 1.5, 3.9):
        print '{0}'.format(alpha0),
        compute_alpha(2, n, alpha0=alpha0, verbose=True)

    # NOTE: this first version fails when alpha0 >= 8.0
    for alpha0 in (0.5, 0.6, 1.0, 1.5, 4.0, 5.0, 7.9):
        print '{0}'.format(alpha0),
        compute_alpha(4, n, alpha0=alpha0, verbose=True)

    for alpha0 in (0.5, 0.6, 1.0, 1.5, 4.0, 8.0):
        print '{0}'.format(alpha0),
        compute_alpha(8, n, alpha0=alpha0, verbose=True)

    for alpha0 in (0.5, 0.6, 1.0, 1.5, 4.0, 8.0):
        print '{0}'.format(alpha0),
        compute_alpha(30, n, alpha0=alpha0, verbose=True)

    for alpha0 in (0.5, 0.6, 1.0, 1.5, 4.0, 8.0):
        print '{0}'.format(alpha0),
        compute_alpha(300, n, alpha0=alpha0, verbose=True)

# test()
