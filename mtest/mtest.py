# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import scipy as sp
import scipy.io
from scipy import stats
import os
import pymc
import mdp.utils

# TODO: remove progressbar dependency (mdp.utils)
# TODO: npy files instead of .mat
# TODO: module documentation
# TODO: profile
# TODO: remove print statements, replace with logging

# templates for the names of the tables
TABLESNAME = 'bayes_ttest_table_n1_%d_n2_%d.mat'
TYPEII_TABLESNAME = 'bayes_ttest_typeII_n1_%d_n2_%d_mdist_%.2f_scale_%.2f.mat'

_DISTR_STD = 1.0
_NPRIOR = 1500


def get_tables_path(path=None):
    """Return the path were the m-test stores its tables.

    The directory is used to store the tables used to compute p-values
    and power of the m-test for different population sizes.

    By default, the data path is set to `MTEST_PATH/tables/`, where
    MTEST_PATH is the path where the mtest package is
    installed. Alternatively, it can be set by the `MTEST_TABLES_PATH`
    environment variable.

    If the directory does not exist, it is created automatically.
    """
    if path is None:
        path = os.environ.get('MTEST_TABLES_PATH',
                              os.path.join(os.path.dirname(__file__),
                                           'tables/'))
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# ============ mtest_marginal_likelihood_ratio

def _marginalize_prior(model, n):
    """Computes the marginal likelihood by integrating over the prior.

    Parameters
    ----------
    model : PyMC model
    n : number of samples from the prior used to compute the integral

    Returns
    -------
    mlhood : estimate of the marginal likelihood
    """

    sm = 0.
    for i in range(n):
        model.draw_from_prior()
        for obs_distr in model.observed_stochastics:
            sm += sp.exp(obs_distr.logp)
    return sm/n

def _prior_from_data(data, sgm_max=3.0):
    """Generate PyMC prior over mean and standard deviation given the data.

    The data is used to inform the prior over the mean: the
    distribution is centered at the empirical mean and its standard
    deviation is the standard error of the mean.

    The prior over the standard deviation is a uniform distribution
    between 10^-3 and `sgm_max`.

    Parameters
    ----------
    data : data to be modeled
    sgm_max: upper limit for the standard deviation

    Returns
    -------
    muy : PyMC model over the mean (a Normal distribution)
    stdy : PyMC model over the standard deviation (a Uniform distribution)
    """

    # empirical mean and std of the mean
    mu_mean = data.mean()
    mu_std = 1/sp.sqrt(len(data))
    # build prior distributions
    muy = pymc.Normal('muy', mu_mean, 1./mu_std**2.)
    stdy = pymc.Uniform('stdy', 0.001, sgm_max)
    return muy, stdy

def mtest_marginal_likelihood_ratio(pop1, pop2, nprior=_NPRIOR):
    """Computes the model selection statistics for the m-test.

    This function returns the statistics for the m-test, namely the
    maximum of two marginal likelihood ratios, comparing two
    alternative models to a null model:

       ( P( M1 | data )  P( M2 | data ) )
    max( --------------, -------------- )
       ( P( M0 | data )  P( M0 | data ) )

    M0 is the null model, where the data comes from a common Gaussian
    distributions; in the first alternative model, M1, the data comes
    from two Gaussian distributions with different means but equal
    standard deviation; finally, in the second alternative model, M2,
    the data comes from two Gaussians with different means and
    standard deviations.

    Note that the probability of each model is computed by integrating
    out the parameters (i.e., the means and standard deviations),
    which avoids problems due to a different number of parameters in
    the models.

    Parameters
    ----------
    pop1, pop2 : 1D arrays, samples from the two populations
    nprior : number of samples from the prior over parameters, used
             to estimate the marginal likelihood integral

    Returns
    -------
    The value of the statistics. A value larger than 1 indicates a
    higher probability for the alternative models with different means
    than for the null model.
    """

    pops = sp.concatenate((pop1, pop2))
    # rescale to zero mean, unit variance
    shift, scale = pops.mean(), pops.std()
    pop1 = (pop1-shift)/scale
    pop2 = (pop2-shift)/scale
    pops = sp.concatenate((pop1, pop2))

    # M0
    muy, stdy = _prior_from_data(pops)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y(value=0., muy=muy, stdy=stdy):
        return pymc.normal_like(pops, muy, 1./(stdy**2.))
    
    model0 = pymc.Model((muy, stdy, y))
    marg_M0 = _marginalize_prior(model0, nprior)

    # M1
    mu1y, std1y = _prior_from_data(pop1, sgm_max=1.)
    mu2y, std2y = _prior_from_data(pop2, sgm_max=1.)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y2(value=0., mu1y=mu1y, std1y=std1y, mu2y=mu2y, std2y=std2y):
        ll = pymc.normal_like(pop1, mu1y, 1./(std1y**2.))
        ll += pymc.normal_like(pop2, mu2y, 1./(std2y**2.))
        return ll
    
    model1 = pymc.Model((mu1y, std1y, mu2y, std2y, y2))
    marg_M1 = _marginalize_prior(model1, nprior)

    # M2
    std3y = pymc.Uniform('std3y', 0.001, 1.)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y3(value=0., mu1y=mu1y, mu2y=mu2y, std3y=std3y):
        ll = pymc.normal_like(pop1, mu1y, 1./(std3y**2.))
        ll += pymc.normal_like(pop2, mu2y, 1./(std3y**2.))
        return ll

    model2 = pymc.Model((mu1y, mu2y, std3y, y3))
    marg_M2 = _marginalize_prior(model2, nprior)

    M1M0 = marg_M1/marg_M0
    M2M0 = marg_M2/marg_M0

    return max(M1M0, M2M0)


# ============ mtest

def mtest(x, y, nprior=_NPRIOR, min_ncases=50000, path=None):
    """Performs the m-test.

    The function computes the m-test statistics (i.e., the marginal
    likelihood ratio statistics computed in `mtest_marginal_likelihood_ratio`
    and compares it with the tables for the same number of samples
    as in `x` and `y`.

    If the tables do not exist, or they have been computed with a
    number of cases smaller than the one requested, they are computed
    from scratch, which might take some time (hours).

    Parameters
    ----------
    x : 1D array, data for first population
    y : 1D array, data for second population
    nprior : number of samples from the prior over parameters, used
             to estimate the marginal likelihood integral
             (see `mtest_marginal_likelihood_ratio`)
    min_ncases : minimum number of cases used to build the tables for
                 the test (default: 50000)
    path : path to the m-test tables (see `get_tables_path`)

    Returns
    -------
    m : statistics of the m-test
    prob : p-value
    ncases : number of cases in the table used to compute the p-value

    See also : `mtest_marginal_likelihood_ratio`
    """
    m = mtest_marginal_likelihood_ratio(x, y, nprior)
    test_values = typeI_table(x.shape[0], y.shape[0], min_ncases, path=path)
    ncases = len(test_values)
    prob = (test_values>m).sum() / float(test_values.shape[0])
    return m, prob, ncases


# ============ typeI_table

def _random_same_mean(n1, n2, ncases):
    """Return random samples from two populations with same mean and
    standard deviation.

    Generate a number of populations from Normal(0, 1).

    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate

    Returns
    -------
    pop1 : 2D array
           pop1[i, :] contains the `n1` samples from population number i
    pop2 : 2D array
           pop2[i, :] contains the `n2` samples from population number i
    """
    pop_distr = stats.norm(loc=0., scale=_DISTR_STD)
    pop1 = pop_distr.rvs(size=(ncases, n1))
    pop2 = pop_distr.rvs(size=(ncases, n2))
    return pop1, pop2

def typeI_table(n1, n2, ncases, path=None):
    """Return a table of the m-test statistics under the null hypothesis.

    The function returns a table containing the value of the
    m-statistics of `ncases` draws from two populations of size `n1`
    and `n2` under the null hypothesis that the mean of the two
    populations is the same.

    If a table for population sizes `n1` and `n2` with more entries than
    `ncases` exists, all the stored values are returned.
    Otherwise, new cases are computed and stored, then returned.

    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate
    path : path to the m-test tables (see `get_tables_path`)

    Returns
    -------
    test_values : 1D array of m-test statistics, containing *at least*
                  `ncases` elements, but possibly more
    """

    print 'N1 = %d, N2 = %d' % (n1, n2)

    fname = os.path.join(get_tables_path(path), TABLESNAME%(n1,n2))
    print fname
    if os.path.exists(fname):
        print 'loading', fname
        matdict = scipy.io.loadmat(fname)
        test_values = matdict['test_values'].flatten()
    else:
        test_values = sp.array([])
    
    nvalues = test_values.shape[0]
    if nvalues>=ncases:
        return test_values

    nmissing = ncases-nvalues
    print 'missing %d entries' % nmissing
    
    # compute missing entries
    pop1_test, pop2_test = _random_same_mean(n1, n2, nmissing)

    missing_values = sp.zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        missing_values[i] = mtest_marginal_likelihood_ratio(pop1_test[i,:],
                                                            pop2_test[i,:],
                                                            nprior=_NPRIOR)

    # update and save table
    test_values = sp.concatenate((test_values, missing_values))
    print 'saving', fname
    scipy.io.savemat(fname, {'test_values': test_values})

    return test_values


# ============ typeII_table

def _random_different_mean(n1, n2, ncases, mean, std):
    """Return random samples from two populations with different 
    standard deviation and different mean

    Generate a number of populations from Normal(mean, std^2) and
    Normal(0, 1).

    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate
    mean -- mean of population 1
    std -- standard deviation of population 1

    Returns
    -------
    pop1 : 2D array
           pop1[i, :] contains the `n1` samples from population number i
    pop2 : 2D array
           pop2[i, :] contains the `n2` samples from population number i
    """
    mean = mean*_DISTR_STD
    pop1_distr = stats.norm(loc=0., scale=_DISTR_STD*std)
    pop2_distr = stats.norm(loc=mean, scale=_DISTR_STD)
    pop1 = pop1_distr.rvs(size=(ncases, n1))
    pop2 = pop2_distr.rvs(size=(ncases, n2))
    return pop1, pop2

def typeII_table(n1, n2, ncases, mean, std, path=None):
    """Return a table of the m-test statistics under a specific hypothesis.

    The function returns a table containing the value of the
    m-statistics and (for comparison) the t-statistics (independent
    t-test) of `ncases` draws from two populations of size `n1` and
    `n2`, the first with distribution Normal(mean, std^2), and the
    second with distribution Normal(0, 1).

    The table is used to compute the power of the test under different
    conditions.
    
    If a table for population sizes `n1` and `n2` with more entries than
    `ncases` exists, all the stored values are returned.
    Otherwise, new cases are computed and stored, then returned.

    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate
    mean -- mean of population 1
    std -- standard deviation of population 1
    path : path to the m-test tables (see `get_tables_path`)

    Returns
    -------
    m_test_values : 1D array of m-test statistics, containing *at least*
                    `ncases` elements, but possibly more
    t_test_values : 1D array of t-test statistics, containing *at least*
                    `ncases` elements, but possibly more
    """

    print 'N1 = %d, N2 = %d' % (n1, n2)
    print 'mean = %f, scale factor for pop1 = %f' % (mean, std)

    fname = os.path.join(get_tables_path(path),
                         TYPEII_TABLESNAME%(n1,n2,mean,std))
    if os.path.exists(fname):
        print 'loading', fname
        matdict = scipy.io.loadmat(fname)
        # TODO: change to m_test_values
        m_test_values = matdict['bayes_test_values'].flatten()
        t_test_values = matdict['t_test_values'].flatten()
    else:
        m_test_values = sp.array([])
        t_test_values = sp.array([])
    
    nvalues = m_test_values.shape[0]
    if nvalues>=ncases:
        return m_test_values, t_test_values

    nmissing = ncases-nvalues
    print 'missing %d entries' % nmissing
    
    # compute missing entries
    pop1_test, pop2_test = _random_different_mean(n1, n2, nmissing, mean, std)

    m_missing_values = sp.zeros((nmissing,))
    t_missing_values = sp.zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        m_missing_values[i] = mtest_marginal_likelihood_ratio(pop1_test[i,:],
                                                              pop2_test[i,:],
                                                              nprior=_NPRIOR)
        t_missing_values[i] = stats.ttest_ind(pop1_test[i,:],
                                              pop2_test[i,:])[1]

    # update and save table
    m_test_values = sp.concatenate((m_test_values, m_missing_values))
    t_test_values = sp.concatenate((t_test_values, t_missing_values))
    print 'saving', fname
    scipy.io.savemat(fname, {'bayes_test_values': m_test_values,
                             't_test_values': t_test_values})

    return m_test_values, t_test_values


# ============ utils

def typeI_threshold(n1, n2, ncases, prob=0.05, path=None):
    """Threshold on the m-statistics to achieve a certain p-value.

    The function requests a table with `ncases` draws of size `n1` and
    `n2` (see `typeI_table`), and computes the threshold value such
    that the m-test would commit a Type I error (rejecting the null
    hypothesis even though the mean is equal) with probability `prob`.

    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate
    prob : the taget p-value
    path : path to the m-test tables (see `get_tables_path`)

    Returns
    -------
    The threshold for the m_test. m-tests of size `n1`, `n2` with m-statistics
    smaller than this value should be rejected as not significant.
    """
    typeI_values = typeI_table(n1, n2, ncases, path=path)
    typeI_values.sort()
    return typeI_values[int(typeI_values.shape[0]*(1.-prob))]

def compare_power(n1, n2, ncases, mean, std, path=None):
    """Compare the power of m-test and an independent t-test.

    Compute the probability of a Type II error (not rejecting the null
    hypothesis even though the means are different) for populations of
    size `n1` and `n2`, the first distributed as Normal(mean, std^2) and
    the second as Normal(0, 1).
    
    Parameters
    ----------
    n1 : number of samples in population 1
    n2 : number of samples in population 2
    ncases : number of populations to generate
    mean : mean of population 1
    std : standard deviation of population 1
    path : path to the m-test tables (see `get_tables_path`)

    Returns
    -------
    m_typeII_prob : probability of type II error for the m-test
    t_typeII_prob : probability of type II error for the t-test
    """
    print 'get type I threshold'
    threshold = typeI_threshold(n1, n2, ncases, prob=0.05, path=path)
    print 'threshold at 5%', threshold

    print 'comparing power'
    mt, tt = typeII_table(n1, n2, ncases, mean, std, path=path)
    
    print ('model selection, type II error:',
           1. - (mt>threshold).sum() / float(mt.shape[0]))
    print ('t-test, type II error:',
           1. - (tt<0.05).sum() / float(tt.shape[0]))
    
    return (1. - (mt>threshold).sum() / float(mt.shape[0]),
            1. - (tt<0.05).sum() / float(tt.shape[0]))
