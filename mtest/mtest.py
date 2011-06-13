# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import scipy as sp
import scipy.io
from scipy import stats
import pymc
import os.path
import mdp.utils

# TODO: remove progressbar dependency (mdp.utils)
# TODO: properly handle tables path (e.g., os-independent)
# TODO: compressed npy files instead of .mat
# TODO: documentation
# TODO: refactor, change names
# TODO: profile
# TODO: remove print statements, replace with logging

TABLESPATH = os.path.join(os.path.dirname(__file__), 'tables/')
TABLESNAME = 'bayes_ttest_table_n1_%d_n2_%d.mat'
TYPEII_TABLESNAME = 'bayes_ttest_typeII_n1_%d_n2_%d_mdist_%.2f_scale_%.2f.mat'
DISTR_STD = 1.
NPRIOR = 1500

def _set_tables_path(path):
    global TABLESPATH
    TABLESPATH = path

def _marginalize_prior(model, n):
    """Computes the marginal likelihood by integrating over the prior.

    Input:
    ----------------

    model -- PyMC model
    n -- number of samples from the prior used to compute the integral

    Output:
    -------

    Returns an estimate of the marginal likelihood
    """
    sm = 0.
    for i in range(n):
        model.draw_from_prior()
        for obs_distr in model.observed_stochastics:
            sm += sp.exp(obs_distr.logp)
    return sm/n
    
# sgm_max: if mean is 3., the std is about 3.14...
def _sameprior_distr(data, sgm_max=3.):
    # mean and std of the mean
    mu_mean = data.mean()
    mu_std = 1/sp.sqrt(len(data))
    # distributions
    muy = pymc.Normal('muy', mu_mean, 1./mu_std**2.)
    stdy = pymc.Uniform('stdy', 0.001, sgm_max)
    return muy, stdy
    
def mtest_marginal_likelihood_ratio(pop1, pop2, nprior=NPRIOR):
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

    Input:
    ------
    
    pop1, pop2 -- samples from the two populations
    
    nprior -- number of samples from the prior over parameters, used
              to estimate the marginal likelihood integral

    Output:
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
    muy, stdy = _sameprior_distr(pops)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y(value=0., muy=muy, stdy=stdy):
        return pymc.normal_like(pops, muy, 1./(stdy**2.))
    
    model0 = pymc.Model((muy, stdy, y))
    marg_M0 = _marginalize_prior(model0, nprior)

    # M1
    mu1y, std1y = _sameprior_distr(pop1, sgm_max=1.)
    mu2y, std2y = _sameprior_distr(pop2, sgm_max=1.)
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

def mtest(pop1, pop2, nprior=NPRIOR, min_ncases=50000):
    data_value = mtest_marginal_likelihood_ratio(pop1, pop2, nprior)
    test_values = get_table(pop1.shape[0], pop2.shape[0], min_ncases)
    N = len(test_values)
    pval = (test_values>data_value).sum() / float(test_values.shape[0])
    return data_value, pval, N

def _random_same_mean(ncases, n1, n2):
    """Return random samples from two populations with same mean and
    standard deviation."""
    pop_distr = stats.norm(loc=0., scale=DISTR_STD)
    pop1 = pop_distr.rvs(size=(ncases, n1))
    pop2 = pop_distr.rvs(size=(ncases, n2))
    return pop1, pop2

def _random_different_mean(ncases, n1, n2, mean_dist, scale1):
    """Return random samples from two populations with same 
    standard deviation and different mean

    mean_dist -- separation in mean, as fraction of the standard deviation
    scale1 -- the standard deviation of population 1 is scale1 times the
              std of population 2
    """
    mean_dist = mean_dist*DISTR_STD
    pop1_distr = stats.norm(loc=0., scale=DISTR_STD*scale1)
    pop2_distr = stats.norm(loc=mean_dist, scale=DISTR_STD)
    pop1 = pop1_distr.rvs(size=(ncases, n1))
    pop2 = pop2_distr.rvs(size=(ncases, n2))
    return pop1, pop2

def get_table(n1, n2, ncases):
    """
    Returns a table of values of the bayesian_ttest statistics under the
    null hypothesis, for two populations of size n1 and n2.

    If a table for n1 and n2 with more entries than ncases is already saved, use it.
    Otherwise, add more entries to the table, save it, then return it.
    """

    print 'N1 = %d, N2 = %d' % (n1, n2)

    fname = TABLESPATH+TABLESNAME%(n1,n2)
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
    pop1_test, pop2_test = _random_same_mean(nmissing, n1, n2)

    missing_values = sp.zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        missing_values[i] = mtest_marginal_likelihood_ratio(pop1_test[i,:],
                                                            pop2_test[i,:],
                                                            nprior=NPRIOR)

    # update and save table
    test_values = sp.concatenate((test_values, missing_values))
    print 'saving', fname
    scipy.io.savemat(fname, {'test_values': test_values})

    return test_values

def get_typeII_table(n1, n2, ncases, mean_dist, scale1):
    """
    Returns a table of values of the bayesian_ttest and ttest statistics
    for two populations of size n1 and n2 and mean difference
    'mean_dist' times their standard deviation.

    If a table for n1 and n2 with more entries than ncases is already saved, use it.
    Otherwise, add more entries to the table, save it, then return it.
    """

    print 'N1 = %d, N2 = %d' % (n1, n2)
    print 'mean_dist = %f, scale factor for pop1 = %f' % (mean_dist, scale1)

    fname = TABLESPATH+TYPEII_TABLESNAME%(n1,n2,mean_dist,scale1)
    if os.path.exists(fname):
        print 'loading', fname
        matdict = scipy.io.loadmat(fname)
        bayes_test_values = matdict['bayes_test_values'].flatten()
        t_test_values = matdict['t_test_values'].flatten()
    else:
        bayes_test_values = sp.array([])
        t_test_values = sp.array([])
    
    nvalues = bayes_test_values.shape[0]
    if nvalues>=ncases:
        return bayes_test_values, t_test_values

    nmissing = ncases-nvalues
    print 'missing %d entries' % nmissing
    
    # compute missing entries
    pop1_test, pop2_test = _random_different_mean(nmissing, n1, n2, mean_dist, scale1)

    bayes_missing_values = sp.zeros((nmissing,))
    t_missing_values = sp.zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        bayes_missing_values[i] = mtest_marginal_likelihood_ratio(pop1_test[i,:],
                                                                  pop2_test[i,:],
                                                                  nprior=NPRIOR)
        t_missing_values[i] = stats.ttest_ind(pop1_test[i,:], pop2_test[i,:])[1]

    # update and save table
    bayes_test_values = sp.concatenate((bayes_test_values, bayes_missing_values))
    t_test_values = sp.concatenate((t_test_values, t_missing_values))
    print 'saving', fname
    scipy.io.savemat(fname, {'bayes_test_values': bayes_test_values,
                             't_test_values': t_test_values})

    return bayes_test_values, t_test_values

def typeI_threshold(n1, n2, ncases, pval=0.05):
    typeI_values = get_table(n1, n2, ncases)
    typeI_values.sort()
    return typeI_values[int(typeI_values.shape[0]*(1.-pval))]

def compare_power(n1, n2, ncases, mean_dist, scale1):
    print 'get type I threshold'
    threshold = typeI_threshold(n1, n2, ncases, pval=0.05)
    print 'threshold at 5%', threshold

    print 'comparing power'
    bt, tt = get_typeII_table(n1, n2, ncases, mean_dist, scale1)
    
    print 'model selection, type II error:', 1. - (bt>threshold).sum() / float(bt.shape[0])
    print 't-test, type II error:', 1. - (tt<0.05).sum() / float(tt.shape[0])
    
    return 1. - (bt>threshold).sum() / float(bt.shape[0]), 1. - (tt<0.05).sum() / float(tt.shape[0])
