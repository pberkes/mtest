# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import scipy, scipy.io, scipy.stats, pymc
import scipy.integrate as sint
import scipy.stats as ss
import private
private.import_scipy_common(globals())
import os.path
import mdp.utils

# TODO: remove personal dependencies
# TODO: properly handle tables path (e.g., os-independent)
# TODO: compressed npy files instead of .mat

TABLESPATH = os.path.join(os.path.dirname(__file__), 'tables/')
TABLESNAME = 'bayes_ttest_table_n1_%d_n2_%d.mat'
TYPEII_TABLESNAME = 'bayes_ttest_typeII_n1_%d_n2_%d_mdist_%.2f_scale_%.2f.mat'
DISTR_STD = 1.
NPRIOR = 1500

def _set_tables_path(path):
    global TABLESPATH
    TABLESPATH = path

def marg_prior(model, n):
    sm = 0.
    for i in range(n):
        model.draw_from_prior()
        for obs_distr in model.observed_stochastics:
            sm += exp(obs_distr.logp)
    return sm/n
    
# sgm_max: if mean is 3., the std is about 3.14...
def _sameprior_distr(data, sgm_max=3.):
    # mean and std of the mean
    mu_mean = data.mean()
    mu_std = 1/sqrt(len(data))
    # distributions
    muy = pymc.Normal('muy', mu_mean, 1./mu_std**2.)
    stdy = pymc.Uniform('stdy', 0.001, sgm_max)
    return muy, stdy
    
def bayesian_ttest(pop1, pop2, nprior=NPRIOR):
    """
    pop1, pop2 -- samples from the two populations
    nprior -- number of samples from prior to estimate the marginal likelihood
    """
    pops = scipy.concatenate((pop1, pop2))
    # rescale to zero mean, unit variance
    shift, scale = pops.mean(), pops.std()
    pop1 = (pop1-shift)/scale
    pop2 = (pop2-shift)/scale
    pops = scipy.concatenate((pop1, pop2))
    
    # M0
    muy, stdy = _sameprior_distr(pops)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y(value=0., muy=muy, stdy=stdy):
        return pymc.normal_like(pops, muy, 1./(stdy**2.))
    
    model0 = pymc.Model((muy, stdy, y))
    marg_M0 = marg_prior(model0, nprior)

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
    marg_M1 = marg_prior(model1, nprior)

    # M2
    std3y = pymc.Uniform('std3y', 0.001, 1.)
    @pymc.observed
    @pymc.stochastic(dtype=float)
    def y3(value=0., mu1y=mu1y, mu2y=mu2y, std3y=std3y):
        ll = pymc.normal_like(pop1, mu1y, 1./(std3y**2.))
        ll += pymc.normal_like(pop2, mu2y, 1./(std3y**2.))
        return ll
    
    model2 = pymc.Model((mu1y, mu2y, std3y, y3))
    marg_M2 = marg_prior(model2, nprior)
    
    M1M0 = marg_M1/marg_M0
    M2M0 = marg_M2/marg_M0
    
    return max(M1M0, M2M0)

def mtest(pop1, pop2, nprior=NPRIOR, min_ncases=50000):
    data_value = bayesian_ttest(pop1, pop2, nprior)
    test_values = get_table(pop1.shape[0], pop2.shape[0], min_ncases)
    N = len(test_values)
    pval = (test_values>data_value).sum() / float(test_values.shape[0])
    return data_value, pval, N

def _random_same_mean(ncases, n1, n2):
    """Return random samples from two populations with same mean and
    standard deviation."""
    pop_distr = ss.norm(loc=0., scale=DISTR_STD)
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
    pop1_distr = ss.norm(loc=0., scale=DISTR_STD*scale1)
    pop2_distr = ss.norm(loc=mean_dist, scale=DISTR_STD)
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
        test_values = array([])
    
    nvalues = test_values.shape[0]
    if nvalues>=ncases:
        return test_values

    nmissing = ncases-nvalues
    print 'missing %d entries' % nmissing
    
    # compute missing entries
    pop1_test, pop2_test = _random_same_mean(nmissing, n1, n2)

    missing_values = zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        missing_values[i] = bayesian_ttest(pop1_test[i,:], pop2_test[i,:], nprior=NPRIOR)

    # update and save table
    test_values = scipy.concatenate((test_values, missing_values))
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
        bayes_test_values = array([])
        t_test_values = array([])
    
    nvalues = bayes_test_values.shape[0]
    if nvalues>=ncases:
        return bayes_test_values, t_test_values

    nmissing = ncases-nvalues
    print 'missing %d entries' % nmissing
    
    # compute missing entries
    pop1_test, pop2_test = _random_different_mean(nmissing, n1, n2, mean_dist, scale1)

    bayes_missing_values = zeros((nmissing,))
    t_missing_values = zeros((nmissing,))
    for i in mdp.utils.progressinfo(range(nmissing), style='timer'):
        bayes_missing_values[i] = bayesian_ttest(pop1_test[i,:], pop2_test[i,:], nprior=NPRIOR)
        t_missing_values[i] = ss.ttest_ind(pop1_test[i,:], pop2_test[i,:])[1]

    # update and save table
    bayes_test_values = scipy.concatenate((bayes_test_values, bayes_missing_values))
    t_test_values = scipy.concatenate((t_test_values, t_missing_values))
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
