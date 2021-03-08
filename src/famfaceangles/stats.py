"""Module containing statistical methods"""
import numpy as np
from joblib.parallel import Parallel, delayed


def nanmean_axis0(x):
    """Computes nanmean along first axis"""
    return np.nanmean(x, axis=0)


def nantvalue_axis0(x):
    """Computes t-value along first axis discarding nans """
    # code taken from scipy.stats.ttest_1samp
    n = (x.shape[0] - np.isnan(x).sum(axis=0)).astype(float)
    d = np.nanmean(x, axis=0)
    v = np.nanvar(x, axis=0, ddof=1)
    denom = np.sqrt(v / n)
    t = np.divide(d, denom)
    return t


def permutation_1sample(orig_data, perm_data, tail=1, nboots=10000,
                        func=nanmean_axis0, master_rng=None, nproc=1):
    """
    Performs permutation testing given permuted maps to estimate a null
    distribution (see Stelzer et al., 2011). A function can be applied for
    aggregation. Default is nanmean for average. NaNs are allowed.

    Parameters
    ----------
    orig_data : array (n_subjects, n_samples, n_features)
        the original, unpermuted data. The final value will be computed by
        applying func over the first axis (across subjects)
    perm_data : array (n_permutations, n_subjects, n_samples, n_features)
        the permuted data. One of these maps should be the original data.
    tail : int (-1 | 0 | 1)
        tail of the distribution to compute. default is positive, thus
        positive one-tailed test
    nboots : int
        number of bootstrap samples to generate for estimating the null
        distribution
    func : callable
        must handle NaNs; it will be applied along the first axis
    master_rng : RandomState | None
        used to generate random samples
    nproc : int
        number of processors to parallelize with

    Returns
    -------
    t0 : array (n_samples, n_features)
        the statistic value (the results of applying func to orig_data on
        the first axis)
    p : array (n_samples, n_features)
        the feature- and sample-wise empirical p-values
    """
    if tail != 1:
        raise NotImplementedError("Non positive tails are not implemented yet")

    t0 = func(orig_data)
    # initialize counts
    p = np.ones(t0.shape)

    if master_rng is None:
        master_rng = np.random.RandomState()

    # sample maps
    nsubjects, nperms, nsamples, nfeatures = perm_data.shape
    # preallocate so we could parallelize it
    bs_idx = master_rng.choice(np.arange(nperms), size=(nboots, nsubjects))
    bs_idx_blocks = np.array_split(bs_idx, nproc)
    idx_subj = np.arange(nsubjects)
    p = Parallel(n_jobs=nproc)(
        delayed(_bootstrap_once)(bs, func, idx_subj, perm_data, t0)
        for bs in bs_idx_blocks)
    p = np.sum(p, axis=0)
    p += 1
    p /= nboots + 1

    return t0, p


def _bootstrap_once(bs, func, idx_subj, perm_data, t0):
    p = np.zeros(t0.shape)
    for sb in bs:
        p += func(perm_data[idx_subj, sb]) >= t0
    return p
