"""Module containing various functions and classes for MVPA"""
import numpy as np
from mvpa2.mappers.fx import argsort, mean_group_sample
from mvpa2.measures.base import Measure, CrossValidation
from mvpa2.measures.rsa import CDist
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import EnsureChoice
from mvpa2.datasets.base import Dataset
from scipy import stats, linalg
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sknorm


def pearsonr_no_pval(x, y):
    """Returns pearsons correlation without pvalues"""
    return stats.pearsonr(x, y)[0]


def spearmanr_no_pval(x, y):
    """Returns spearman correlation without pvalues"""
    return stats.pearsonr(stats.rankdata(x), stats.rankdata(y))[0]


def pearsonr_no_pval_vec(X, Y):
    """Returns pearsons correlation, vectorized version

    Parameters
    ----------
    X : ndarray (n_samples, n_observations)
    Y : ndarray (n_samples, 1)

    Returns
    -------
    R : array (n_observations,)
        pearson correlation between each column of X and Y
    """
    x_ = X - X.mean(axis=0)
    y_ = Y - Y.mean(axis=0)
    x_ /= np.sqrt(np.sum(x_**2, axis=0))
    y_ /= np.sqrt(np.sum(y_**2, axis=0))

    return (x_*y_).sum(axis=0)


def spearmanr_no_pval_vec(X, Y):
    """Returns spearmans correlation, vectorized version

    Parameters
    ----------
    X : ndarray (n_samples, n_observations)
    Y : ndarray (n_samples, 1)

    Returns
    -------
    R : array (n_observations,)
        spearmans correlation between each column of X and Y
    """
    x_ = np.apply_along_axis(stats.rankdata, axis=0, arr=X)
    y_ = np.apply_along_axis(stats.rankdata, axis=0, arr=Y)

    return pearsonr_no_pval_vec(x_, y_)


def pcorr(Y, X, corrfx=spearmanr_no_pval_vec, normalize=True):
    """
    Compute partial correlation between each column of Y and columns of X.

    Parameters
    ----------
    Y : np.array (n_samples, n_observations)
        target vector (e.g., neural dissimilarity matrix)
    X : np.array (n_samples, p)
        vectors that will be correlated with y, partialing out the effect
        of the other columns (e.g., behavioral RDMs)
    corrfx : function (x, y)
        function to apply on the residuals to compute the correlation;
        default is spearman r, but pearson r could be used as well.
    normalize : boolean
        whether to normalize (demean and unit norm) both y and X. Otherwise
        you'll have to input an additional columns of ones to account for
        differences in means.

    Returns
    -------
    rp : np.array (p, n_observations)
        pairwise partial correlations between y and the columns of X
        rp[i] is the correlation between Y and X[:, i] after partialling out
        X[:, j], j != i
    """
    # This function was inspired by the code by Fabian Pedregosa, available
    # here: https://gist.github.com/fabianp/9396204419c7b638d38f

    X = np.asarray(X)
    Y = np.asarray(Y)
    n_observations = Y.shape[1]
    n_corr = X.shape[1]
    if n_corr < 2:
        raise ValueError("Need more than one column in X to run partial corr")
    if normalize:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        X = sknorm(X, axis=0)
        Y = sknorm(Y, axis=0)

    rp = np.zeros((n_corr, n_observations), dtype=np.float)
    for i in range(n_corr):
        idx = np.ones(n_corr, dtype=np.bool)
        idx[i] = False
        beta_y = linalg.lstsq(X[:, idx], Y)[0]
        beta_i = linalg.lstsq(X[:, idx], X[:, i])[0]

        res_y = Y - X[:, idx].dot(beta_y)
        res_i = X[:, i] - X[:, idx].dot(beta_i)

        rp[i] = corrfx(res_y, res_i[:, None])
    return rp


def corr(Y, X, corrfx=spearmanr_no_pval_vec, normalize=None):
    """
    Compute correlation between each column of Y and X.

    Parameters
    ----------
    Y : np.array (n_samples, n_observations)
        target vector (e.g., neural dissimilarity matrix)
    X : np.array (n_samples, 1)
        vector that will be correlated with y
    corrfx : function (x, y)
        function to compute the correlation;
        default is spearman r, but pearson r could be used as well.
    normalize : none

    Returns
    -------
    r : np.array (1, n_observations)
        correlation between columns of Y and X
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    return corrfx(X, Y)


CORRFX = {
    'pearson': pearsonr_no_pval_vec,
    'spearman': spearmanr_no_pval_vec
}


class PCorrTargetSimilarity(Measure):
    """Calculate the partial correlations of a neural RDM with more than two
    target RDMs. This measure can be used for example when comparing a
    neural RDM with more than one behavioral RDM, and one desires to look
    only at the correlations of the residuals.
    
    If only one target RDM is passed, then it will compute correlation 
    instead of partial correlation.

    NOTA BENE: this measure computes a distance internally through
    scipy.spatial.pdist, thus you should make sure that the
    predictors are in the correct direction, that is smaller values imply
    higher similarity!
    """

    is_trained = True
    """Indicate that this measure is always trained."""

    pairwise_metric = Parameter('correlation', constraints='str', doc="""\
          Distance metric to use for calculating pairwise vector distances for
          the neural dissimilarity matrix (DSM).  See 
          scipy.spatial.distance.pdist for all possible metrics.""")

    correlation_type = Parameter('spearman', constraints=EnsureChoice(
        'spearman', 'pearson'), doc="""\
          Type of correlation to use between the compute neural RDM and the 
          target RDMs. If spearman, the residuals are ranked 
          prior to the correlation.""")

    normalize_rdms = Parameter(True, constraints='bool', doc="""\
          If True then center and normalize each column of the neural RDM 
          and of the predictor RDMs by subtracting the
          column mean from each element and imposing unit L2 norm.""")

    def __init__(self, target_rdms, **kwargs):
        """
        Parameters
        ----------
        target_rdms : array (length N*(N-1)/2, n_predictors)
          Target dissimilarity matrices
        """
        # init base classes first
        super(PCorrTargetSimilarity, self).__init__(**kwargs)
        self.target_rdms = target_rdms
        self.corrfx = CORRFX[self.params.correlation_type]
        self.normalize_rdms = self.params.normalize_rdms

    def _call(self, dataset):
        """
        Parameters
        ----------
        dataset : input dataset

        Returns
        -------
        Dataset
        each sample `i` correspond to the partial correlation between the
        neural RDM and the `target_dsms[:, i]` partialling out
        `target_dsms[:, j]` with `j != i`.
        """
        data = dataset.samples
        dsm = pdist(data, self.params.pairwise_metric)
        fx = pcorr if self.target_rdms.shape[1] > 1 else corr
        rp = fx(dsm[:, None],
                self.target_rdms,
                corrfx=self.corrfx,
                normalize=self.normalize_rdms)
        return Dataset(rp[:, None])


class MeanCrossValidation(CrossValidation):
    """Averages results of CrossValidation without returning the individual
    folds. This is used with e.g. cross-validated RSA to avoid clogging up
    too much memory by returning all the folds."""

    def __init__(self, learner, generator=None,
                 sample_attr='targets', *args, **kwargs):
        """Use `sample_attr` to specify the grouping labels: all samples
        within each unique value of `sample_attr` will be averaged"""
        self.sample_attr = sample_attr
        super(MeanCrossValidation, self).__init__(learner, generator,
                                                  *args, **kwargs)

    def _call(self, ds):
        sattr = self.sample_attr
        res = super(MeanCrossValidation, self)._call(ds)
        # XXX: ad hoc fix for CDist because it returns tuples
        if isinstance(self.learner, CDist):
            res.sa[sattr] = [':'.join(c) for c in res.sa[sattr]]
        mgs = mean_group_sample(attrs=[sattr])
        return mgs(res)


class FisherCDist(CDist):
    """Converts correlation distance to correlation and apply fisher
    transform"""
    def __init__(self, **kwargs):
        if 'pairwise_metric' in kwargs:
            raise ValueError("pairwise_metric argument is disabled, "
                             "only correlation is allowed.")
        super(FisherCDist, self).__init__(pairwise_metric='correlation',
                                          **kwargs)

    def _call(self, ds):
        ds = super(FisherCDist, self)._call(ds)
        ds.samples = np.arctanh(1. - ds.samples)
        ds.samples = np.nan_to_num(ds.samples)
        return ds


def make_symmetric(ds, correlation=True, diagonal=False):
    """
    Given a dataset containing an RDM for each feature, make it symmetric.

    Parameters
    ----------
    ds : Dataset (n_samples, n_features)
       dataset containing the RDM
    correlation : bool
        whether the dataset contains correlation distances. If True, then the
        distances will be converted back to correlation, Fisher-transformed
        prior to averaging, and converted back afterwards.
    diagonal : bool
        whether to keep the diagonal

    Returns
    -------
    ds : Dataset (n_samples*(n_samples-1)/2 + d, n_features)
         with d = n_samples if diagonal is True, otherwise d = 0

        the upper triangular portion of the matrix
    """
    # Convert back to Fisher-transformed z-values
    if correlation:
        ds.samples = np.arctanh(1. - ds.samples)
        ds.samples = np.nan_to_num(ds.samples)
    nsamples = ds.nsamples
    nstim = int(np.sqrt(nsamples))

    # average with the transpose
    idx = np.arange(nsamples).reshape((nstim, nstim))
    ds.samples += ds.samples[idx.T.flatten()]
    ds.samples /= 2.

    if correlation:
        ds.samples = 1. - np.tanh(ds.samples)

    # take only triu, discard diagonal if needed
    k = 0 if diagonal else 1
    triu = idx[np.triu_indices_from(idx, k=k)]
    ds = ds[triu]
    return ds


class CDistPcorrTargetSimilarity(CDist):
    """Computes fisher-transformed correlation distance, then performs
    correlation or partial correlation against target neural RDM, 
    depending on the number of predictors"""

    correlation_type = Parameter('spearman', constraints=EnsureChoice(
        'spearman', 'pearson'), doc="""\
          Type of correlation to use between the compute neural RDM and the 
          target RDMs. If spearman, the residuals are ranked 
          prior to the correlation.""")

    normalize_rdms = Parameter(True, constraints='bool', doc="""\
          If True then center and normalize each column of the neural RDM 
          and of the predictor RDMs by subtracting the
          column mean from each element and imposing unit L2 norm.""")

    def __init__(self, target_rdms, **kwargs):
        super(CDistPcorrTargetSimilarity, self).__init__(
            pairwise_metric='correlation', **kwargs)
        self.target_rdms = target_rdms
        self.corrfx = CORRFX[self.params.correlation_type]
        self.normalize_rdms = self.params.normalize_rdms

    def _call(self, ds):
        # first compute cdist
        data = super(CDistPcorrTargetSimilarity, self)._call(ds)
        # make it correlation and not distance
        # make symmetric and take upper diagonal
        data = make_symmetric(data)
        fx = pcorr if self.target_rdms.shape[1] > 1 else corr
        rp = fx(data.samples,
                self.target_rdms,
                corrfx=self.corrfx,
                normalize=self.normalize_rdms)
        return Dataset(rp)


def sort_samples(ds, chunks_attr='chunks', targets_attr='targets'):
    """
    Reorder the samples according to `chunks_attr` first, then `targets_attr`

    Parameters
    ----------
    ds
    chunks_attr
    targets_attr

    Returns
    -------
    ds

    """
    idx_reordered = argsort(zip(ds.sa[chunks_attr], ds.sa[targets_attr]))
    return ds[idx_reordered]


class SymmetricSplitHalfPartitioner(NFoldPartitioner):
    """Like `NFoldPartitioner(cvtype=0.5)`, but returns only half of the
    partitions to reduce the number of computations (for example, if using
    RSA, the metric is symmetric -- unless we're using Crossnobis)"""

    def __init__(self, **kwargs):
        super(SymmetricSplitHalfPartitioner, self).__init__(
            cvtype=0.5, **kwargs)

    def _get_partition_specs(self, uniqueattrs):
        specs = \
            super(SymmetricSplitHalfPartitioner, self). _get_partition_specs(
                uniqueattrs)
        uc = set(uniqueattrs)
        specs_ = set()
        for _, s in specs:
            partitions = (tuple(uc.difference(s)), tuple(s))
            partitions = tuple(sorted(partitions))
            specs_.add(partitions)
        specs_ = list(specs_)
        return [(None, s) for _, s in specs_]
