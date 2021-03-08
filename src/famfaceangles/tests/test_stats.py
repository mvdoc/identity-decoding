import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.stats import ttest_1samp
from ..stats import permutation_1sample, nanmean_axis0, nantvalue_axis0


def test_nantvalue_axis0():
    x = np.random.randn(10, 20)
    assert_array_equal(nantvalue_axis0(x), ttest_1samp(x, 0)[0])

    # impute some nans
    x[0, 0:2] = np.nan
    x[0, 4] = np.nan

    tval = [ttest_1samp(a[np.logical_not(np.isnan(a))], 0)[0] for a in x.T]
    assert_array_almost_equal(nantvalue_axis0(x), tval)


@pytest.mark.parametrize("func", (nanmean_axis0, nantvalue_axis0))
@pytest.mark.parametrize("nproc", (1, 3))
def test_permutation_1sample(func, nproc):
    np.random.seed(32)
    nsubjects, nperms, nsamples, nfeatures = 20, 100, 4, 5
    orig_data = np.random.randn(nsubjects, nsamples, nfeatures)
    perm_data = np.random.randn(nsubjects, nperms, nsamples, nfeatures)
    
    t0, p = permutation_1sample(orig_data, perm_data, nboots=1000, func=func,
                                nproc=nproc)
    assert t0.shape == p.shape
    assert t0.shape == (nsamples, nfeatures)
    assert (p > 0.).all()
    assert (p <= 1.).all()
    assert_array_equal(func(orig_data), t0)

    # now let's do a bit of semantic test
    nsubjects, nperms, nsamples, nfeatures = 20, 100, 1, 5
    orig_data = np.random.randn(nsubjects, nsamples, nfeatures)
    perm_data = np.random.randn(nsubjects, nperms, nsamples, nfeatures)
    orig_data[:, 0, 0] += 1
    orig_data[:, 0, 4] += 1
    t0, p = permutation_1sample(orig_data, perm_data, nboots=1000, func=func,
                                nproc=nproc)

    truth = np.zeros((nsamples, nfeatures), dtype=bool)
    truth[0, 0] = 1
    truth[0, 4] = 1
    assert_array_equal(truth, p < 0.05)
