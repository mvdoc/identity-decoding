"""Modules containing tests for mvpa module"""
import numpy as np
import pytest
from scipy import stats, linalg, spatial
from sklearn.preprocessing import normalize as sknorm
from numpy.testing import assert_equal, assert_almost_equal
from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import OddEvenPartitioner, NFoldPartitioner
from mvpa2.measures.base import CrossValidation
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.measures.rsa import CDist
from ..mvpa import (pcorr, corr, pearsonr_no_pval_vec, spearmanr_no_pval_vec,
                    PCorrTargetSimilarity, MeanCrossValidation,
                    FisherCDist, sort_samples, SymmetricSplitHalfPartitioner,
                    CDistPcorrTargetSimilarity, CORRFX)


def test_pearsonr_nopval_vec():
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 1)
    x = X.copy()
    y = Y.copy()
    r = pearsonr_no_pval_vec(X, Y)
    # check no operation occurred in place
    assert_equal(X, x)
    assert_equal(Y, y)
    # check output
    assert_equal(X.shape[1], r.shape[0])
    for i in range(X.shape[1]):
        assert_almost_equal(stats.pearsonr(X[:, i], Y.ravel())[0], r[i])


def test_spearmanr_nopval_vec():
    X = np.random.randn(10, 5)
    Y = np.random.randn(10, 1)
    r = spearmanr_no_pval_vec(X, Y)
    assert_equal(X.shape[1], r.shape[0])
    for i in range(X.shape[1]):
        assert_almost_equal(
            stats.pearsonr(
                stats.rankdata(X[:, i]), stats.rankdata(Y.ravel()))[0], r[i])


@pytest.mark.parametrize("normalize", [True, False])
def test_pcorr(normalize):
    X = np.random.randn(10, 3)
    Y = np.random.randn(10, 5)
    rp = pcorr(Y, X, normalize=normalize)
    assert_equal(rp.shape, (X.shape[1], Y.shape[1]))

    # test it explodes with one element
    with pytest.raises(ValueError):
        rp = pcorr(Y, X[:, 0][:, None])
      
    # generate some data and check it works
    Y = np.random.randn(10, 4)*3 + 10
    X = np.random.randn(10, 2)*2 + 4
    if normalize:
        Y -= Y.mean(axis=0)
        X -= X.mean(axis=0)
        Y = sknorm(Y, axis=0)
        X = sknorm(X, axis=0)
        
    pred1 = X[:, 0][:, None]
    pred2 = X[:, 1][:, None]
    
    out = pcorr(Y, X, corrfx=pearsonr_no_pval_vec, normalize=normalize)
    for i_p, (a, z) in enumerate([[pred1, pred2], [pred2, pred1]]):
        beta_a = linalg.lstsq(z, a)[0]
        res_a = a - z.dot(beta_a)
        beta_y = linalg.lstsq(z, Y)[0]
        res_y = Y - z.dot(beta_y)
        corr_res = [stats.pearsonr(y, res_a.ravel())[0] for y in res_y.T]
        assert_almost_equal(corr_res, out[i_p])


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("corrtype", ["spearman", "pearson"])
@pytest.mark.parametrize("ntargetrdms", [3, 1])
def test_pcorrtargetsimilarity(normalize, corrtype, ntargetrdms):
    nsamples = 4
    nfeatures = 10
    npairwisesamples = (nsamples * (nsamples - 1))/2
    ds = Dataset(np.random.randn(nsamples, nfeatures))
    target_rdms = np.random.randn(npairwisesamples, ntargetrdms)
    
    meas = PCorrTargetSimilarity(
        target_rdms=target_rdms,
        correlation_type=corrtype,
        normalize_rdms=normalize
    )
    out = meas(ds)
    assert out.nsamples == ntargetrdms
    assert out.nfeatures == 1


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("corrtype", ["spearman", "pearson"])
@pytest.mark.parametrize("ntargetrdms", [3, 1])
def test_fishercdistpcorrtargetsimilarity(normalize, corrtype, ntargetrdms):
    nsamples = 4
    nfeatures = 10
    npairwisesamples = int((nsamples * (nsamples - 1))/2)
    ds_train = Dataset(np.random.randn(nsamples, nfeatures))
    ds_test = Dataset(np.random.randn(nsamples, nfeatures))
    target_rdms = np.random.randn(npairwisesamples, ntargetrdms)

    meas = CDistPcorrTargetSimilarity(
        target_rdms=target_rdms,
        correlation_type=corrtype,
        normalize_rdms=normalize
    )
    ds_train.sa['targets'] = np.arange(nsamples)
    ds_test.sa['targets'] = np.arange(nsamples)
    meas.train(ds_train)
    out = meas(ds_test)
    assert out.nsamples == ntargetrdms
    assert out.nfeatures == 1
    assert out.samples.ndim == 2
    cout = spatial.distance.cdist(ds_train.samples, ds_test.samples,
                                  metric='correlation')
    cout = np.nan_to_num(np.arctanh(1 - cout))
    cout = (cout + cout.T)/2.
    cout = cout[np.triu_indices_from(cout, k=1)][:, None]
    cout = 1. - np.tanh(cout)
    fx = pcorr if ntargetrdms > 1 else corr
    pr = fx(cout, target_rdms, corrfx=CORRFX[corrtype], normalize=normalize)
    assert_equal(out.samples, np.atleast_2d(pr))


def test_mean_crossvalidation():
    nsamples = 4
    nchunks = 10
    nfeatures = 10
    ds = Dataset(np.random.randn(nsamples*nchunks, nfeatures))
    labels = list('abcd') * nchunks
    chunks = np.repeat(np.arange(4), nchunks)
    ds.sa['chunks'] = chunks
    ds.sa['labels'] = labels

    partitioner = OddEvenPartitioner()
    meas = CDist(sattr=['labels'])

    cv = CrossValidation(meas, partitioner, errorfx=None)
    mcv = MeanCrossValidation(meas, partitioner, sample_attr='labels',
                              errorfx=None)

    out_cv = cv(ds)
    out_mcv = mcv(ds)

    assert_equal(out_mcv.sa.labels,
                 [':'.join(c) for c in out_cv[:out_mcv.nsamples].sa.labels])
    # need to fix the labels for cdist
    out_cv.sa['labels'] = [':'.join(c) for c in out_cv.sa.labels]
    mean_out_cv = mean_group_sample(['labels'])(out_cv)
    assert out_cv.nfeatures == out_mcv.nfeatures
    assert out_cv.nsamples == out_mcv.nsamples * 2
    assert_equal(mean_out_cv.samples, out_mcv.samples)


def test_fishercdist():
    with pytest.raises(ValueError):
        FisherCDist(pairwise_metric='euclidean')

    fcdist = FisherCDist()
    cdist = CDist()

    ds = Dataset(np.random.randn(20, 10))
    ds.sa['targets'] = np.tile(np.arange(10), 2)
    fcdist.train(ds[:10])
    cdist.train(ds[:10])
    fout = fcdist(ds[10:])
    cout = cdist(ds[10:])

    cout.samples = np.arctanh(1. - cout.samples)
    cout.samples = np.nan_to_num(cout.samples)
    assert_equal(cout.samples, fout.samples)


ds = Dataset(np.random.randn(20, 10))
ds.sa['targets'] = np.tile(np.arange(10), 2)
ds.sa['labels'] = ds.sa.targets
ds.sa['chunks'] = np.repeat(np.arange(2), 10)


def test_sort_sample():
    randperm = np.random.permutation(np.arange(ds.nsamples))
    ds_perm = ds.copy()[randperm]
    ds_perm = sort_samples(ds_perm)

    assert_equal(ds.sa.chunks, ds_perm.sa.chunks)
    assert_equal(ds.sa.targets, ds_perm.sa.targets)
    assert_equal(ds.samples, ds_perm.samples)


def test_symmetricsplithalfpartitioner():
    # check that it works with arbitrary attr as well
    p1 = NFoldPartitioner(cvtype=0.5, attr='labels')
    p2 = SymmetricSplitHalfPartitioner(attr='labels')

    partitions_1 = p1.get_partition_specs(ds)
    partitions_2 = p2.get_partition_specs(ds)

    assert len(partitions_1) == len(partitions_2) * 2
    # reconstruct full partitions
    uc = set(np.unique(ds.sa.labels))
    extra_partitions_2 = [
        (None, list(uc.difference(s))) for _, s in partitions_2
    ]
    assert_equal(
        sorted(partitions_1), sorted(partitions_2 + extra_partitions_2))
