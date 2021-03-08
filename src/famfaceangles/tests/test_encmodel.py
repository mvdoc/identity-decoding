"""Tests for encoding models module"""
import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
                           assert_array_almost_equal)
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from ..encmodel import (global_motion, corr, ridge_optim, fit_encmodel,
                        fit_encmodel_bwsj, add_delays, reduce_spatial_features,
                        _ridge_search)


def test_global_motion():
    nframes = 20
    video = np.random.randn(nframes, 20, 20)
    motion = global_motion(video)
    assert isinstance(motion, np.ndarray)
    assert motion.dtype == np.float
    assert motion.ndim == 1
    assert len(motion) == nframes

    motion = global_motion(video, max_frames=10)
    assert isinstance(motion, np.ndarray)
    assert motion.ndim == 1
    assert len(motion) == 10

    prev = np.random.randn(20, 20)
    motion = global_motion(video, prev_frame=prev)
    v = video[0].copy()
    v -= v.mean()
    prev -= prev.mean()
    assert_array_equal(motion[0], np.sqrt(np.mean((v-prev)**2)))

    # check it works also if we ask for more frames
    motion = global_motion(video, max_frames=30)
    assert isinstance(motion, np.ndarray)
    assert motion.ndim == 1
    assert len(motion) == 30

    # because it's motion, should always be greater than 0
    assert np.all(motion >= 0)


def test_corr():
    X = np.random.randn(10, 4)
    Y = np.random.randn(10, 4)
    R = corr(X, Y)
    n = X.shape[0]
    for i in range(n):
        assert np.allclose(R[i], np.corrcoef(X[i], Y[i])[0, 1])


@pytest.mark.parametrize("method", ["optimridge", "sklearn"])
@pytest.mark.parametrize("njobs", [1, 2])
def test_ridge_optim(njobs, method):
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)

    logo = LeaveOneGroupOut()
    cv = list(logo.split(X, groups=groups))
    alphas = np.logspace(1, 3, 10)
    n_alphas = len(alphas)
    out = ridge_optim(X, Y, groups, cv, alphas, method=method, njobs=njobs)

    assert len(out) == 2
    best_alpha, score_curve = out
    assert_array_equal(best_alpha.shape, (n_runs,))
    assert_array_equal(score_curve.shape, (n_runs, n_alphas))


@pytest.mark.parametrize("method", ["optimridge", "sklearn"])
@pytest.mark.parametrize("njobs", [1, 2])
def test_fit_encmodel(njobs, method):
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(1, 3, 10)
    n_alphas = len(alphas)
    score, weights, best_alpha, score_curve = \
        fit_encmodel(X, Y, groups, alphas, njobs=njobs, method=method)

    assert_array_equal(score.shape, (n_runs, n_features))
    assert_array_equal(best_alpha.shape, (n_runs,))
    assert_array_equal(score_curve.shape, (n_runs, n_alphas))
    assert_array_equal(weights.shape, (n_runs, n_features, n_preds))


def test_fit_encmodel_nocoef():
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(1, 3, 10)
    n_alphas = len(alphas)
    score, weights, best_alpha, score_curve = \
        fit_encmodel(X, Y, groups, alphas, compute_coef=False)

    assert_array_equal(score.shape, (n_runs, n_features))
    assert_array_equal(best_alpha.shape, (n_runs,))
    assert_array_equal(score_curve.shape, (n_runs, n_alphas))
    assert weights == []

    # test against when we compute the weights
    score_, weights_, best_alpha_, score_curve_ = \
        fit_encmodel(X, Y, groups, alphas, compute_coef=True)
    assert_array_equal(best_alpha, best_alpha_)
    assert_array_equal(score_curve, score_curve_)
    assert_array_almost_equal(score, score_)
    assert_array_equal(weights_.shape, (n_runs, n_features, n_preds))


def test_fit_encmodel_savetodisk(tmpdir):
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(1, 3, 10)
    n_alphas = len(alphas)

    fn_tmpl = str(tmpdir.join('test_cv{0:02d}_{1}'))

    score, weights, best_alpha, score_curve = \
        fit_encmodel(X, Y, groups, alphas, save_results_todisk=fn_tmpl)

    assert_array_equal(best_alpha.shape, (n_runs,))
    assert_array_equal(score_curve.shape, (n_runs, n_alphas))

    assert len(score) == len(weights)
    assert np.all([isinstance(s, str) for s in score])
    assert np.all([isinstance(w, str) for w in weights])
    scores = np.stack([np.load(s)['scores'] for s in score])
    weights = np.stack([np.load(w)['coef'] for w in weights])
    assert_array_equal(scores.shape, (n_runs, n_features))
    assert_array_equal(weights.shape, (n_runs, n_features, n_preds))
    assert scores.dtype == np.float32
    assert weights.dtype == np.float32


@pytest.mark.parametrize("compute_coef", [True, False])
def test_fit_encmodel_bwsbj(compute_coef):
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4
    n_sbjs = 10

    X = np.random.randn(n_samples, n_preds)
    Y = [np.random.randn(n_samples, n_features) for _ in range(n_sbjs)]
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(1, 3, 6)
    n_alphas = len(alphas)
    score, weights, best_alpha, score_curve = \
        fit_encmodel_bwsj(X, Y, groups, alphas, compute_coef=compute_coef)

    assert_array_equal(score.shape, (n_runs*n_sbjs, n_features))
    assert_array_equal(best_alpha.shape, (n_runs*n_sbjs,))
    assert_array_equal(score_curve.shape, (n_runs*n_sbjs, n_alphas))
    if compute_coef:
        assert_array_equal(weights.shape, (n_runs*n_sbjs, n_features, n_preds))
    else:
        assert weights == []


@pytest.mark.parametrize("compute_coef", [True, False])
def test_fit_encmodel_bwsbj_against_encmodel(compute_coef):
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4
    n_sbjs = 10

    X = np.random.randn(n_samples, n_preds)
    # get same Ys for each subject
    Y = [np.random.randn(n_samples, n_features)] * n_sbjs
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(1, 3, 6)
    n_alphas = len(alphas)

    # technically these two should be the same up to rounding error for
    # averaging
    score_bw, weights_bw, best_alpha_bw, score_curve_bw = \
        fit_encmodel_bwsj(X, Y, groups, alphas, compute_coef=compute_coef)
    score, weights, best_alpha, score_curve = \
        fit_encmodel(X, Y[0], groups, alphas, compute_coef=compute_coef)

    # need to average each *_bw; indices of outer folds are slower than
    # subjects, that is the results are n_sbjs * n_runs with
    # Test Run, Test Sbj
    # 0 0
    # 0 1
    # 0 2
    # ...
    assert_array_almost_equal(
        score, score_bw.reshape((n_runs, n_sbjs, n_features)).mean(axis=1))
    assert_array_almost_equal(
        best_alpha, best_alpha_bw.reshape((n_runs, n_sbjs)).mean(axis=1))
    assert_array_almost_equal(
        score_curve,
        score_curve_bw.reshape((n_runs, n_sbjs, n_alphas)).mean(axis=1))
    if compute_coef:
        assert_array_almost_equal(
            weights,
            weights_bw.reshape((n_runs, n_sbjs, n_features, n_preds)).mean(
                axis=1))
    else:
        assert weights == []
        assert weights_bw == []


def test_compare_encmodel_methods():
    n_samples = 100
    n_features = 20
    n_runs = 5
    n_preds = 4

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    X = (X - X.mean(0))/X.std(0)
    Y = (Y - Y.mean(0))/Y.std(0)
    groups = np.repeat(np.arange(n_runs), n_samples / n_runs)
    alphas = np.logspace(0, 3, 20)
    score_sk, weights_sk, best_alpha_sk, score_curve_sk = \
        fit_encmodel(X, Y, groups, alphas, method='sklearn')
    score_or, weights_or, best_alpha_or, score_curve_or = \
        fit_encmodel(X, Y, groups, alphas, method='optimridge')

    assert_array_almost_equal(score_sk, score_or)
    assert_array_almost_equal(best_alpha_sk, best_alpha_or)
    assert_array_almost_equal(score_curve_sk, score_curve_or)
    assert_array_almost_equal(weights_sk, weights_or)


def test_ridge_search_methods():
    n_samples = 100
    n_features = 20
    n_preds = 5

    X = np.random.randn(n_samples, n_preds)
    Y = np.random.randn(n_samples, n_features)
    alphas = np.logspace(0, 3, 20)
    train_idx = np.arange(n_samples)
    scores_or = _ridge_search(X, Y, train_idx, train_idx, alphas=alphas,
                              method='optimridge')
    scores_sk = _ridge_search(X, Y, train_idx, train_idx, alphas=alphas,
                              method='sklearn')
    # check that correlation of scores is almost the same
    # there are probably some differences because of matrix factorization
    R = corr(scores_or, scores_sk)
    assert_array_almost_equal(R, [1.] * len(R), decimal=2)


def test_add_delays():
    s, p = 10, 4
    X = np.random.randn(s, p)
    delays = [2, 4, 6]
    ndelays = len(delays)
    TR = 1.0
    Xd = add_delays(X, delays, TR)

    assert_array_equal(Xd.shape, (X.shape[0], X.shape[1] * (ndelays + 1)))
    for start, d in enumerate([0] + delays):
        col_start = start * p
        col_end = col_start + p
        assert_array_equal(X[:s-d], Xd[d:, col_start:col_end])
    assert_array_equal(X, Xd[:, :X.shape[1]])


def test_reduce_spatial_features():
    h, w = 200, 200
    X = np.random.randn(h, w)
    # first try it out with a large number of maxfeatures
    X_ = reduce_spatial_features(X, h*w*10)
    assert_equal(X_.ndim, 4)
    assert_array_equal(X_[0, ..., 0], X)
    # now reduce it by a factor of 4 in each one
    factor = 4
    X_ = reduce_spatial_features(X, h*w/factor**2)
    _, nh, nw, _ = X_.shape
    assert_equal(nh, h/factor)
    assert_equal(nw, w/factor)


@pytest.mark.parametrize("factor", np.arange(1, 9, dtype=np.float))
@pytest.mark.parametrize("hw", 
                         [(200, 320), (201, 321), (205, 325), (200, 329)])
def test_reduce_spatial_features_sweep(hw, factor):
    h, w = hw
    # check what happens with different sizes
    nimgs, chans = 10, 3
    X = np.random.randn(nimgs, h, w, chans)
    maxfeatures = np.ceil(h * w * chans / factor**2).astype(int)
    X_ = reduce_spatial_features(X, maxfeatures)
    assert_equal(X_.ndim, X.ndim)
    newimgs, nh, nw, newchans = X_.shape
    assert_equal(newimgs, nimgs)
    assert_equal(newchans, chans)
    # assert_equal((nh, nw), (np.ceil(h/factor), np.ceil(w/factor)))
    assert chans * nh * nw <= maxfeatures
