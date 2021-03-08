from famfaceangles.optimridge.ridge import ridge, ridge_corr, zs
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy import linalg
from sklearn.linear_model import Ridge


def test_ridge():
    nfeatures = 100
    npred = 80
    nsamples = 50
    X = np.random.randn(nsamples, npred)
    Y = np.random.randn(nsamples, nfeatures)

    wt = ridge(X, Y, 0.)
    assert_array_equal((npred, nfeatures), wt.shape)

    wt2 = ridge(X, Y, [0.] * nfeatures)
    assert_array_equal(wt, wt2)

    # In this implementation of ridge, the parameters is squared; in
    # scikit-learn instead it's not squared. so let's compare if they return
    #  similar results whenever we pass it already squared to sklearn
    alphas = np.logspace(0, 3, 10)
    for a in alphas:
        wt = ridge(X, Y, a)
        ridge_sk = Ridge(alpha=a**2, fit_intercept=False, solver='svd')
        ridge_sk.fit(X, Y)
        assert_array_almost_equal(wt, ridge_sk.coef_.T)


def test_ridge_corr():
    nfeatures = 1000
    npred = 50
    nsamples_train = 50
    nsamples_test = 20
    X_tr = np.random.randn(nsamples_train, npred)
    Y_tr = np.random.randn(nsamples_train, nfeatures)
    X_te = np.random.randn(nsamples_test, npred)
    Y_te = np.random.randn(nsamples_test, nfeatures)
    alphas = np.logspace(0, 3, 20)

    corr = ridge_corr(X_tr, X_te, Y_tr, Y_te, alphas=alphas)
    assert_array_equal(corr.shape, (len(alphas), nfeatures))


def test_ridge_corr_svd(tmpdir):
    nfeatures = 1000
    npred = 50
    nsamples_train = 50
    nsamples_test = 20
    X_tr = np.random.randn(nsamples_train, npred)
    Y_tr = np.random.randn(nsamples_train, nfeatures)
    X_te = np.random.randn(nsamples_test, npred)
    Y_te = np.random.randn(nsamples_test, nfeatures)
    alphas = np.logspace(0, 3, 20)

    # precompute SVD
    U, S, Vh = linalg.svd(X_tr, full_matrices=False)
    # store USVh
    USVh_fn = str(tmpdir.join('usvh.npz'))
    np.savez(USVh_fn, **{'U': U, 'S': S, 'Vh': Vh})

    corr = ridge_corr(X_tr, X_te, Y_tr, Y_te, alphas=alphas)
    corr1 = ridge_corr(X_tr, X_te, Y_tr, Y_te, alphas=alphas, USVh=(U, S, Vh))
    corr2 = ridge_corr(X_tr, X_te, Y_tr, Y_te, alphas=alphas, USVh=USVh_fn)
    assert_array_equal(corr, corr1)
    assert_array_equal(corr, corr2)
