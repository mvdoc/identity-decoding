"""Module containing optimized ridge regression. Original implementation by
Alex Huth, available here: https://github.com/alexhuth/ridge

Refactoring and modifications by Matteo Visconti di Oleggio Castello"""


import numpy as np
import itertools as itools
import logging
import random
import sys

from numpy.linalg import multi_dot
from scipy import linalg
from .utils import mult_diag, counter
ridge_logger = logging.getLogger("ridge_corr")
ridge_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ridge_logger.addHandler(ch)


def zs(v):
    """Zscore columns of v, using dof=1"""
    return (v-v.mean(axis=0))/v.std(axis=0, ddof=1)


def ridge(X, Y, alpha, singcutoff=1e-15, normalpha=False,
          logger=ridge_logger, USVh=None):
    """Uses ridge regression to find a linear transformation of [X] that
    approximates [Y]. The regularization parameter is [alpha].

    Parameters
    ----------
    X : array_like, shape (T, N)
        Stimuli with T time points and N features.
    Y : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is
        applied to
        all M responses) or separate values for each response.
    singcutoff : float or None
        The first step in ridge regression is computing the singular value
        decomposition (SVD) of the stimulus X_tr. If X_tr is not full rank,
        some singular values will be approximately equal to zero and the
        corresponding singular vectors will be noise.
        These singular values/vectors should be removed both for speed
        (the fewer multiplications the better!) and accuracy. Any singular
        values less than singcutoff will be removed.
        if None, no singular values will be removed.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest
        singular value of X. Good for
        comparing models with different numbers of parameters.
    logger : logger instance
    USVh : None or tuple or filename
        pre-computed SVD for X. Can be passed either as a tuple (U, S,
        Vh) or as a filename .npz, where matrices are stored with names
        ['U', 'S', 'Vh'].

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    alpha = np.asarray(alpha)
    U, S, Vh = _process_svd(X, USVh, singcutoff, logger)
    # precompute UY
    UtY = np.dot(U.T, Y)

    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalpha = alpha * norm
    else:
        nalpha = alpha

    # Compute weights -- this code is similar to _solve_svd in scikit-learn
    # Except that scikit-learn uses alpha, while here we use alpha ** 2
    S = S[:, None]
    D = S/(S**2 + nalpha**2)
    # this is like dot product of diagonal and a matrix
    DUtY = D * UtY
    return np.dot(Vh.T, DUtY)


def ridge_corr(X_tr, X_te, Y_tr, Y_te, alphas, normalpha=False,
               singcutoff=1e-15, use_corr=True, logger=ridge_logger,
               USVh=None):
    """Uses ridge regression to find a linear transformation of [X_tr] that
    approximates [Y_tr], then tests by comparing the transformation of
    [X_te] to [Y_te]. This procedure is repeated for each regularization
    parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT
    returned, because computing the correlations without computing regression
    weights is much, MUCH faster.

    Parameters
    ----------
    X_tr : array_like, shape (TR, N)
        Training stimuli with TR time points and N features.
        Each feature should be Z-scored across time.
    X_te : array_like, shape (TP, N)
        Test stimuli with TP time points and N features.
        Each feature should be Z-scored across time.
    Y_tr : array_like, shape (TR, M)
        Training responses with TR time points and M responses
        (voxels, neurons, what-have-you). Each response should be Z-scored
        across time.
    Y_te : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced.
        np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular
        value (LSV) norm of X_tr. Good for comparing models with different
        numbers of parameters.
    singcutoff : float or None
        The first step in ridge regression is computing the singular value
        decomposition (SVD) of the stimulus X_tr. If X_tr is not full rank,
        some singular values will be approximately equal to zero and the
        corresponding singular vectors will be noise.
        These singular values/vectors should be removed both for speed
        (the fewer multiplications the better!) and accuracy. Any singular
        values less than singcutoff will be removed.
        if None, no singular values will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model
        fit.
        If False, this function will instead use variance explained
        (R-squared) as its metric of model fit. For ridge regression this can
        make a big difference -- highly regularized solutions will have very
        small norms and will thus explain very little variance while still
        leading to high correlations, as correlation is scale-free
        while R**2 is not.
    logger : logger instance
    USVh : None or tuple or filename
        pre-computed SVD for X_tr. Can be passed either as a tuple (U, S,
        Vh) or as a filename .npz, where matrices are stored with names
        ['U', 'S', 'Vh'].

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of
        Presp for each alpha.
    
    """
    U, S, Vh = _process_svd(X_tr, USVh, singcutoff, logger)
    # Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: {0:0.03f}".format(norm))
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    # Precompute some products for speed
    logger.info("Precomputing product matrices")
    UtY = np.dot(U.T, Y_tr)
    XV = np.dot(X_te, Vh.T)
    logger.info("Done precomputing product matrices")

    zY_te = zs(Y_te)
    nsamples = zY_te.shape[0]
    if not use_corr:
        Prespvar_actual = Y_te.var(0)
        Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
        logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = []  # Holds training correlations for each alpha
    # add extra dim to use broadcasting for diagonal multiplication later
    S = S[:, None]
    for na, a in zip(nalphas, alphas):
        logger.info("Computing correlation for alpha = {0}".format(a))
        # Reweight singular vectors by the (normalized?) ridge parameter
        D = S / (S ** 2 + na ** 2)
        # pred = np.dot(mult_diag(D, XV, left=False), UtY)
        DUtY = D * UtY
        pred = np.dot(XV, DUtY)
        if use_corr:
            Rcorr = (zY_te * zs(pred)).sum(0)/(nsamples - 1)
        else:
            # Compute variance explained
            resvar = (Y_te - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
        # Rcorr[np.isnan(Rcorr)] = 0
        # Rcorr = np.nan_to_num(Rcorr)
        Rcorrs.append(Rcorr)
    return np.asarray(Rcorrs)


def _process_svd(X, USVh=None, singcutoff=None, logger=ridge_logger):
    """
    Process SVD for X

    Parameters
    ----------
    X : array_like, shape (TR, N)
        Stimuli with TR time points and N features.
        Each feature should be Z-scored across time.
    USVh : None or tuple or filename
        pre-computed SVD for X_tr. Can be passed either as a tuple (U, S,
        Vh) or as a filename .npz, where matrices are stored with names
        ['U', 'S', 'Vh'].
    singcutoff : float or None
        The first step in ridge regression is computing the singular value
        decomposition (SVD) of the stimulus X_tr. If X_tr is not full rank,
        some singular values will be approximately equal to zero and the
        corresponding singular vectors will be noise.
        These singular values/vectors should be removed both for speed
        (the fewer multiplications the better!) and accuracy. Any singular
        values less than singcutoff will be removed.
        if None, no singular values will be removed.
    logger : logger instance

    Returns
    -------
    U, S, Vh : arrays
        SVD decomposition of X
    """
    # Calculate SVD of stimulus matrix
    if USVh is None:
        logger.info("Running SVD...")
        try:
            # Use scipy's SVD which is faster
            U, S, Vh = linalg.svd(X, full_matrices=False)
        except ValueError:
            # for some obscure reasons, this SVD in numpy and scipy (default
            # using 'gesdd' lapack driver) fails even if there are no NaNs in
            # the matrix, with the error message
            #
            # ValueError: On entry to DLASCL parameter number 4 had an illegal
            # value
            #
            # Using scipy's linalg and selecting `gesvd` driver might solve
            # the problem
            logger.warning("NumPy's SVD didn't converge, trying with scipy's "
                           "and different driver")
            U, S, Vh = linalg.svd(X, full_matrices=False,
                                  lapack_driver='gesvd')
        logger.info("Done running SVD...")
    elif isinstance(USVh, (tuple, list)):
        logger.info("Using passed SVD...")
        U, S, Vh = USVh
    elif isinstance(USVh, str):
        if not USVh.endswith('.npz'):
            raise ValueError("I can only load npz files, not {}".format(
                USVh.split('.')[-1]))
        logger.info("Loading SVD from {}...".format(USVh))
        tmp = np.load(USVh)
        U = tmp['U']
        S = tmp['S']
        Vh = tmp['Vh']
        del tmp

    # Truncate tiny singular values for speed
    if singcutoff is not None:
        origsize = S.shape[0]
        ngoodS = np.sum(S > singcutoff)
        nbad = origsize - ngoodS
        U = U[:, :ngoodS]
        S = S[:ngoodS]
        Vh = Vh[:ngoodS]
        logger.info(
            "Dropped {0} tiny singular values. (U is now {1})".format(nbad,
                                                                      U.shape))
    return U, S, Vh


def bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, nboots, chunklen, nchunks,
                    corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
                    use_corr=True, logger=ridge_logger):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.
    
    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).
    
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    
    Returns
    -------
    wt : array_like, shape (N, M)
        Regression weights for N features and M responses.
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    valinds = [] # Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = zip(*[iter(allinds)]*chunklen)
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        valinds.append(heldinds)
        
        RRstim = Rstim[notheldinds,:]
        PRstim = Rstim[heldinds,:]
        RRresp = Rresp[notheldinds,:]
        PRresp = Rresp[heldinds,:]
        
        # Run ridge regression using this test set
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr,
                           logger=logger)
        
        Rcmats.append(Rcmat)
    
    # Find best alphas
    if nboots>0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None
    
    if not single_alpha:
        if nboots==0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")
        
        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        if nboots==0:
            if len(alphas)==1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = allRcorrs.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]
        
        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)

    # Find weights
    logger.info("Computing weights for each response using entire training set..")
    wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)

    # Predict responses on prediction set
    logger.info("Predicting responses for predictions set..")
    pred = np.dot(Pstim, wt)

    # Find prediction correlations
    nnpred = np.nan_to_num(pred)
    if use_corr:
        corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
                                        for ii in range(Presp.shape[1])]))
    else:
        resvar = (Presp-pred).var(0)
        Rsqs = 1 - (resvar / Presp.var(0))
        corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)

    return wt, corrs, valphas, allRcorrs, valinds
