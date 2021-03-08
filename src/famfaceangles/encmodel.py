"""Module containing functions for Forward-Encoding Models"""
import numpy as np
import logging
from joblib.parallel import Parallel, delayed
from .optimridge.ridge import ridge_corr, ridge, _process_svd
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import butter, filtfilt
from skimage.measure import block_reduce
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
import sys
import types

logger = logging.getLogger("encmodel")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def reduce_spatial_features(X, maxfeatures=25000):
    """
    Reduces the number of features of an image or a layer extracted from a
    Convolutional Neural Network.

    First, we find the value of the factor required to downsample the input so
    that the total number of features is less or equal than `maxfeaures`.
    Then, the input is smoothed with a 2D gaussian separately for each channel.
    Finally, the smoothed image is downsampled by averaging values in blocks
    of determined size.

    This approach is taken from Eickenberg, M., Gramfort, A., Varoquaux, G.,
    Thirion, B. (2017). Seeing it all: Convolutional network layers map the
    function of the human visual system. NeuroImage, 152, 184-194.
    https://doi.org/10.1016/j.neuroimage.2016.10.001

    Parameters
    ----------
    X : numpy.ndarray (images, height, width, channels)
        input tensor containing the images to process
    maxfeatures : int
        maximum number of features

    Returns
    -------
    X_resampled : numpy.ndarray (images, new_height, new_width, channels)
        the resampled tensor such that for each image
        `new_width * new_height * channels <= maxfeatures`
    """
    # conform input, we expect a 4D tensor
    if X.ndim == 3:
        X = X[None]
    elif X.ndim == 2:
        X = X[None, ..., None]
    # work on floats
    X = X.astype(np.float)

    images, height, width, chans = X.shape
    if height * width * chans <= maxfeatures:
        return X

    maxfeatures = float(maxfeatures)
    factor = np.ceil(np.sqrt((height*width*chans)/maxfeatures)).astype(int)
    sigma = 0.35 * factor  # see the paper
    if height % factor != 0 or width % factor != 0:
        factor += 1

    new_width, new_height = \
        np.ceil(np.array((width, height))/float(factor)).astype(int)
    X_ = np.zeros((images, chans, new_height, new_width))
    X = X.transpose([0, 3, 1, 2])
    for i_img in range(images):
        for i_chan in range(chans):
            tmp = gaussian_filter(X[i_img, i_chan, ...], sigma=(sigma, sigma))
            X_[i_img, i_chan, ...] = block_reduce(
                tmp, block_size=(factor, factor), func=np.mean)
    X_ = X_.transpose([0, 2, 3, 1])
    return X_


# from https://stackoverflow.com/questions/25191620/
# creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, method='gust')
    return y


def resample(array, origfreq=30.0, sampfreq=1.0):
    """
    Resample data to match TR frequency. First data is low-passed to sampfreq
    Hz, then it is decimated to match the TR frequency.

    Parameters
    ----------
    array : array-like
        the data to resample
    origfreq : float
        original sampling frequency (Hz)
    sampfreq : float
        new sampling frequency

    Returns
    -------
    resampled : array-like
        the resampled data
    """

    resampled = butter_lowpass_filter(array, sampfreq, origfreq)
    select = slice(None, None, int(origfreq/sampfreq))
    return resampled[select]


def _conform_input(video):
    """Return an array regardless of the input. Will have to store all data
    in memory"""
    video_ = [np.asarray(v, dtype=np.float) for v in video]
    video_ = np.array(video_)
    if video_.ndim == 4:
        video_ = video_.mean(axis=-1)
    # normalize by the mean luminance for each frame
    video_ -= video_.mean(axis=(1, 2))[:, None, None]
    return video_


def global_motion(video, prev_frame=None, max_frames=None):
    """
    Compute global motion estimate for a video. For two consecutive frames,
    global motion is computed as the average difference between the two
    frames across all channels. Each frame is normalized by the average
    luminance to avoid counting changes in luminance as motion.

    Parameters
    ----------
    video : array or iterable
        contains frames of the video. If array, the first dimension needs
        to be time. Otherwise a list of frames, or an imageio.Reader
        object can be passed.
    prev_frame : array (n_x, n_y, n_channels) or None
        previous frame used to compute the global motion for the first frame
        of the passed video. This can be used to stitch together a series of
        shorter clips.
    max_frames : int or None
        max number of frames required. Use this to make sure that the output
        has length exactly `max_frames`. If the video length is less than
        `max_frames`, the last motion estimate will be repeated to match the
        desired length.

    Returns
    -------
    motion : array-like (max_frames, )
        the global motion estimate
    """
    video = _conform_input(video)
    prev = None
    if prev_frame is not None:
        prev = prev_frame.copy()
        # preprocess prev as _conform_input
        prev = _conform_input([prev])[0]
    n_frames = len(video)
    max_frames = n_frames if max_frames is None else max_frames
    extra_frames = max_frames - n_frames
    video = video[:max_frames]
    motion = np.sqrt(
        np.mean((video[1:] - video[:-1])**2, axis=(1, 2))).tolist()
    first = 0. if prev is None else np.sqrt(np.mean((video[0] - prev)**2))
    motion = [first] + motion + [motion[-1]] * extra_frames
    return np.asarray(motion)


def add_delays(X, delays=(2, 4, 6), tr=1):
    """
    Given a design matrix X, add additional columns for delayed responses.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
    delays : array-like (n_delays, )
        delays in seconds
    tr : float

    Returns
    -------
    Xd : array (n_samples, n_predictors * n_delays)
        design matrix with additional predictors
    """
    s, p = X.shape
    d = len(delays)
    Xd = np.zeros((s, p * (d + 1)))
    Xd[:, :p] = X
    for i, delay in enumerate(delays, 2):
        start = int(delay * tr)
        Xd[start:, (i - 1) * p:i * p] = X[:s - start]
    return Xd


def zscore(x):
    """x is (samples, features). zscores each sample individually"""
    axis = None if x.ndim == 1 else 1
    return (x - x.mean(axis=axis)[:, None])/x.std(axis=axis, ddof=1)[:, None]


def corr(x, y):
    """Return the correlation between the pairwise rows of x and y"""
    nf = len(x) if x.ndim == 1 else x.shape[1]
    axis = None if x.ndim == 1 else 1
    x_ = zscore(x)
    y_ = zscore(y)
    r = (x_ * y_).sum(axis=axis)/(nf-1)
    return r


def _ridge_search(X, Y, train, test, alphas=np.logspace(0, 4, 20),
                  scoring=corr, method='optimridge'):
    """Fit ridge regression sweeping through alphas

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    train : array
        index array for training
    test : array
        index array for testing
    alphas : array
        alpha values used to fit
    scoring : callable
        function used to score the fit (default correlation)
        this is not used if method is 'optimridge'
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth

    Returns
    -------
    scores : array (n_alphas, n_features)
        score of prediction for each alpha and each feature
    """
    if method not in ['optimridge', 'sklearn']:
        raise ValueError("Method {} is not valid".format(method))
    if method == 'sklearn':
        ridge = Ridge(fit_intercept=False, solver='svd')
        scores = []
        for alpha in alphas:
            ridge.set_params(alpha=alpha)
            ridge.fit(X[train], Y[train])
            scores.append(scoring(Y[test].T, ridge.predict(X[test]).T))
    else:
        scores = ridge_corr(X[train], X[test],
                            Y[train], Y[test],
                            alphas=alphas)
    return np.array(scores)


def ridge_search(X, Y, cv, alphas=np.logspace(0, 4, 20), scoring=corr,
                 method='optimridge',
                 njobs=1):
    """Fit ridge regression sweeping through alphas across all
    cross-validation folds

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    cv : iterable or generator
        returning (train, test) tuples with indices for row of X and Y
    alphas : array
        alpha values used to fit
    scoring : callable
        function used to score the fit (default correlation)
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth
    njobs : int
        number of parallel jobs to run. Each cross-validation fold will be
        run in parallel

    Returns
    -------
    scores : array (n_folds, n_alphas, n_features)
        score of prediction for each alpha and each feature
    """
    scores = Parallel(n_jobs=njobs, backend='multiprocessing')(
        delayed(_ridge_search)(X, Y, train, test, alphas=alphas,
                               scoring=scoring, method=method)
        for train, test in cv)
    return np.array(scores)


def ridge_optim_fold(X, Y, group, train, test, alphas,
                     method='optimridge',
                     njobs=1, nblocks=1):
    """First step for nested cross-validation. It will perform parameter search
    within the training set, and then return the best alpha and the score
    (correlation) curves. Parameter search is performed by averaging the
    prediction score curves across features and folds, and finding the
    optimal global alpha (across features and folds).

    Note that X and Y are assumed to be centered (e.g., z-scored), because
    the intercept is not fitted.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation within training set will be performed according to
        this grouping factor using LeaveOneGroupOut.
    train : array
        indices for training set
    test : array
        indices for testing set (not used but just for consistent API)
    alphas : array
        alphas to search through for optimal search
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth
    njobs : int
        number of parallel jobs
    nblocks : int
        number of blocks for parallel jobs (more use less memory)

    Returns
    -------
    best_alpha : float
        global best alpha used to obtain score
    score_curve : array (n_alphas)
        global score curve across n_features
    """
    # Split into training, testing
    y_tr, y_te = Y[train], Y[test]
    x_tr, x_te = X[train], X[test]
    # grouping factor, e.g., runs
    gr_tr, gr_te = group[train], group[test]
    n_features = Y.shape[1]
    n_blocks = max(njobs, nblocks)
    blocks = np.array_split(np.arange(n_features), n_blocks)
    # Work on training, find best alpha by cross-validation
    # this is parallelized across blocks to reduce memory consumption
    # score_tr is (n_folds, n_alphas, n_features)
    cv = list(LeaveOneGroupOut().split(x_tr, groups=gr_tr))
    score_tr = Parallel(n_jobs=njobs, backend='multiprocessing')(
        delayed(ridge_search)(x_tr, y_tr[:, ib], cv, alphas=alphas,
                              method=method) for ib in
        blocks)
    score_tr = np.dstack(score_tr)
    score_curve = score_tr.mean(axis=(0, 2))
    # because this is correlation, we need to find the max
    best_alpha = alphas[np.argmax(score_curve)]
    return best_alpha, score_curve


def ridge_optim(X, Y, group, cv, alphas, method='optimridge',
                njobs=1, nblocks=1):
    """Main loop for nested cross-validation. It will perform parameter search
    within the training set, and then return the best alpha and the score
    (correlation). Parameter search is performed by averaging the prediction
    score curves across features and folds of the inner cross-validation loop,
    and finding the optimal global alpha (across features and folds).

    Note that X and Y are assumed to be centered (e.g., z-scored), because
    the intercept is not fitted.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation within training set will be performed according to
        this grouping factor using LeaveOneGroupOut.
    cv : generator or list
        generates (train, test) splits
    alphas : array
        alphas to search through for optimal search
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth
    njobs : int
        number of parallel jobs
    nblocks : int
        number of blocks for parallel jobs (more use less memory)

    Returns
    -------
    best_alpha : array (n_folds, )
        global best alpha used to obtain score
    score_curve : array (n_folds, n_alphas)
        global score curve
    """
    out = []
    for train, test in cv:
        out.append(ridge_optim_fold(X, Y, group, train, test, alphas,
                                    method, njobs, nblocks))
    best_alphas, score_curve = zip(*out)
    return np.array(best_alphas), np.array(score_curve)


def _fit_ridge(X, Y, cv, alphas, method='optimridge',
               save_results_todisk=''):
    """

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    cv : generator or list
        generates (train, test) splits
    alphas : array (n_folds)
        alpha to use for each fold of cv
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth
    save_results_todisk : str
        if it's a non-empty string, then both score and weights will be
        saved to disk instead of being returned. This is needed whenever
        `n_predictors` and `n_features` are large, since the results will be
        very large. The string needs to have two elements to be filled using
        the .format syntax, for example
        '/path/to/output/sub-sid000021_cv{0:02d}_{1}'.
        This will save the weights for the first CV fold as
        '/path/to/output/sub-sid000021_cv01_coef.npz' and the scores as
        '/path/to/output/sub-sid000021_cv01_scores.npz'. Data is saved as
        float32 to reduce space.

    Returns
    -------
    score : array (n_folds, n_features) or []
        prediction scores for each feature and each fold.
        list of filenames if `save_results_todisk` is set
    weights : array (n_folds, n_features, n_predictors) or []
        coefficient weights.
        list of filenames if `save_results_todisk` is set
    """
    if method not in ['optimridge', 'sklearn']:
        raise ValueError("Method {} is not valid".format(method))
    score = []
    coef = []
    for i_cv, (alpha, (train, test)) in enumerate(zip(alphas, cv)):
        x_tr, x_te = X[train], X[test]
        y_tr, y_te = Y[train], Y[test]
        if method == 'sklearn':
            ridge_sk = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
            ridge_sk.fit(x_tr, y_tr)
            pred = ridge_sk.predict(x_te)
            c = ridge_sk.coef_
        else:
            wt = ridge(x_tr, y_tr, alpha=np.array([alpha]))
            pred = np.dot(x_te, wt)
            c = wt.T
        sc = corr(pred.T, y_te.T).astype(np.float32)
        c = c.astype(np.float32)
        if save_results_todisk:
            fn_coef = save_results_todisk.format(i_cv, 'coef') + '.npz'
            coef.append(fn_coef)
            fn_score = save_results_todisk.format(i_cv, 'scores') + '.npz'
            score.append(fn_score)
            logger.info("Saving coef to {}".format(fn_coef))
            np.savez(fn_coef, coef=c)
            logger.info("Saving scores to {}".format(fn_score))
            np.savez(fn_score, scores=sc)
        else:
            score.append(sc)
            coef.append(c)
    if score:
        score = np.stack(score)
        coef = np.stack(coef)
    return score, coef


def _fit_ridge_score(X, Y, cv, alphas):
    """
    Fit ridge and return only scores

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    cv : generator or list
        generates (train, test) splits
    alphas : array (n_folds)
        alpha to use for each fold of cv

    Returns
    -------
    score : array (n_folds, n_features)
        prediction scores for each feature and each fold.
    """
    score = []
    for i_cv, (alpha, (train, test)) in enumerate(zip(alphas, cv)):
        x_tr, x_te = X[train], X[test]
        y_tr, y_te = Y[train], Y[test]
        sc = ridge_corr(x_tr, x_te, y_tr, y_te, alphas=np.array([alpha]))
        score.append(sc)
    score = np.vstack(score)
    return score


def fit_encmodel(X, Y, group, alphas, method='optimridge', njobs=1,
                 nblocks=1, save_results_todisk='', compute_coef=True):
    """
    Fit encoding model using features in X to predict data in Y, testing a
    range of different alphas.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : array (n_samples, n_features)
        response matrix
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation will be performed according to this grouping factor
        using LeaveOneGroupOut.
    alphas : array
        alphas to search through for optimal search
    method : str ('optimridge' | 'sklearn')
        which implementation to use.
        'sklearn' uses Ridge from scikit-learn
        'optimridge' (default) uses the optimized implementation by Alex Huth
    njobs : int
        number of parallel jobs
    nblocks : int
        number of blocks for parallel jobs (more use less memory)
    save_results_todisk : str
        if it's a non-empty string, then both score and weights will be
        saved to disk instead of being returned. This is needed whenever
        `n_predictors` and `n_features` are large, since the results will be
        very large. The string needs to have two elements to be filled using
        the .format syntax, for example
        '/path/to/output/sub-sid000021_cv{0:02d}_{1}'.
        This will save the weights for the first CV fold as
        '/path/to/output/sub-sid000021_cv01_coef.npz' and the scores as
        '/path/to/output/sub-sid000021_cv01_scores.npz'. Data is saved as
        float32 to reduce space.
    compute_coef : bool
        whether to compute the coefficients for the regression with the
        optimal alphas. If the numbers of predictors and features are large,
        the coefficients can occupy a lot of disk space. Also, it's faster to
        compute predictions (and scores) without computing the coefficients.
        `compute_coef` can be set to False only if `method == 'optimridge'`.

    Returns
    -------
    score : array (n_folds, n_features) or list
        score of optimal prediction for each feature. Or list of filenames
        if `save_results_todisk` is set
    weights : array (n_folds, n_features, n_predictors)
        weights of optimal estimator. Or list of filenames
        if `save_results_todisk` is set
    best_alpha : array (n_folds, )
        global best alpha used to obtain score
    score_curve : array (n_folds, n_alphas)
        global score curves

    """
    if method == 'sklearn' and not compute_coef:
        raise ValueError("Only method='optimridge' is allowed if "
                         "compute_coef is False")
    logo = LeaveOneGroupOut()
    cv = list(logo.split(X, groups=group))
    if method == 'sklearn':
        # square alphas so we get the same values for both methods
        # since the Huth's one uses squared alphas while sklearn uses alphas
        alphas = alphas ** 2
    # Step 1. Find optimal alphas across folds
    logger.info("Running Step 1: finding optimal alpha")
    n_features = Y.shape[1]
    if n_features <= njobs:
        n_blocks = n_features
    else:
        n_blocks = max(njobs, nblocks)
    blocks = np.array_split(np.arange(n_features), n_blocks)
    out = Parallel(n_jobs=njobs, verbose=50, backend='multiprocessing')(
        delayed(ridge_optim)(X, Y[:, ib], group, cv, alphas, method=method)
        for ib in blocks)
    _, score_curve = zip(*out)
    score_curve = np.dstack(score_curve).mean(axis=-1)
    best_alphas = alphas[np.argmax(score_curve, axis=1)]

    # Step 2. Fit with best alphas
    logger.info("Got the following best alphas {}".format(best_alphas))
    logger.info("Running Step 2: fitting with optimal alpha")
    # out = Parallel(n_jobs=njobs, verbose=50, backend='multiprocessing')(
    #     delayed(_fit_ridge)(X, Y[:, ib], cv, best_alphas, method=method,
    #                         save_results_todisk=save_results_todisk)
    #     for ib in blocks)
    # final_score, coef = zip(*out)
    # final_score = np.hstack(final_score)
    # coef = np.hstack(coef)
    if compute_coef:
        final_score, coef = _fit_ridge(X, Y, cv, best_alphas, method=method,
                                       save_results_todisk=save_results_todisk)
    else:
        final_score = _fit_ridge_score(X, Y, cv, best_alphas)
        coef = []

    if method == 'sklearn':
        # now square root the best alphas so we get the alphas in the
        # original order of magnitude
        best_alphas = np.sqrt(best_alphas)
    return final_score, coef, best_alphas, score_curve


def fit_encmodel_bwsj(X, Y, group, alphas, compute_coef=True):
    """
    Fit encoding model across subjects using features in X to predict data
    in Y, testing a range of different alphas.

    Compared to `fit_encmodel`, this version assumes that a list of N datasets
    is given as input. The model is fitted on the average of N-1 datasets
    (and M-1 runs) and tested on the left-out run of the left-out dataset.

    The parameter search is performed as in `fit_encmodel` in the average of
    the N-1 datasets leaving one run out for testing.

    Parameters
    ----------
    X : array (n_samples, n_predictors)
        design matrix/predictors
    Y : list of arrays (n_samples, n_features)
        list of response matrices. The order of the samples is assumed to be
        consistent across matrices.
    group : array (n_samples,)
        indicator variable used for grouping the sample (e.g., runs or chunks),
        cross-validation will be performed according to this grouping factor
        using LeaveOneGroupOut.
    alphas : array
        alphas to search through for optimal search
    compute_coef : bool
        whether to compute the coefficients for the regression with the
        optimal alphas. If the numbers of predictors and features are large,
        the coefficients can occupy a lot of disk space. Also, it's faster to
        compute predictions (and scores) without computing the coefficients.
        `compute_coef` can be set to False only if `method == 'optimridge'`.

    Returns
    -------
    score : array (n_folds, n_features) or list
        score of optimal prediction for each feature. Or list of filenames
        if `save_results_todisk` is set
    weights : array (n_folds, n_features, n_predictors)
        weights of optimal estimator. Or list of filenames
        if `save_results_todisk` is set
    best_alpha : array (n_folds, )
        global best alpha used to obtain score
    score_curve : array (n_folds, n_alphas)
        global score curves

    """
    logo = LeaveOneGroupOut()
    cv = list(logo.split(X, groups=group))
    nds = len(Y)
    final_score = []
    coef = []
    best_alphas = []
    score_curve = []
    for icv, (train, test) in enumerate(cv):
        logger.info(f"Running fold {icv}/{len(cv)}")
        # set up internal cross-validation loop for parameter search
        group_tr = group[train]
        X_tr = X[train]
        # STEP 1. PARAMETER SEARCH
        cv_search = list(logo.split(X_tr, groups=group_tr))
        score_search = []
        for icvs, (train_search, test_search) in enumerate(cv_search):
            logger.info(f"Running search fold {icvs}/{len(cv_search)}")
            X_search_tr = X_tr[train_search]
            X_search_te = X_tr[test_search]
            # precompute SVD for X_search_tr
            USVh_search_tr = _process_svd(X_search_tr, singcutoff=1e-15)
            # now we need to get the actual data
            score_search_subj = []
            for ids in range(nds):
                Y_tr = np.mean(
                    [Y[i][train] for i in range(nds) if i != ids],
                    axis=0)
                # select data for search
                Y_search_tr = Y_tr[train_search]
                Y_search_te = Y_tr[test_search]
                # now get scores averaged across features in this search fold
                score = ridge_corr(
                    X_search_tr, X_search_te,
                    Y_search_tr, Y_search_te,
                    USVh=USVh_search_tr,
                    alphas=alphas,
                    singcutoff=None).mean(axis=1)
                score_search_subj.append(score)
            # this is a list of length nfolds of lists of length nsubjects
            score_search.append(score_search_subj)
        # now I need to get for each left-out subject the average of the nfolds
        score_curve_fold = []
        for ids in range(nds):
            score_curve_fold.append(
                np.mean([fold[ids] for fold in score_search], axis=0))
        score_curve_fold = np.stack(score_curve_fold)  # nsubjects, nalphas
        # now get best alpha across internal folds
        alpha = alphas[np.argmax(score_curve_fold, axis=1)]
        logger.info(f"Got best alphas {alpha}")
        # store
        score_curve.append(score_curve_fold)
        best_alphas.append(alpha)

        # STEP 2. Fit on the whole dataset
        logger.info("Fitting on whole dataset")
        X_te = X[test]
        # precompute SVD for X_tr
        USVh_tr = _process_svd(X_tr, singcutoff=1e-15)
        for ids in range(nds):
            alpha_ = alpha[ids]
            Y_tr = np.mean([Y[i][train] for i in range(nds) if i != ids],
                           axis=0)
            Y_te = Y[ids][test]
            if compute_coef:
                wt = ridge(X_tr, Y_tr, alpha=np.array([alpha_]),
                           USVh=USVh_tr, singcutoff=None)
                pred = np.dot(X_te, wt)
                c = wt.T
                sc = corr(pred.T, Y_te.T)[None]
            else:
                sc = ridge_corr(X_tr, X_te, Y_tr, Y_te,
                                alphas=np.array([alpha_]), USVh=USVh_tr,
                                singcutoff=None)
                c = []
            final_score.append(sc)
            if compute_coef:
                coef.append(c)
    final_score = np.vstack(final_score)
    coef = np.stack(coef) if coef else []
    best_alphas = np.hstack(best_alphas)
    score_curve = np.vstack(score_curve)
    return final_score, coef, best_alphas, score_curve
