#!/usr/bin/env python
"""Script to run decoding or RSA within face-responsive ROIs. If decoding
is run, for each ROI, the number of features is selected by cross-validation
in the training set, selecting the percentage of features that yields maximal
training accuracy. If RSA is run, partial correlation is computed between
the neural RDM and three predictors (identity, mirror symmetry,
and orientation). It is possible to run permutations to obtain an
estimate of the null distribution."""
# ignore futurewarnings so we can see what's happening
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os.path as op
from mvpa2.suite import *
from famfaceangles.mvpa import PCorrTargetSimilarity
from run_sl import (_load_ds, _setup_ds, _load_surface, _make_eventrelated,
                    _filter_targets, _run_sl, _prepare_labels,
                    setup_partitioner)
from run_rsa import PREDFNS, LBL2TARGETRDMS, _make_prednames, filter_ds as \
    filter_ds_rsa
from joblib.parallel import Parallel, delayed
from numpy.testing import assert_array_equal

DERIVDIR = '/idata/DBIC/castello/famface_angles/derivatives'
GLMDIR = 'glm-tent7-nosmooth'
MAXINT = 2**32


def load_subj_ds(subj, target, fshemi, glmdir=GLMDIR):
    """Load dataset for one subject, and perform default pre-processing for
    specific target classification."""
    hemi = 'L' if fshemi == 'lh' else 'R'
    glmdir = op.join(DERIVDIR, glmdir)
    if 'tent' in glmdir:
        glmtype = 'tent'
        task = 'both'
    elif 'block' in glmdir:
        glmtype = 'block'
        task = target[:3]
        assert task in ['fam', 'str']
        task += '1back'
    else:
        raise ValueError("I know only tent or block")
    tent_template = '{0}_task-{1}_space-fsaverage6_hemi-{2}_' \
                    'deconvolve-{3}.niml.dset'
    tent_fn = op.join(
        glmdir, subj,
        tent_template.format(subj, task, hemi, glmtype))
    ds, is_surface = _load_ds(tent_fn)
    if 'node_indices' not in ds.fa:
        ds.fa['node_indices'] = np.arange(ds.nfeatures)
    # setup
    ds = _setup_ds(ds)
    if 'id+or+mi' in target:
        ds = filter_ds_rsa(ds, target[:3])
        # check the order is what we expect for predictors
        assert_array_equal(sorted(ds.sa.labels), ds.sa.labels)
        targets_filtered = ''
    else:
        ds, targets_filtered = _filter_targets(ds, target)
    if ds.sa.ntent.max() > 0:
        # keep only first five tent
        ds = ds[ds.sa.ntent < 5]
        ds = _make_eventrelated(ds)
    else:
        ds = _prepare_labels(ds)
        # if we are classifying within male/female, filter out
        if targets_filtered.endswith('female'):
            ds = ds[ds.sa.gender == 0]
            targets_filtered = targets_filtered.replace('-female', '')
        elif targets_filtered.endswith('male'):
            ds = ds[ds.sa.gender == 1]
            targets_filtered = targets_filtered.replace('-male', '')
        ds.sa['chunks'] = ds.sa.run
    if targets_filtered:
        ds.sa['targets'] = ds.sa[targets_filtered]
    zscore(ds)
    return ds


def _crossval_percent_feature(ds, partitioner, percent_features=0.25,
                              clf=LinearCSVMC()):
    """Run decoding with fixed feature selection with cross-validation,
    used in Step 1 for parameter search"""
    fsel = FractionTailSelector(felements=percent_features,
                                tail='upper', mode='select', sort=False)
    sbfs = SensitivityBasedFeatureSelection(
        OneWayAnova(), fsel, enable_ca=['sensitivities'])
    fsclf = FeatureSelectionClassifier(clf, sbfs)
    cvte = CrossValidation(fsclf, partitioner, errorfx=mean_match_accuracy)
    out = cvte(ds)
    return out


def _traintest_percent_feature(ds, percent_features, clf=LinearCSVMC(),
                               splitter=Splitter('partitions')):
    """Run decoding once without cross-validation; ds must have
    ds.sa.partitions. This is used in Step 2 to fit once the best number of
    features has been found"""
    fsel = FractionTailSelector(felements=percent_features,
                                tail='upper', mode='select', sort=False)
    sbfs = SensitivityBasedFeatureSelection(
        OneWayAnova(), fsel, enable_ca=['sensitivities'])
    fsclf = FeatureSelectionClassifier(clf, sbfs)
    tm = TransferMeasure(
        fsclf, splitter,
        postproc=BinaryFxNode(mean_match_accuracy, space='targets'))
    return tm(ds)


def _traintest_percent_feature_cf(ds, percent_features, clf=LinearCSVMC(),
                                  splitter=Splitter('partitions'),
                                  enable_stats=True):
    """Run decoding once without cross-validation; ds must have
    ds.sa.partitions. This is used in Step 2 to fit once the best number of
    features has been found"""
    fsel = FractionTailSelector(felements=percent_features,
                                tail='upper', mode='select', sort=False)
    sbfs = SensitivityBasedFeatureSelection(
        OneWayAnova(), fsel, enable_ca=['sensitivities'])
    fsclf = FeatureSelectionClassifier(clf, sbfs)
    enable_ca = ['stats'] if enable_stats else []
    tm = TransferMeasure(
        fsclf, splitter,
        postproc=BinaryFxNode(mean_match_accuracy, space='targets'),
        enable_ca=enable_ca)
    out = tm(ds)
    cf = tm.ca.stats.matrix if enable_stats else []
    return out, cf


def run_decoding_featsel_cv(ds, partitioner,
                            percent_features_grid=np.round(
                                np.arange(0.1, 1.05, 0.05), 2),
                            nproc=1):
    """Run decoding with feature selection by selecting the optimal number
    of features in the training set using F-scores and parameter search.

    Parameters
    ----------
    ds : mvpa2.Dataset
    partitioner : partitioner
        partitioner used to divide into training/testing set; the same
        partitioner will be used within the training set to perform
        parameter search
    percent_features_grid : array_like
        parameter grid to search through. Must be within 0 and 1, and represent
        percentage of top features. Default is from 10% to 100% in 5% steps
    nproc : int
        number of parallel processes

    Returns
    -------
    acc_fold : mvpa2.Dataset
        Dataset containing accuracies for each fold, after selecting optimal
        number of features in each training set
    best_pfs : list
        optimal percentage feature selected for each fold
    out_pf : list of list
        Each element of out_pf correspond to the result of the parameter
        search for each iteration.
        You should average each of these elements to obtain the accuracy
        curve whose argmax was used to select the optimal number of features.
    """
    splitter = Splitter('partitions')
    out_pf = []  # accuracies for parameter search
    acc_fold = []  # accuracies with best parameter
    best_pfs = []  # best fraction of features for each fold
    for isplit, ds_part in enumerate(partitioner.generate(ds)):
        print("Running fold {}".format(isplit + 1))
        # XXX: changed these in case we do between subject decoding so we
        # have more than two partitions
        split_ds = list(splitter.generate(ds_part))
        ds_train = split_ds[0]
        # Step 1. Find optimal number of features in training set
        out_pf_fold = Parallel(n_jobs=nproc)(
            delayed(_crossval_percent_feature)
            (ds_train, partitioner, percent_features=pf)
            for pf in percent_features_grid)
        # average across folds in training
        acc_pf_fold = [x.samples.mean() for x in out_pf_fold]
        # get best number of features, and prefer less features in case of ties
        best_pf = percent_features_grid[np.argmax(acc_pf_fold)]
        best_pfs.append(best_pf)
        # store accuracies for parameter search
        out_pf.append(out_pf_fold)

        # Step 2. Train/test using optimal number of features
        acc_fold.append(_traintest_percent_feature(ds_part, best_pf))
    acc_fold = vstack(acc_fold)
    return acc_fold, best_pfs, out_pf


def run_decoding_featsel_cv_cf(ds, partitioner,
                               percent_features_grid=np.round(
                                 np.arange(0.1, 1.05, 0.05), 2),
                               nproc=1):
    """Run decoding with feature selection by selecting the optimal number
    of features in the training set using F-scores and parameter search.
    Also returns confusion matrices.

    Parameters
    ----------
    ds : mvpa2.Dataset
    partitioner : partitioner
        partitioner used to divide into training/testing set; the same
        partitioner will be used within the training set to perform
        parameter search
    percent_features_grid : array_like
        parameter grid to search through. Must be within 0 and 1, and represent
        percentage of top features. Default is from 10% to 100% in 5% steps
    nproc : int
        number of parallel processes

    Returns
    -------
    acc_fold : mvpa2.Dataset
        Dataset containing accuracies for each fold, after selecting optimal
        number of features in each training set
    best_pfs : list
        optimal percentage feature selected for each fold
    out_pf : list of list
        Each element of out_pf correspond to the result of the parameter
        search for each iteration.
        You should average each of these elements to obtain the accuracy
        curve whose argmax was used to select the optimal number of features.
    confusion_matrices : list of array
        Final confusion matrices for each fold.
    """
    splitter = Splitter('partitions')
    out_pf = []  # accuracies for parameter search
    acc_fold = []  # accuracies with best parameter
    best_pfs = []  # best fraction of features for each fold
    confusion_matrices = []
    for isplit, ds_part in enumerate(partitioner.generate(ds)):
        print("Running fold {}".format(isplit + 1))
        # XXX: changed these in case we do between subject decoding so we
        # have more than two partitions
        split_ds = list(splitter.generate(ds_part))
        ds_train = split_ds[0]
        # Step 1. Find optimal number of features in training set
        out_pf_fold = Parallel(n_jobs=nproc)(
            delayed(_crossval_percent_feature)
            (ds_train, partitioner, percent_features=pf)
            for pf in percent_features_grid)
        # average across folds in training
        acc_pf_fold = [x.samples.mean() for x in out_pf_fold]
        # get best number of features, and prefer less features in case of ties
        best_pf = percent_features_grid[np.argmax(acc_pf_fold)]
        best_pfs.append(best_pf)
        # store accuracies for parameter search
        out_pf.append(out_pf_fold)

        # Step 2. Train/test using optimal number of features
        acc, cf = _traintest_percent_feature_cf(ds_part, best_pf)
        acc_fold.append(acc)
        confusion_matrices.append(cf)
    acc_fold = vstack(acc_fold)
    return acc_fold, best_pfs, out_pf, confusion_matrices


def run_decoding_within_rois(ds, qe, partitioner, roi_ids, nproc=1):
    """Run decoding (using linear SVM) within ROIs defined by roi center
    indices specified in `roi_ids` and mapped by the queryengine `qe`.

    Parameters
    ----------
    ds : dataset
    qe : queryengine
    partitioner : partitioner
    roi_ids : array-like
        center ids (refer to ds.fa.node_indices)
    nproc : int
        numper of procs for parallel processing

    Returns
    -------
    acc_fold : dataset
        each sample corresponds to one fold, each feature to one roi
    best_pfs : list
        fraction of features used for each fold
    out_pf : list of list
        Each element of out_pf correspond to the result of the parameter
        search for each iteration.
        You should average each of these elements to obtain the accuracy
        curve whose argmax was used to select the optimal number of features.
    """
    # train first just in case
    qe.train(ds)
    out_parallel = Parallel(n_jobs=nproc)(
        delayed(run_decoding_featsel_cv)(
            ds[:, qe.query_byid(roi_id)], partitioner) for roi_id in roi_ids)
    acc_fold, best_pfs, out_pf = zip(*out_parallel)
    acc_fold = hstack(acc_fold)
    acc_fold.fa['center_ids'] = roi_ids
    return acc_fold, best_pfs, out_pf


def run_rsa_oneds(ds, meas):
    return meas(ds)


def run_rsa_within_rois(ds, target, qe, roi_ids, nproc=1):
    # train first just in case
    qe.train(ds)
    # get predictors
    target_ = 'familiar' if target.startswith('fam') else 'stranger'
    target_rdms_names = LBL2TARGETRDMS[target[3:]]
    pred = pd.read_csv(
        PREDFNS[target_], sep='\t')[target_rdms_names].as_matrix()
    # setup measure
    meas = PCorrTargetSimilarity(target_rdms=pred)
    out_parallel = Parallel(n_jobs=nproc)(
        delayed(run_rsa_oneds)(
            ds[:, qe.query_byid(roi_id)], meas)
        for roi_id in roi_ids)
    rsa = hstack(out_parallel)
    rsa.fa['center_ids'] = roi_ids
    rsa.sa['pred_names'] = target_rdms_names
    return rsa


def run_one_subj(subj, dss, targets, qes, df_rois, roi_names,
                 meas_type='decoding', maxproc=1,
                 permuter_seed=None):
    """Run the whole process for one subject

    Parameters
    ----------
    subj : str
        subject -- correspond to one of df_rois.index
    targets : list of str
        targets to run classifications or RSA for
    dss : dict of dicts
        maps fshemi ('lh' | 'rh') -> (target) -> dataset
    qes : dict
        maps fshemi ('lh' | 'rh') -> queryengine
    df_rois : pd.DataFrame
        dataframe containing ROIs
    roi_names : names of the rois (without 'lh-' or 'rh-')
    meas_type : str ('decoding' | 'rsa')
        type of multivariate measure to use
    maxproc : int
        max number of processes to use
    permuter_seed : int | None
        seed used to initialize permutator; if None, no permutation is
        performed

    Returns
    -------
    df : pd.DataFrame
        dataframe containing results; can be made in long format by doing
        `df.melt(id_vars=['subject', 'target])`
    """
    score_targ = []  # accuracies/correlations for each target
    targs = []  # target names
    preds = []  # predictor names for RSA
    for targ in targets:
        score_hemi = []
        preds_ = []
        for hemi in ['lh', 'rh']:
            print("Running {0} {1} {2}".format(subj, targ, hemi))
            # copy ds because we're going to permute it later
            ds = dss[hemi][targ].copy()
            targ_filtered = '-'.join(targ.split('-')[1:])
            targ_filtered = targ_filtered.replace('-female', '').replace('-male', '')
            if meas_type == 'decoding':
                partitioner, chunks_attr = setup_partitioner(targ_filtered)
                print("Using partitioner {}".format(partitioner.__repr__()))
                print("Using dataset {}".format(ds.summary(
                    chunks_attr=chunks_attr)))
            if permuter_seed is not None:
                if meas_type == 'decoding':
                    partitioner_attr = partitioner.attr
                    permuter = AttributePermutator('targets',
                                                   limit=[partitioner_attr,
                                                          'chunks'],
                                                   count=1, rng=permuter_seed)
                    ds = list(permuter.generate(ds))[0]
                else:
                    rng = np.random.RandomState(permuter_seed)
                    ds = ds[rng.permutation(ds.nsamples)]
            qe = qes[hemi]

            roi_names_hemi = [hemi + '-' + rn for rn in roi_names]
            n_rois = len(roi_names_hemi)
            nproc = min(n_rois, maxproc)
            roi_ids = df_rois.loc[subj][roi_names_hemi]
            # skip those with nans
            idx_roi_ids = [
                (i, r) for i, r in enumerate(roi_ids) if not np.isnan(r)]
            idx, roi_ids = zip(*idx_roi_ids)
            if meas_type == 'decoding':
                acc_fold, best_pf, out_pf = run_decoding_within_rois(
                    ds, qe, partitioner, roi_ids, nproc=nproc)
                # XXX: for now just store acc_fold
                scores = np.ones(n_rois) * np.nan
                scores[np.array(idx)] = acc_fold.samples.mean(axis=0)
            elif meas_type == 'rsa':
                rsa_corr = run_rsa_within_rois(
                    ds, targ, qe, roi_ids, nproc=nproc)
                npred = rsa_corr.shape[0]
                scores = np.ones((npred, n_rois)) * np.nan
                scores[:, np.array(idx)] = rsa_corr.samples
                if len(preds_) == 0:
                    preds_ = rsa_corr.sa.pred_names
            else:
                raise ValueError("meas_type {} not understood".format(
                    meas_type))
            score_hemi.append(scores)
        score_targ.append(np.hstack(score_hemi))
        preds.extend(preds_)
        targs += [targ] * len(preds_) if len(preds_) else [targ]
    score_targ = np.vstack(score_targ)
    # return a dataframe so we can easily put them together later
    col_names = ['lh-' + rn for rn in roi_names] +\
                ['rh-' + rn for rn in roi_names]
    df = pd.DataFrame(score_targ,
                      index=[subj] * score_targ.shape[0],
                      columns=col_names)
    if preds:
        df['predictor'] = preds
    df['target'] = targs
    df['subject'] = df.index
    return df


# We cannot run {fam,str}-orientation-{male,female} because we don't have enough
# samples for finding the optimal number of features
TARGETS_CLF = ['fam-identity', 'str-identity',
               'fam-orientation', 'str-orientation',
               'fam-gender', 'str-gender',
               'fam-identity-male', #'fam-orientation-male',
               'fam-identity-female', #'fam-orientation-female',
               'str-identity-male', #'str-orientation-male',
               'str-identity-female', #'str-orientation-female'
               ]
TARGETS_RSA = ['famid+or+mi', 'strid+or+mi']
ALL_TARGETS = TARGETS_CLF + TARGETS_RSA
HEMIS = ['lh', 'rh']
SURF_TMPL = op.join(DERIVDIR, 'freesurfer/fsaverage6/SUMA/{}.pial.gii')
HERE = op.dirname(op.abspath(__file__))
ROIS_FN = op.join(HERE, 'data', 'face_responsive_rois_max.csv')


def run(subject, meas_type='decoding', targets=ALL_TARGETS,
        surf_template=SURF_TMPL, rois_fn=ROIS_FN, permute=0, nproc=1,
        glmdir=GLMDIR):
    """This is run for one single subject only"""
    # load surfaces
    surfs = {hemi: surf.read(surf_template.format(hemi)) for hemi in HEMIS}
    qes = {
        hemi: SurfaceQueryEngine(surfs[hemi], radius=10.0) for hemi in HEMIS}
    roi_names = ['OFA', 'pFFA', 'mFFA', 'aFFA', 'ATL',
                 'pSTS', 'mSTS', 'aSTS', 'IFG']
    df_rois = pd.read_csv(rois_fn, index_col=0)

    dss = {h: dict() for h in HEMIS}
    for h in HEMIS:
        for targ in targets:
            dss[h][targ] = load_subj_ds(subject, targ, h, glmdir=glmdir)

    if permute:
        rng = np.random.RandomState(3423)
        rng_permutations = [rng.randint(MAXINT) for _ in range(permute)]
        dfs = Parallel(n_jobs=nproc)(
            delayed(run_one_subj)(subject, dss, targets, qes, df_rois,
                                  roi_names, meas_type=meas_type, maxproc=1,
                                  permuter_seed=rng_seed)
            for rng_seed in rng_permutations)
        for iperm, d in enumerate(dfs, 1):
            d['perm'] = iperm
        df = pd.concat(dfs, axis=0)
    else:
        df = run_one_subj(subject, dss, targets, qes,
                          df_rois, roi_names, meas_type=meas_type,
                          maxproc=nproc)
    return df


def main():
    p = parse_args()
    subject = p.subject if 'sub-' in p.subject else 'sub-' + p.subject
    df = run(subject, meas_type=p.measure_type, targets=p.targets,
             surf_template=p.surf_tmpl, rois_fn=p.rois_fn, permute=p.permute,
             nproc=p.nproc, glmdir=p.glmdir)
    outdir = op.join(p.output_dir, subject)
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    sfx = 'clf' if p.measure_type == 'decoding' else 'rsa'
    sfx += '_{}p'.format(p.permute) if p.permute else ''
    for targ in p.targets:
        print("Saving data for {}".format(targ))
        df_ = df[df.target == targ]
        fnout = '{0}_space-fsaverage6_targ-{1}_' \
                'roi{2}.csv'.format(subject, targ, sfx)
        df_.to_csv(op.join(outdir, fnout), na_rep='n/a')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--subject', '-s', type=str,
                        help='subject (with or without sub-)',
                        required=True)
    parser.add_argument('--measure-type', type=str,
                        help='type of measure (decoding or RSA)',
                        required=False, default='decoding',
                        choices=['decoding', 'rsa'])
    parser.add_argument('--targets', '-t', type=str, nargs='+',
                        help='targets to run classification for '
                             '(default: {})'.format(TARGETS_CLF),
                        required=False, default=TARGETS_CLF,
                        choices=ALL_TARGETS)
    parser.add_argument('--surf-tmpl', type=str,
                        help='template for surface '
                             '(default: {})'.format(SURF_TMPL),
                        required=False, default=SURF_TMPL)
    parser.add_argument('--rois-fn', type=str,
                        help='file containing centers for rois '
                             '(default: {})'.format(ROIS_FN),
                        required=False, default=ROIS_FN)
    parser.add_argument('--permute', type=int,
                        help='number of permutations, with 0 meaning no '
                             'permutations '
                             '(default: {})'.format(0),
                        required=False, default=0)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of max processors to use '
                             '(default: {})'.format(1),
                        required=False, default=1)
    parser.add_argument('--glmdir', '-g', type=str,
                        help='folder under {0} containing the glm data ('
                             'default {1})'.format(DERIVDIR, GLMDIR),
                        required=False)
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory (subfolder with the subject '
                             'data will be created)',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
