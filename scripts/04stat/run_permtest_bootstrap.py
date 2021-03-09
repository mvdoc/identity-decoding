#!/usr/bin/env python
"""Script to run bootstrapping on permuted maps. It returns the
uncorrected **one-tailed** p-vals computed from bootstrapping null maps,
as well as FDR corrected p-vals"""
import argparse
from mvpa2.suite import *
import numpy as np
import os.path as op
import os
from glob import glob
from statsmodels.stats.multitest import multipletests
from joblib.parallel import Parallel, delayed


def _load_ds_block(fns):
    dss = np.dstack(
        [niml.read(fn).samples for fn in fns])
    return dss


def load_dss(fns, nproc=1):
    """Load a bunch of datasets in parallel."""
    fns_split = np.array_split(fns, nproc)
    dss = Parallel(n_jobs=nproc)(
        delayed(_load_ds_block)(fns_) for fns_ in fns_split)
    dss = np.dstack(dss)
    return dss


def generate_rng(nrngs, startseed=None):
    """Generator for random states"""
    start_rng = np.random.RandomState(startseed)
    for i in range(nrngs):
        yield np.random.RandomState(start_rng.randint(2**32))


def _count_block_perm(orig_vals, perm_ds, nbs, tail, rng):
    """
    Count the number of times a permuted null map is greater or equal than
    the original value. The permuted map is generated by bootstrapping
    the permuted maps (at every iteration, one for every observation,
    and then averaged). This is done element-by-element of orig_vals.

    Parameters
    ----------
    orig_vals : array-like (n_features,)
        original values
    perm_ds : array-like (n_observations, n_features, n_permutations)
        permuted maps
    nbs : int
        number of bootstraps
    tail : -1, 0, 1
        whether to run a one-tailed (1: positive, -1: negative) or
        two-tailed test (0)
    rng : np.RandomState instance

    Returns
    -------
    count_vals : array (n_features,)
        contains the number of times a permuted map bootstrapped from
        perm_ds is greater or equal than orig_vals (if tail == 1), or less
        or equal than orig_vals (if tail == -1), or both (if tail == 0).
        The p-value for orig_vals[i] can be computed as
        (count_vals[i] + 1)/(nbs + 1).
    """
    orig_vals = np.asarray(orig_vals)
    if orig_vals.ndim > 1:
        raise ValueError("orig_vals should be a one-dimensional array")
    count_vals = np.zeros(orig_vals.shape)
    for ibs in range(nbs):
        randidx = rng.choice(np.arange(perm_ds.shape[-1]),
                             size=perm_ds.shape[0])
        perm_vals = np.mean(
            [perm_ds[i, :, idx] for i, idx in enumerate(randidx)], axis=0)
        if tail == 1:
            count_vals += (perm_vals >= orig_vals).astype(int)
        elif tail == -1:
            count_vals += (perm_vals <= orig_vals).astype(int)
        else:
            count_vals += (np.abs(perm_vals) >= np.abs(orig_vals)).astype(int)
    return count_vals


def compute_perm_pvals(orig_vals, perm_ds, nbs=10000, nblocks=10, nproc=1,
                       tail=1, seed=53534):
    """Compute pvals for orig_vals by bootstrapping datasets in perm_ds
    Arguments
    ---------
    orig_vals : array-like (n_features,)
        original values
    perm_ds : array-like (n_observations, n_features, n_permutations)
        permuted maps
    nbs : int
        number of bootstraps (default 10000)
    nblocks : int
        number of blocks (make sure that nblocks divides nbs)
    nproc : int
        number of processors
    tail : -1, 0, 1
        whether to run a one-tailed (1: positive, -1: negative) or
        two-tailed test (0)
    seed : int
        seed for random generation of bootstraps

    Returns
    -------
    pvals : array (n_features,)
        permuted pvalues. These are computed according to the biased
        estimator p = (B+1/n+1), see [1]

    [1] Phipson, B., & Smyth, G. K. (2010). Permutation P-values should
    never be zero: calculating
        exact P-values when permutations are randomly drawn. Statistical
        Applications in Genetics and
        Molecular Biology, 9, Article39. https://doi.org/10.2202/1544-6115.1585
    """
    if nbs % nblocks != 0:
        raise ValueError("nblocks needs to evenly divide nbs")

    nbs_block = nbs / nblocks
    count_vals = Parallel(n_jobs=nproc, verbose=50)(
        delayed(_count_block_perm)(orig_vals, perm_ds, nbs_block, tail, rng)
        for rng in generate_rng(nblocks, startseed=seed))

    count_vals = np.sum(count_vals, axis=0) + 1
    count_vals /= float(nbs + 1)
    return count_vals


def main():
    p = parse_args()
    nproc = p.nproc
    # load as a dataset because we need the feature attributes
    orig_ds = niml.read(p.input)
    node_indices = orig_ds.fa.node_indices
    orig_ds_avg = orig_ds.samples.mean(axis=0)
    print("Got {} permuted maps".format(len(p.permuted)))
    perm_ds = load_dss(p.permuted, nproc)
    # add the original ds to the permutations as well
    # perm_ds = np.dstack((orig_ds.samples, perm_ds))
    # check if we need to use a different number of blocks
    rem_nproc = p.nbootstraps % p.nproc
    nblocks = p.nproc + rem_nproc
    print("Starting computation of p-values")
    pvals = compute_perm_pvals(orig_ds_avg, perm_ds,
                               nbs=p.nbootstraps,
                               nblocks=nblocks,
                               nproc=p.nproc,
                               tail=p.tail)
    print("Done computing p-values")
    _, pvals_corrected, _, _ = multipletests(pvals, p.alpha, method='fdr_bh')

    outdir = op.dirname(p.prefix)
    if not op.exists(outdir):
        os.makedirs(outdir)
    prfx = op.basename(p.prefix)
    prfx = prfx.replace('.niml.dset', '')

    tosave = {
        'avg': orig_ds_avg,
        'pval': -np.log10(pvals),
        'fdrpval': -np.log10(pvals_corrected),
        'avgthr': orig_ds_avg * (pvals < p.alpha).astype(float),
        'avgfdrthr': orig_ds_avg * (pvals_corrected < p.alpha).astype(float),

    }

    for sfx, data in tosave.iteritems():
        tmp = Dataset(data[None], fa={'node_indices': node_indices})
        outfn = op.join(outdir, '{0}_{1}.niml.dset'.format(prfx, sfx))
        print("Saving {}".format(outfn))
        niml.write(outfn, tmp)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input', '-i', type=str,
                        help='original (unpermuted) dataset',
                        required=True)
    parser.add_argument('--permuted', '-p', type=str, nargs='+',
                        help='permuted maps (these will be sampled to '
                             'generate null distributions for each node)',
                        required=True)
    parser.add_argument('--nbootstraps', '-b', type=int,
                        help='number of bootstraps (default 100,000)',
                        required=False, default=10**5)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of processors for parallel computation '
                             '(default 1)',
                        required=False, default=1)
    parser.add_argument('--alpha', '-a',  type=float,
                        help='alpha value (default=0.05)',
                        required=False, default=0.05)
    parser.add_argument('--tail', '-t',  type=int,
                        help='one-side, positive (1); one-side, negative ('
                             '-1); two-sided (0) (default=1)',
                        required=False, default=1)
    parser.add_argument('--prefix',  type=str,
                        help='output file. We will create the output '
                             'directories if necessary. The prefix will be '
                             'used to generate four files ('
                             'original map unthresholded, uncorrected pvals, '
                             'FDR-corrected pvals, original map thresholded '
                             'according to uncorrected pvals). The pvals are '
                             'modified by applying -log10 '
                             'for easier visualization',
                        required=True)
    return parser.parse_args()


def test_compute_perm_pvals():
    n_nodes = 100
    n_subjects = 20

    orig = np.random.randn(n_subjects, n_nodes)
    # fake activation
    act_start = 10
    act_end = 40
    orig[:, act_start:act_end] += 10
    # now create a bunch of null maps
    nperms = 100
    # add original maps as zeroth permutation
    perm_ds = [orig] + [np.random.randn(n_subjects, n_nodes)
                        for _ in range(nperms-1)]
    perm_ds = np.dstack(perm_ds)

    # now compute the bootstrapped pvals
    nbs = 1000
    ps = compute_perm_pvals(orig.mean(axis=0), perm_ds, nbs,
                            nblocks=1, nproc=1)
    # apply bh FDR correction to clean it up
    _, ps_c, _, _ = multipletests(ps, method='fdr_bh')
    # we could have 5% false positives
    assert (ps_c < 0.05).sum() <= (act_end - act_start) + 0.05*n_nodes
    # stronger test
    assert set(np.where(ps_c < 0.05)[0]).issuperset(
        np.arange(act_start, act_end))


def test_compute_perm_pvals_twotailed():
    n_nodes = 100
    n_subjects = 20

    orig = np.random.randn(n_subjects, n_nodes)
    # fake activation
    act_start = 10
    act_end = 40
    act_middle = act_end // 2
    orig[:, act_start:act_middle] += 10
    orig[:, act_middle:act_end] -= 10
    # now create a bunch of null maps
    nperms = 100
    # add original maps as zeroth permutation
    perm_ds = [orig] + [np.random.randn(n_subjects, n_nodes)
                        for _ in range(nperms-1)]
    perm_ds = np.dstack(perm_ds)

    # now compute the bootstrapped pvals
    nbs = 1000
    ps = compute_perm_pvals(orig.mean(axis=0), perm_ds, nbs, tail=0,
                            nblocks=1, nproc=1)
    # apply bh FDR correction to clean it up
    _, ps_c, _, _ = multipletests(ps, method='fdr_bh')
    # we could have 5% false positives
    assert (ps_c < 0.05).sum() <= (act_end - act_start) + 0.05*n_nodes
    # stronger test
    assert set(np.where(ps_c < 0.05)[0]).issuperset(
        np.arange(act_start, act_end))


if __name__ == '__main__':
    main()


