#!/usr/bin/env python
"""Perform cross-validated RSA within each searchlight. For memory
efficiency only the average RSA across folds is stored. It splits the data
in half, averages the resulting datasets, and computes a distance between
the two halves. All the possible splits are computed, and then finally
averaged.

For efficiency, we consider the metric to be symmetric, so that we only
consider combinations of (train, test) once. This is not correct if some
parameters need to be estimated within the training set, for example for
CrossNobis or classification."""

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# we need our version of Searchlight with preallocation to speed up
# computations
import sys
sys.path.insert(1, '/idata/DBIC/castello/famface_angles/3rd/PyMVPA')

import argparse
import os
import re
import numpy as np
import pytest
from famfaceangles.mvpa import MeanCrossValidation, FisherCDist, \
    sort_samples, SymmetricSplitHalfPartitioner
from mvpa2.suite import *
from run_sl import (_setup_ds, _load_ds, _prepare_labels, get_queryengine)


MAXINT = 2**32
rng = np.random.RandomState(5414)
RNGS = [rng.randint(MAXINT) for _ in range(100)]


def run_and_save(tent_fn, metric, output_dir, ztrans=True, freesurfer_dir=None,
                 mask_fn=None, voxsel_fn=None, permute=0, nproc=1,
                 overwrite=False):
    # first check if the file already exists
    outfn = make_output(tent_fn, metric, output_dir, permute, ztrans)
    if os.path.exists(outfn) and not overwrite:
        raise OSError("File {} exists, not overwriting".format(outfn))
    ds, is_surface = _load_ds(tent_fn, mask_fn)
    roi_ids = ds.fa.node_indices if is_surface else None
    # roi_ids = roi_ids[:100]  # debug

    qe = get_queryengine(is_surface,
                         tent_fn=tent_fn,
                         voxsel_fn=voxsel_fn,
                         freesurfer_dir=freesurfer_dir)
    # setup
    ds = _setup_ds(ds)
    is_tent = ds.sa.ntent.max() > 0
    # keep only first five tent
    if is_tent:
        raise ValueError("NOT IMPLEMENTED FOR TENT GLM")
    ds.sa['chunks'] = ds.sa.run
    ds.sa['targets'] = ds.sa.labels
    ds = sort_samples(ds)
    if ztrans:
        zscore(ds)

    ds_input = ds.copy(deep=False, sa=['chunks', 'targets'])
    # ds_input = ds_input[:100]  # debug

    partitioner = SymmetricSplitHalfPartitioner()
    if permute:
        partitioner_attr = partitioner.attr
        limit = list(set(['chunks'] + [partitioner_attr]))
        partitioner = ChainNode([
            AttributePermutator('targets', count=1,
                                rng=RNGS[permute - 1],
                                limit=limit),
            partitioner], space='partitions')
    print("Using partitioner {}".format(partitioner.__repr__()))
    if metric == 'correlation':
        measure = FisherCDist()
    else:
        measure = CDist(pairwise_metric=metric)

    print("Using the following dataset as input")
    print(ds_input.summary())

    slmap = _run_sl(ds_input, qe, measure, partitioner, nproc,
                    roi_ids=roi_ids)
    slmap.sa['perm'] = [permute] * slmap.nsamples

    if metric == 'correlation':
        # map back to correlation distance
        slmap.samples = np.nan_to_num(slmap.samples)
        slmap.samples = 1. - np.tanh(slmap.samples)

    # keep only targets otherwise niml complains
    slmap = slmap.copy(deep=False, sa=['targets'])

    try:
        os.makedirs(os.path.dirname(outfn))
    except OSError:
        pass
    print("Saving to {0}".format(outfn))
    niml.write(outfn, slmap)


def _create_outfn(tent_fn, metric, ztrans):
    """Create the output fn -- for now it's fixed but who knows"""
    pattern = r'(?P<sid>sub-sid\d{6}).*task-(?P<task>\w*)_space-(' \
              r'?P<space>\w*)_(?:hemi-(?P<hemi>[LR]))?'
    match = re.search(pattern, tent_fn)
    sid = match.group('sid')
    if not sid:
        raise ValueError("Couldn't parse subject identifier from {0} -- I "
                         "can only parse DBIC subject ids such as "
                         "sub-sid000021".format(tent_fn))
    hemi = match.group('hemi')
    task = match.group('task')
    space = match.group('space')
    # ext = tent_fn.split('.', 1)[-1]
    ext = 'niml.dset'
    outfn = '{0}/{0}_task-{1}_space-{2}'.format(sid, task, space)
    if hemi:
        outfn += '_hemi-{}'.format(hemi)
    outfn += '_cvrsa_metric-{}_sl.'.format('z' + metric if ztrans else
                                           metric) + ext
    return outfn


def make_output(tent_fn, metric, output_dir, permute, ztrans):
    outfn = os.path.join(output_dir, _create_outfn(tent_fn, metric, ztrans))
    if permute:
        outfn = outfn.replace('sl.', 'sl_{0:03d}perm.'.format(permute))
    return outfn


def _run_sl(ds_input, qe, measure, partitioner, nproc, roi_ids):
    # setup cross validation
    cvte = MeanCrossValidation(
        measure,
        partitioner,
        errorfx=None)
    sl = Searchlight(cvte, queryengine=qe, nproc=nproc, roi_ids=roi_ids, preallocate_output=True)
    # increase debug level
    mvpa2.debug.active += ['SLC']
    # run it
    slmap = sl(ds_input)
    return slmap


def _run_sl_permute(ds_input, qe, measure, partitioner, permute, nproc,
                    roi_ids):
    mvpa2.debug.active += ['SLC']
    rng = np.random.RandomState(3423)
    rng_permutations = [rng.randint(MAXINT) for _ in range(permute)]

    slmap = []
    partitioner_attr = partitioner.attr
    limit = list(set(['chunks'] + [partitioner_attr]))
    for p, randomgen in enumerate(rng_permutations):
        permuter = AttributePermutator('targets',
                                       limit=limit,
                                       count=1, rng=randomgen)
        partitioner_rng = ChainNode([partitioner, permuter],
                                    space=partitioner.get_space())
        # setup cross validation
        cvte = MeanCrossValidation(
            measure,
            partitioner_rng,
            errorfx=None)
        sl = Searchlight(cvte, queryengine=qe, nproc=nproc, roi_ids=roi_ids, preallocate_output=True)
        print("Running permutation {}".format(p))
        tmp = sl(ds_input)
        tmp.sa['permutation'] = [p]
        slmap.append(tmp)
    slmap = vstack(slmap)
    return slmap


def main():
    parsed = parse_args()
    run_and_save(
        tent_fn=parsed.input,
        metric=parsed.metric,
        ztrans=parsed.no_zscore,
        permute=parsed.permute,
        freesurfer_dir=parsed.fs_dir,
        mask_fn=parsed.mask,
        voxsel_fn=parsed.voxsel,
        output_dir=parsed.output,
        nproc=parsed.nproc)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fs-dir', '-f', type=str,
                        help='freesurfer directory',
                        required=False)
    parser.add_argument('--mask', '-m', type=str,
                        help='mask file',
                        required=False)
    parser.add_argument('--voxsel', '-v', type=str,
                        help='voxel selection file',
                        required=False)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of procs to use',
                        default=1)
    parser.add_argument('--permute', '-p', type=int,
                        help='number of permutation to run (0 = disabled). '
                             'Note that differently from `run_sl.py`, '
                             'this is used to indicate WHICH permutation to '
                             'run based on a fixed random seed. In this way '
                             'we can parallelize across permutations because '
                             'this searchlight takes a loooong time '
                             '(well, relatively).',
                        default=0)
    parser.add_argument('--input', '-i', type=str,
                        help='input file',
                        required=True)
    parser.add_argument('--output', '-o', type=str,
                        help='output directory (will create sub-sid* '
                             'directory)',
                        required=True)
    parser.add_argument('--metric', type=str,
                        help='metric to use for RSA',
                        choices=['correlation', 'euclidean'],
                        required=False, default='correlation')
    parser.add_argument('--no-zscore', action='store_false',
                        help='do not zscore',
                        required=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()


