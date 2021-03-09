#!/usr/bin/env python
"""Script to run searchlight between-subject decoding"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import argparse
import numpy as np
from mvpa2.suite import *
from run_sl import (_setup_ds, _make_eventrelated, get_queryengine,
                    _filter_targets, _load_ds, MAXINT)
import os.path as op
# FS_DIR = '/idata/DBIC/castello/famface_angles/derivatives103-budapest/freesurfer'
HERE = op.dirname(__file__)
FS_DIR = op.abspath(op.join(HERE, '../../derivatives103-budapest/freesurfer'))

rng = np.random.RandomState(534)
RNGS = [rng.randint(MAXINT) for _ in range(100)]
mgs = mean_group_sample(['targets'])


def load_files(inputs, target):
    dss = []
    for i, tfn in enumerate(inputs):
        print("Loading {}".format(op.basename(tfn)))
        ds, is_surface = _load_ds(tfn)
        # setup
        ds = _setup_ds(ds)
        ds, targets_filtered = _filter_targets(ds, target)
        # keep only first five tent
        ds = ds[ds.sa.ntent < 5]
        ds = _make_eventrelated(ds)
        zscore(ds)
        # average across runs
        ds = mgs(ds)
        ds.sa['subject'] = [i]
        dss.append(ds)
    dss = vstack(dss)
    return dss


def setup_partitioner(ds, target):
    if 'identity' in target:
        chunk_attr = 'orientation'
        target_attr = 'identity'
    elif 'orientation' in target:
        chunk_attr = 'identity'
        target_attr = 'orientation'
    else:
        raise ValueError("Shouldn't get here")
    partitioner = ChainNode(
        [NFoldPartitioner(attr='subject'),
         ExcludeTargetsCombinationsPartitioner(
             k=1, targets_attr=chunk_attr, space='partitions')],
        space='partitions')
    ds.sa['targets'] = ds.sa[target_attr]
    return partitioner


def run(inputs, target, permute, freesurfer_dir=FS_DIR, nproc=1):
    dss = load_files(inputs, target)
    print("Got dataset of shape ({0}, {1})".format(dss.nsamples, dss.nfeatures))
    qe = get_queryengine(1, tent_fn=inputs[0], freesurfer_dir=freesurfer_dir)
    roi_ids = np.unique(dss.fa.node_indices)
    # roi_ids = np.unique(dss.fa.node_indices)[:100]  # debug 

    partitioner = setup_partitioner(dss, target)
    if permute:
        partitioner = ChainNode([
            AttributePermutator('targets', count=1,
                                rng=RNGS[permute - 1],
                                limit=['subject', partitioner[1].targets_attr]),
            partitioner], space='partitions')
    print("Using partitioner {}".format(partitioner))
    ds_input = dss.copy(
        sa=['identity', 'orientation', 'familiarity', 'targets', 'subject'])

    baseclf = LinearCSVMC()
    cvte = CrossValidation(
        baseclf,
        partitioner,
        errorfx=mean_match_accuracy)
    sl = Searchlight(cvte, queryengine=qe, nproc=nproc, roi_ids=roi_ids)
    # increase debug level
    mvpa2.debug.active += ['SLC']
    # run it
    slmap = sl(ds_input)
    # store perm attribute just in case
    slmap.sa['perm'] = [permute] * slmap.nsamples
    return slmap


def main():
    p = parse_args()
    validate_input(p)

    out = run(p.inputs, p.target, p.permute,
              freesurfer_dir=p.fs_dir, nproc=p.nproc)
    # check if the output directory exists
    outdir = op.dirname(op.abspath(p.output))
    if not op.exists(outdir):
        os.makedirs(outdir)
    print("Saving to {}".format(p.output))
    niml.write(p.output, out)


def validate_input(parsed):
    if parsed.permute > len(RNGS):
        raise ValueError("permute should be between 0 and {}".format(len(
            RNGS)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--inputs', '-i', type=str,
                        nargs='+',
                        help='input files',
                        required=True)
    parser.add_argument('--fs-dir', '-f', type=str,
                        help='freesurfer directory',
                        required=False,
                        default=FS_DIR)
    parser.add_argument('--target', '-t', type=str,
                        help='target for decoding',
                        choices=['fam-identity', 'str-identity',
                                 'fam-orientation', 'str-orientation'])
    parser.add_argument('--permute', '-p', type=int,
                        help='number of permutation to run (0 = disabled). '
                             'Note that differently from `run_sl.py`, '
                             'this is used to indicate WHICH permutation to '
                             'run based on a fixed random seed. In this way '
                             'we can parallelize across permutations because '
                             'this searchlight takes a loooong time '
                             '(well, relatively).',
                        default=0)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of procs to use',
                        default=1)
    parser.add_argument('--output', '-o', type=str,
                        help='output file',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
