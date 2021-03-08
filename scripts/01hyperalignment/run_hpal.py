#!/usr/bin/env python
"""Script to run hyperalignment for Budapest movie. You should remember to
have enough space in /tmp (so for example by mounting /tmp in the singularity
container to a location with enough storage), as well as setting
OMP_NUM_THREADS to 1 in your environment variables, to avoid using too many
resources."""

import matplotlib
matplotlib.use('Agg')
from mvpa2 import debug
debug.active += ['SHPAL', 'SLC']
import argparse
import numpy as np
from mvpa2.datasets import niml, vstack
from mvpa2.mappers.zscore import zscore
from mvpa2.base.hdf5 import h5save
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.algorithms.searchlight_hyperalignment import \
    SearchlightHyperalignment
from mvpa2.support.nibabel import surf
import os
import os.path as op
from glob import glob
from dateutil.parser import parse
from joblib.parallel import Parallel, delayed


def get_trs_toskip():
    # Get additional times to skip based on the overlap of each run
    # This comes from the bash script used for splitting
    timings = """
    00:45:22.899  00:55:00.299
    00:54:50  01:02:47.133
    01:02:27  01:11:01.599
    01:10:50  01:20:47.699
    01:20:37  01:33:39.533
    """.split()
    timings = map(parse, timings)
    fixation_s = 10

    skip = []
    for i in range(1, 9, 2):
        skip.append(fixation_s + (timings[i] - timings[i+1]).seconds)
    skip = [0] + skip
    return skip


def load_datasets(fns, skip):
    """Load a list of datasets, skipping initial TRs as specified in skip,
    and return a stacked dataset"""
    if len(fns) != len(skip):
        raise ValueError("Please provide the same number of fns and skip")
    dss = []
    for i, (fn, sk) in enumerate(zip(fns, skip)):
        ds = niml.read(fn)
        ds.sa['chunks'] = [i]
        # remove first 10 seconds + overlap
        ds = ds[sk:]
        zscore(ds)
        ds.fa['node_indices'] = np.arange(ds.nfeatures)
        dss.append(ds)
    dss = vstack(dss)
    return dss


def hyperalign(dss, qe, **hpalkwargs):
    hpal = SearchlightHyperalignment(queryengine=qe, **hpalkwargs)
    mappers = hpal(dss)
    return mappers


def load_mask(hemi, fsdir):
    mask_fn = '{}.maskmedial.niml.dset'.format(hemi)
    mask = niml.read(op.join(fsdir, 'fsaverage6/SUMA', mask_fn))
    return mask


def get_qe(hemi, fsdir, radius=20.0):
    surf_fn = '{}.white.gii'.format(hemi)
    s = surf.read(op.join(fsdir, 'fsaverage6/SUMA', surf_fn))
    qe = SurfaceQueryEngine(s, radius=radius)
    return qe


def run(list_fns, subjects, hemi, fsdir, outdir, nproc=1):
    hemi2fshemi = {'L': 'lh', 'R': 'rh'}
    fshemi = hemi2fshemi[hemi]
    skip = get_trs_toskip()
    print("Got the following datasets: {}".format(list_fns))
    print("Loading datasets")
    dss = Parallel(n_jobs=nproc)(delayed(load_datasets)(fns, skip) for fns in list_fns)
    mask = load_mask(fshemi, fsdir)
    # mask_idx = np.where(mask.samples[0])[0][:100]  # debug
    mask_idx = np.where(mask.samples[0])[0]
    # mask datasets
    print("Masking datasets")
    dss = [ds[:, mask_idx] for ds in dss]
    qe = get_qe(fshemi, fsdir)

    kwargs = dict(
        ref_ds=0, nproc=nproc, featsel=1.0,
        mask_node_ids=np.unique(dss[0].fa.node_indices),
        compute_recon=False,
    )
    print("Running Hyperalignment")
    mappers = hyperalign(dss, qe, **kwargs)

    out_fn = '{0}_task-movie_space-fsaverage6_hemi-{1}_target-hpal_mapper.h5'
    for subj, mapper in zip(subjects, mappers):
        outdir_ = op.join(outdir, subj)
        if not op.exists(outdir_):
            os.makedirs(outdir_)
        out_fn_ = op.join(outdir_,
                          out_fn.format(subj, hemi))
        print("Saving {}".format(out_fn_))
        h5save(out_fn_, mapper)


def main():
    p = parse_args()
    subjects = sorted(map(op.basename, glob(op.join(p.input_dir, 'sub-*'))))
    list_fns = [
        sorted(glob(op.join(
            p.input_dir, subj, '*{}.func*.niml.dset').format(p.hemi)))
        for subj in subjects
    ]
    run(list_fns, subjects, p.hemi, p.fsdir, p.output_dir, p.nproc)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-dir', '-i', type=str,
                        help='input directory containing subdirectories '
                             'with data to train hyperalignment on for ' 
                             'each participant',
                        required=True)
    parser.add_argument('--hemi', '-m', type=str,
                        help='which hemisphere to run hpal on',
                        required=True, choices=['L', 'R'])
    parser.add_argument('--fsdir', '-f', type=str,
                        help='freesurfer directory',
                        required=True)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of procs to use for hpal',
                        required=True)
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory. mappers will be stored '
                             'within one folder for each subject',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
