#!/usr/bin/env python
"""Script to run apply hyperalignment mappers (computed e.g. using
`run_hpal.py`) onto a list of datasets. Make sure to pass the correct mask,
as there is no way to make sure that the order of the nodes corresponds,
other than checking that the input datasets have the same dimensions as the
mapper matrix."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
from mvpa2.datasets import niml
from mvpa2.mappers.zscore import zscore
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from joblib.parallel import Parallel, delayed
import os
import os.path as op
import re
from nipype import MapNode
from nipype.interfaces.afni import ConvertDset


def convert_datasets(fns):
    convert2niml = MapNode(
        ConvertDset(),
        iterfield=['in_file', 'out_file'],
        name='convert2niml')
    convert2niml.inputs.in_file = fns
    convert2niml.inputs.out_type = 'niml'
    convert2niml.inputs.out_file = [
        op.basename(fn).replace('gii', 'niml.dset') for fn in fns]
    res = convert2niml.run()
    return res.outputs.out_file


def load_datasets(fns):
    """Load a list of datasets, skipping initial TRs as specified in skip,
    and return a stacked dataset"""
    dss = []
    for i, fn in enumerate(fns):
        ds = niml.read(fn)
        ds.sa['chunks'] = [i]
        zscore(ds)
        if 'node_indices' not in ds.fa:
            ds.fa['node_indices'] = np.arange(ds.nfeatures)
        dss.append(ds)
    return dss


def _apply_hpal_one(ds, mp):
    ds_ = mp.forward(ds)
    zscore(ds_)
    return ds_


def apply_hpal(dss, mapper, mapper_reverse=None, nproc=1):
    node_indices = dss[0].fa.node_indices
    if mapper_reverse is None:
        mp = mapper
    else:
        # we are going to compose the two mappers to reduce the number of
        # operations
        print("Got reverse mapper, combining transformations")
        mapper_proj = mapper.proj
        mapper_reverse_proj = mapper_reverse.proj.T
        one_proj = mapper_proj.dot(mapper_reverse_proj)
        mp = StaticProjectionMapper(proj=one_proj)
        print("Done combining transformations")
    dss = Parallel(n_jobs=nproc)(
        delayed(_apply_hpal_one)(ds, mp) for ds in dss)
    # re-impute node-indices after projection
    for ds in dss:
        ds.fa['node_indices'] = node_indices
    return dss


def load_mask(maskfn):
    mask = niml.read(maskfn)
    mask_idx = np.where(mask.samples[0])[0]
    return mask_idx


def parse_fn(fn):
    subj_re = re.compile(r'(sub-sid[0-9]{6})')
    space_re = re.compile(r'space-([a-zA-Z0-9]*)')
    ses_re = re.compile(r'ses-([a-zA-Z0-9]*)')

    subj = subj_re.findall(fn)[0]
    space = space_re.findall(fn)
    space = space[0] if space else ''
    ses = ses_re.findall(fn)
    ses = ses[0] if ses else ''

    return dict(subj=subj, space=space, ses=ses)


def run(fns, mask_fn, mapper_fn, outdir, reverse=None, nproc=1):
    print("Checking if extensions are compatible")
    exts = map(lambda x: x.endswith('gii'), fns)
    if any(exts):
        print("Temporarily converting datasets to NIML first")
        fns = convert_datasets(fns)
    print("Loading input datasets")
    dss = load_datasets(fns)
    print("Loading mask")
    mask_idx = load_mask(mask_fn)
    print("Applying mask")
    dss = [ds[:, mask_idx] for ds in dss]
    print("Applying hyperalignment")
    mapper = h5load(mapper_fn)
    mapper_reverse = h5load(reverse) if reverse is not None else None
    dss = apply_hpal(dss, mapper, mapper_reverse, nproc=nproc)

    # get the subject id so we store this info in the filename
    if reverse is not None:
        pr = parse_fn(reverse)
        subj_rev = pr['subj']
        subj_rev = subj_rev.replace('sub-', '')
    else:
        subj_rev = ''

    for fn, ds in zip(fns, dss):
        p = parse_fn(fn)
        subj, space, ses = p['subj'], p['space'], p['ses']
        outdir_ = op.join(outdir, subj)
        outfn = op.basename(fn)
        outfn = outfn.replace(space, 'hpal' + subj_rev + space)
        if ses:
            outdir_ = op.join(outdir_, 'ses-{}'.format(ses))
        outdir_ = op.join(outdir_, 'func')
        if not op.exists(outdir_):
            os.makedirs(outdir_)
        outfn = op.join(outdir_, outfn)
        print("Saving {}".format(outfn))
        niml.write(outfn, ds)


def main():
    p = parse_args()
    run(p.inputs, p.mask, p.mapper, p.output_dir, p.reverse, p.nproc)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--inputs', '-i', type=str, nargs='+',
                        help='input files',
                        required=True)
    parser.add_argument('--mask', '-m', type=str,
                        help='mask',
                        required=True)
    parser.add_argument('--mapper', '-p', type=str,
                        help='Hyperalignment Mapper',
                        required=True)
    parser.add_argument('--reverse', type=str,
                        help="Hyperalignment Mapper used to reverse "
                             "projection. This is for example another "
                             "subject's mapper. The resulting transformation "
                             "will project the data from one participant's "
                             "space to another participant's space, passing "
                             "through the common model space.",
                        required=False, default=None)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of procs to use for parallel processing',
                        required=False, default=1)
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory. intermediate subdirectories '
                             'will be created automatically',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
