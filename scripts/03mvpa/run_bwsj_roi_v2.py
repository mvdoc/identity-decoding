#!/usr/bin/env python
"""Script to run bwsj identity decoding within face responsive and extended
system ROIs. It will run the unpermuted and permuted analysis for all ROIs."""

from collections import defaultdict
from mvpa2.suite import *
import numpy as np
import os.path as op
import os
from glob import glob
import joblib
from run_sl_bwsbj import load_files, setup_partitioner, RNGS
from run_sl_roi import run_decoding_featsel_cv_cf, TARGETS_CLF, ALL_TARGETS
HERE = op.dirname(op.abspath(__file__))

ROIs = ['OFA', 'pFFA', 'mFFA', 'aFFA', 'ATL', 'pSTS', 
        'mSTS', 'aSTS', 'IFG', 'MPFC', 'Precuneus', 
        'TPJ', 'Insula']


def load_mask_medial(infiles):
    hemi = 'lh' if 'hemi-L' in infiles[0] else 'rh'
    mask_medial = niml.read(op.join(
        HERE, '../../derivatives/freesurfer/fsaverage6/SUMA/{}.'
              'maskmedial.niml.dset'.format(hemi)))
    mask_medial = mask_medial.samples[0] == 1
    return mask_medial


def run_decoding_simple(ds, partitioner, clf=LinearCSVMC(), nproc=1):
    """Run vanilla decoding without fancy feature selection"""
    cvte = CrossValidation(clf, partitioner, errorfx=mean_match_accuracy)
    return cvte(ds)


def run(infiles, target, ds_roi_file, roi='all', n_permute=50, nproc=1,
        run_featsel=True):
    # Load dataset with ROIs. We expect one sample for each ROI.
    ds_roi = niml.read(op.abspath(ds_roi_file))
    n_rois = ds_roi.nsamples
    print("Got {0} rois: {1}".format(n_rois, ds_roi.sa.labels))
    # We need to mask the medial wall because such are the input datasets
    mask_medial = load_mask_medial(infiles)
    mask_roi = ds_roi[:, mask_medial].samples == 1.
    roi_labels = ds_roi.sa.labels
    if roi.lower() != 'all':
        print("Selecting ROI {}".format(roi))
        which_roi = [roi in lbl for lbl in roi_labels]
        mask_roi = mask_roi[which_roi]
        roi_labels = roi_labels[which_roi]
    # Load dataset with brain data
    dss = load_files(infiles, target)
    # Do we need to use the fancy feature selection?
    if run_featsel:
        print("Running with feature selection")
        decoding_fx = run_decoding_featsel_cv_cf
    else:
        print("Running without feature selection")
        decoding_fx = run_decoding_simple
    # Now it's just a double for loop across ROIs and permutations
    results = defaultdict(list)
    for roi, mask in zip(roi_labels, mask_roi):
        print("Running for {}".format(roi))
        ds_input = dss[:, mask]
        ds_input = ds_input.copy(
            sa=['identity', 'orientation', 'familiarity', 'targets',
                'subject'])
        # now loop across permutations
        for permute in range(n_permute + 1):
            # Setup partitioner
            partitioner = setup_partitioner(ds_input, target=target)
            print("  running permutation {0}/{1}".format(permute, n_permute))
            # Change partitioner if we are permuting
            if permute:
                partitioner = ChainNode([
                    AttributePermutator(
                        'targets', count=1, rng=RNGS[permute - 1],
                        limit=['subject', partitioner[1].targets_attr]),
                    partitioner
                ], space='partitions')
            out = decoding_fx(ds_input, partitioner, nproc=nproc)
            results[roi].append(out)
    return dict(results)


def main():
    p = parse_args()
    print("Running")
    results = run(p.input, p.target, p.ds_roi_file, p.roi, p.n_permute,
                  p.nproc, p.no_feature_selection)
    print("Saving to {}".format(p.output))
    # h5save(p.output, results)
    joblib.dump(results, p.output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input', '-i', type=str,
                        help='input file',
                        nargs='+',
                        required=True)
    parser.add_argument('--target', '-t', type=str,
                        help='target to run classification for '
                             '(default: {})'.format(TARGETS_CLF),
                        required=False, default=TARGETS_CLF,
                        choices=ALL_TARGETS)
    parser.add_argument('--ds_roi_file', '-d', type=str,
                        help='niml dataset with one sample for each ROI. It '
                             'will be used to mask the input datasets for '
                             'each ROI.', required=True)
    parser.add_argument('--roi', '-r', type=str,
                        help='Whether to run for a single ROI. default "all"',
                        choices=['all'] + ROIs,
                        required=False, default='all')
    parser.add_argument('--no-feature-selection', 
                        action='store_false', default=True,
                        help='Do not run with the fancy feature selection; '
                             'use all features in each ROI')
    parser.add_argument('--n_permute', '-p', type=int,
                        help='number of permutations to run. All '
                             'permutations will be saved to the same file',
                        default=50)
    parser.add_argument('--output', '-o', type=str,
                        help='output h5 file',
                        required=True)
    parser.add_argument('--nproc', '-n', type=int,
                        help='number of parallel processes',
                        default=1)
    return parser.parse_args()


if __name__ == '__main__':
    main()
