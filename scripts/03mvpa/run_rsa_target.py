#!/usr/bin/env python
"""Script to run partial correlation of arbitrary data and target RDMs"""
import argparse
import numpy as np
import os
import os.path as op
import pandas as pd
from mvpa2.datasets import niml, Dataset
from famfaceangles.mvpa import pcorr, corr


def make_symmetric(ds, correlation=True, diagonal=False):
    """
    Given a dataset containing an RDM for each feature, make it symmetric.

    Parameters
    ----------
    ds : Dataset (n_samples, n_features)
       dataset containing the RDM
    correlation : bool
        whether the dataset contains correlation distances. If True, then the
        distances will be converted back to correlation, Fisher-transformed
        prior to averaging, and converted back afterwards.
    diagonal : bool
        whether to keep the diagonal

    Returns
    -------
    ds : Dataset (n_samples*(n_samples-1)/2 + d, n_features)
         with d = n_samples if diagonal is True, otherwise d = 0

        the upper triangular portion of the matrix
    """
    # Convert back to Fisher-transformed z-values
    if correlation:
        ds.samples = np.arctanh(1. - ds.samples)
        ds.samples = np.nan_to_num(ds.samples)
    nsamples = ds.nsamples
    nstim = int(np.sqrt(nsamples))

    # average with the transpose
    idx = np.arange(nsamples).reshape((nstim, nstim))
    ds.samples += ds.samples[idx.T.flatten()]
    ds.samples /= 2.

    if correlation:
        ds.samples = 1. - np.tanh(ds.samples)

    # take only triu, discard diagonal if needed
    k = 0 if diagonal else 1
    triu = idx[np.triu_indices_from(idx, k=k)]
    ds = ds[triu]
    return ds


def load_predictors(predictors):
    seps = ['\t' if p.endswith('tsv') else ',' for p in predictors]
    preds = []
    for p, s in zip(predictors, seps):
        preds.append(pd.read_csv(p, sep=s))
    preds = pd.concat(preds, axis=1)
    return preds


def apply_rsa(ds_fn, predictors, output, symmetric=True):
    # we are assuming input surfaces
    print("Loading dataset")
    ds = niml.read(ds_fn)

    if symmetric:
        ds = make_symmetric(ds)
    preds = load_predictors(predictors)

    # run correlation
    p = preds.as_matrix()
    if p.shape[1] > 1:
        print("Running partial correlation")
        rdm = pcorr(ds.samples, preds.as_matrix())
    else:
        print("Running correlation")
        rdm = corr(ds.samples, preds.as_matrix())
        rdm = rdm[None]

    # save dataset
    rdm = Dataset(rdm, fa=ds.fa, a=ds.a)
    rdm.sa['labels'] = preds.columns

    output = op.abspath(output)
    outdir = op.dirname(output)
    if not op.exists(outdir):
        os.makedirs(outdir)
    print("Saving to {}".format(output))
    niml.write(output, rdm)


def main():
    p = parse_args()
    apply_rsa(p.ds_rsa, p.predictors, p.output, symmetric=p.no_symmetric)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--ds-rsa', type=str,
                        help='File containing neural RDMs. It is assumed '
                             'that each columns of the dataset is a feature '
                             '(voxel, center node, ROI), and each sample is '
                             'a pairwise distance between stimuli.',
                        required=True)
    parser.add_argument('--predictors', type=str, nargs='+',
                        help='TSV files containing the target RDM. Each row '
                             'is a pairwise distance between stimuli ('
                             'corresponding to the upper triangular portion '
                             'of the matrix). These are assumed to be '
                             'distances, so smaller values indicate higher '
                             'similarity. Multiple files can be passed, '
                             'and the resulting matrices will be '
                             'concatenated column-wise. The output dataset '
                             'will have as many samples as the number of '
                             'predictors.',
                        required=True)
    parser.add_argument('--no-symmetric', action="store_false",
                        help='By default the input dataset will be made '
                             'symmetric, as required when a cross-validated '
                             'RSA is passed. Use this flag to disable this '
                             'operation; however, the predictor files need '
                             'to have the correct number of row',
                        required=False)
    parser.add_argument('--output', '-o', type=str,
                        help='output dataset',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
