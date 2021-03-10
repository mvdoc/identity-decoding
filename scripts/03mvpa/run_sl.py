#!/usr/bin/env python
"""Run searchlight classification on the surface. If the input is a surface
(.gii), then it performs surface searchlight. If the input is a volume,
then a voxel selection file must be passed, and it performs surface to
volume searchlight."""

import argparse
import os
import re
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from mvpa2.suite import *
from sklearn.preprocessing import LabelEncoder
MAXINT = 2**32


def _load_surface(tent_fn, freesurfer_dir, which='pial', fsaverage='fsaverage6'):
    """Automatically load pial surface given infile"""
    hemi = re.findall('hemi-([LR])', tent_fn)[0]
    pial_fn = '{0}h.{1}.gii'.format(hemi.lower(), which)
    pial_fn = os.path.join(freesurfer_dir, fsaverage, 'SUMA', pial_fn)
    print("Loading {}".format(pial_fn))
    return surf.read(pial_fn)


def _setup_ds(ds):
    """Performs setup of dataset. Returns a copy"""
    ds_ = ds.copy()
    # take only Tstat samples
    ds_ = ds_[['Tstat' in lbl and 'GLT' not in lbl for lbl in ds.sa.labels]]
    # rename labels without the Tstat
    m = re.compile(r'((?:str_|fam_)?id\d_a\d)')
    # get only stimuli
    mask = map(lambda x: True if m.search(x) else False, ds_.sa.labels)
    ds_ = ds_[mask]
    labels = ds_.sa.labels
    ds_.sa['labels'] = map(lambda x: m.findall(x)[0], labels)
    # add also tent number
    m = re.compile(r'#([0-9]*)')
    ds_.sa['ntent'] = map(lambda x: int(m.findall(x)[0]), labels)
    # and also add run if present
    if 'run' in labels[0]:
        m = re.compile(r'run([0-9]*)')
        runs = map(lambda x: int(m.findall(x)[0]), labels)
        ds_.sa['run'] = runs
    # add node_indices
    if 'node_indices' not in ds_.fa:
        ds_.fa['node_indices'] = np.arange(ds_.nfeatures, dtype=int)
    return ds_


def _make_eventrelated(ds):
    """Makes event-related dataset to perform classification 
    on multiple features for each node"""
    events = find_events(targets=ds.sa.labels, chunks=[1]*ds.nsamples)
    ds_evs = eventrelated_dataset(ds, events=events)
    ds_evs = _prepare_labels(ds_evs)
    return ds_evs


def _prepare_labels(ds_evs):
    # add unique labels
    ds_evs.sa['labels'] = [np.unique(l)[0] for l in ds_evs.sa.labels]
    if 'familiarity' in ds_evs.sa:
        ds_evs.sa['familiarity'] = ds_evs.sa.familiarity[:, 0]
    # split into identity and orientation
    ds_evs.sa['identity'], ds_evs.sa['orientation'] = zip(
        *map(lambda x: x.split('_'), ds_evs.sa.labels))
    # convert labels into ints for SVC
    le = LabelEncoder()
    ds_evs.sa['orientation'] = le.fit_transform(ds_evs.sa.orientation)
    ds_evs.sa['identity'] = le.fit_transform(ds_evs.sa.identity)
    # add gender -- first two ids are always female
    ds_evs.sa['gender'] = [0 if l.startswith(('id1', 'id2')) else 1
                           for l in ds_evs.sa.labels]
    return ds_evs


def _create_outfn(tent_fn, targets):
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
    outfn += '_target-{}_sl.'.format(targets) + ext
    return outfn


def get_queryengine(is_surface, tent_fn, voxsel_fn=None, freesurfer_dir=None):
    if is_surface:
        # load surface
        pial = _load_surface(tent_fn, freesurfer_dir=freesurfer_dir)
        # create query engine
        qe = SurfaceQueryEngine(pial, radius=10.0)
    else:
        voxsel = h5load(voxsel_fn)
        qe = SurfaceVerticesQueryEngine(voxsel)
    return qe


def _filter_targets(ds, targets):
    """Filters the dataset with familiar or stranger identities and updates
    labels, as well as the targets information"""
    labels = ds.sa.labels
    mask = []
    if targets.startswith('str-'):
        mask = map(lambda x: x.startswith('str'), labels)
        labels = map(lambda x: x.replace('str_', ''), labels[mask])
        targets = targets.replace('str-', '')
    elif targets.startswith('fam-'):
        mask = map(lambda x: x.startswith('fam'), labels)
        labels = map(lambda x: x.replace('fam_', ''), labels[mask])
        targets = targets.replace('fam-', '')
    elif targets == 'familiarity':
        ds.sa['familiarity'] = [1 if l.startswith('fam') else 0 for l in labels]
        labels = map(lambda x: x.replace('fam_', '').replace('str_', ''),
                     labels)
    if mask:
        ds = ds[mask]
    ds.sa['labels'] = labels
    return ds, targets


def run_and_save(tent_fn, targets, output_dir, freesurfer_dir=None,
                 mask_fn=None, voxsel_fn=None, permute=0, nproc=1,
                 overwrite=False):
    # first check if the file already exists
    outfn = make_output(tent_fn, targets, output_dir, permute)
    if os.path.exists(outfn) and not overwrite:
        raise OSError("File {} exists, not overwriting".format(outfn))
    ds, is_surface = _load_ds(tent_fn, mask_fn)
    roi_ids = ds.fa.node_indices if is_surface else None

    qe = get_queryengine(is_surface,
                         tent_fn=tent_fn,
                         voxsel_fn=voxsel_fn,
                         freesurfer_dir=freesurfer_dir)
    # setup
    ds = _setup_ds(ds)
    ds, targets_filtered = _filter_targets(ds, targets)
    is_tent = ds.sa.ntent.max() > 0
    # keep only first five tent
    if is_tent:
        ds = ds[ds.sa.ntent < 5]
        ds = _make_eventrelated(ds)
        # check number of samples
        nsamples = 40 if targets == 'familiarity' else 20
        assert ds.nsamples == nsamples
        assert len(np.unique(ds.sa.labels)) == nsamples
    else:
        ds = _prepare_labels(ds)
        if 'run' in ds.sa:
            ds.sa['chunks'] = ds.sa.run

    # if we are classifying within male/female, filter out
    if targets_filtered.endswith('female'):
        ds = ds[ds.sa.gender == 0]
        targets_filtered = targets_filtered.replace('-female', '')
    elif targets_filtered.endswith('male'):
        ds = ds[ds.sa.gender == 1]
        targets_filtered = targets_filtered.replace('-male', '')

    if 'run' in ds.sa:
        # We need to zscore within each run if we are computing betas
        # within each run. Otherwise we are not going to zscore.
        zscore(ds)
    ds_input = ds.copy(sa=[
        'identity', 'orientation', 'familiarity', 'chunks', 'gender'
    ])

    partitioner, chunks_attr = setup_partitioner(targets_filtered)
    print("Using partitioner {}".format(partitioner.__repr__()))

    targets_filtered = targets_filtered.replace('-run', '')
    ds_input.sa['targets'] = ds_input.sa[targets_filtered]
    print("Using the following dataset as input")
    print(ds_input.summary(chunks_attr=chunks_attr))

    if permute:
        if targets == 'familiarity' or 'run' in targets:
            raise NotImplementedError("Still need to implement permutation "
                                      "for {}".format(targets))
        slmap = _run_sl_permute(ds_input, qe, partitioner, permute, nproc,
                                roi_ids=roi_ids)
    else:
        slmap = _run_sl(ds_input, qe, partitioner, nproc, roi_ids=roi_ids)

    try:
        os.makedirs(os.path.dirname(outfn))
    except OSError:
        pass
    print("Saving to {0}".format(outfn))
    niml.write(outfn, slmap)


def setup_partitioner(targets_filtered):
    # create classification problem
    if targets_filtered.startswith(('identity', 'gender')):
        chunks_attr = 'orientation'
    elif targets_filtered.startswith(('orientation', 'familiarity')):
        chunks_attr = 'identity'
    else:
        chunks_attr = 'chunks'
    partitioner = _get_partitioner(targets_filtered, chunks_attr)
    return partitioner, chunks_attr


def make_output(tent_fn, targets, output_dir, permute):
    outfn = os.path.join(output_dir, _create_outfn(tent_fn, targets))
    if permute:
        outfn = outfn.replace('sl.', 'sl_{}perm.'.format(permute))
    return outfn


def _get_partitioner(targets, chunks_attr):
    """Return a partitioner given the targets label"""
    partitioner = None
    if targets in ['identity', 'orientation', 'familiarity', 'gender']:
        partitioner = NFoldPartitioner(attr=chunks_attr)
        if targets == 'familiarity':
            partitioner = FactorialPartitioner(partitioner, attr='familiarity')
    elif targets in ['identity-run', 'orientation-run']:
        # we are doing identity/orientation classification within
        # leave-one-run-out
        partitioner = ChainNode([
            NFoldPartitioner(attr='chunks'),
            ExcludeTargetsCombinationsPartitioner(
                k=1, targets_attr=chunks_attr, space='partitions')
        ], space='partitions')
    return partitioner


def _run_sl(ds_input, qe, partitioner, nproc, roi_ids):
    # setup cross validation
    cvte = CrossValidation(
        LinearCSVMC(),
        partitioner,
        errorfx=mean_match_accuracy)
    sl = Searchlight(cvte, queryengine=qe, nproc=nproc, roi_ids=roi_ids)
    # increase debug level
    mvpa2.debug.active += ['SLC']
    # run it
    slmap = sl(ds_input)
    return slmap


def _run_sl_permute(ds_input, qe, partitioner, permute, nproc, roi_ids):
    mvpa2.debug.active += ['SLC']
    rng = np.random.RandomState(3423)
    rng_permutations = [rng.randint(MAXINT) for _ in range(permute)]

    slmap = []
    partitioner_attr = partitioner.attr
    limit = [partitioner_attr]
    if 'chunks' in ds_input.sa:
        # this occurs whenever we have runs
        limit += ['chunks']
    for p, randomgen in enumerate(rng_permutations):
        permuter = AttributePermutator('targets',
                                       limit=limit,
                                       count=1, rng=randomgen)
        partitioner_rng = ChainNode([partitioner, permuter],
                                    space=partitioner.get_space())
        # setup cross validation
        cvte = CrossValidation(
            LinearCSVMC(),
            partitioner_rng,
            errorfx=mean_match_accuracy)
        sl = Searchlight(cvte, queryengine=qe, nproc=nproc, roi_ids=roi_ids)
        print("Running permutation {}".format(p))
        tmp = sl(ds_input)
        tmp.sa['permutation'] = [p]
        slmap.append(tmp)
    slmap = vstack(slmap)
    return slmap


def _get_bricklabels(fn):
    """Return labels associated with a volume"""
    img = nib.load(fn)
    pattern = '<AFNI_atr.*atr_name="BRICK_LABS" >(.*?)<\/AFNI_atr>'
    reg = re.compile(pattern, flags=re.DOTALL)
    m = reg.search(img.header.extensions[0].get_content())
    return m.group(1).translate(None, '\n "').split('~')


def _load_ds(tent_fn, mask_fn=None):
    # load dataset
    is_surface = False
    if tent_fn.endswith(('gii', 'niml.dset')):
        if mask_fn is not None:
            print("WARNING: if you're masking the surface, make sure to set "
                  "explicitly the `roi_ids` in the Searchlight call!")
        ds = niml.read(tent_fn)
        if 'node_indices' not in ds.fa:
            ds.fa['node_indices'] = np.arange(ds.nfeatures)
        if mask_fn:
            if not mask_fn.endswith(('gii', 'niml.dset')):
                raise ValueError("Provide a mask in gifti or niml.dset format")
            mask = niml.read(mask_fn)
            mask_node_indices = mask.fa.node_indices[mask.samples[0].astype(bool)]
            ds = ds.select(fadict={'node_indices': mask_node_indices})
        is_surface = True
    elif tent_fn.endswith(('nii', 'nii.gz')):
        ds = fmri_dataset(tent_fn, mask=mask_fn)
        ds.sa['labels'] = _get_bricklabels(tent_fn)
    else:
        raise ValueError("Can't use file {}".format(tent_fn))
    return ds, is_surface


def main():
    parsed = parse_args()
    run_and_save(
        tent_fn=parsed.input,
        targets=parsed.target,
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
                        help='number of permutations to run (0 = disabled)',
                        default=0)
    parser.add_argument('--input', '-i', type=str,
                        help='input file',
                        required=True)
    parser.add_argument('--output', '-o', type=str,
                        help='output directory (will create sub-sid* '
                             'directory)',
                        required=True)
    parser.add_argument('--target', '-t', type=str,
                        help='target for classification',
                        choices=[
                            'identity', 'orientation',
                            'fam-identity', 'fam-orientation',
                            'str-identity', 'str-orientation',
                            'fam-identity-run', 'fam-orientation-run',
                            'str-identity-run', 'str-orientation-run',
                            'fam-gender', 'str-gender',
                            'fam-identity-male', 'fam-orientation-male',
                            'fam-identity-female', 'fam-orientation-female',
                            'str-identity-male', 'str-orientation-male',
                            'str-identity-female', 'str-orientation-female',
                            'familiarity'
                        ],
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()


def test_create_outfn():
    tent_fn = 'sub-sid000005_task-str1back_space-fsaverage6_' \
              'hemi-R_glm-tent.niml.dset'
    out_fn = _create_outfn(tent_fn, 'identity')
    assert out_fn == 'sub-sid000005/sub-sid000005_task-str1back_' \
                     'space-fsaverage6_hemi-R_target-identity_sl.niml.dset'

    tent_fn = 'sub-sid000005_task-str1back_space-T1w_glm-tent.nii.gz'
    out_fn = _create_outfn(tent_fn, 'identity')
    assert out_fn == 'sub-sid000005/sub-sid000005_task-str1back_' \
                     'space-T1w_target-identity_sl.niml.dset'


def test_load_ds():
    here = os.path.dirname(os.path.abspath(__file__))
    ds, is_surface = _load_ds(os.path.join(here, 'test_afnilabels.nii.gz'))
    assert ds.nsamples == 626
    assert not is_surface


def test_setup_ds():
    here = os.path.dirname(os.path.abspath(__file__))
    ds, is_surface = _load_ds(os.path.join(here, 'test_afnilabels.nii.gz'))
    assert ds.nsamples == 626
    assert not is_surface

    # add known values
    ds.samples[:, 0] = np.arange(ds.nsamples)
    tstat_idx = map(lambda x: 'GLT' not in x and 'Tstat' in x, ds.sa.labels)
    tstat_labels = ds.sa.labels[tstat_idx]
    assert len(tstat_labels) == 280
    tstat_values = ds.samples[tstat_idx]

    ds_ = _setup_ds(ds)
    assert ds_.nsamples == 280
    assert_array_equal(tstat_values, ds_.samples)
    new_labels = map(lambda x: x.split('#')[0], tstat_labels)
    assert_array_equal(new_labels, ds_.sa.labels)


@pytest.mark.parametrize('targets', ['fam-identity', 'fam-orientation',
                                     'str-identitiy', 'str-orientation'])
def test_make_eventrelated(targets):
    here = os.path.dirname(os.path.abspath(__file__))
    ds, is_surface = _load_ds(os.path.join(here, 'test_afnilabels.nii.gz'))
    ds_ = _setup_ds(ds)
    ds_, targets_ = _filter_targets(ds_.copy(), targets)
    ds_.samples[:, 0] = np.arange(ds_.nsamples)
    ds_ev = _make_eventrelated(ds_)

    assert ds_ev.nsamples == 20
    assert ds_ev.nfeatures == ds_.nsamples / 20
    # the samples should be reshaped as (20, 7)
    assert_array_equal(ds_ev.samples, np.arange(ds_.nsamples).reshape(
        ds_ev.nsamples, -1))
    # and now let's check that we have the correct labels
    assert_array_equal(ds_ev.sa.labels,
                       ds_.sa.labels[np.arange(0, ds_.nsamples, 7)])
    # and finally check we have split them in identity and orientation
    # correctly
    assert_array_equal(ds_ev.sa.labels,
                       ['id{0}_a{1}'.format(i + 1, o + 1)
                        for i, o in zip(ds_ev.sa.identity,
                                        ds_ev.sa.orientation)])


def test_filter_targets():
    here = os.path.dirname(os.path.abspath(__file__))
    ds, is_surface = _load_ds(os.path.join(here, 'test_afnilabels.nii.gz'))
    ds = _setup_ds(ds)
    nsamples = ds.nsamples

    ds_, targets_ = _filter_targets(ds.copy(), 'familiarity')
    assert ds_.nsamples == nsamples
    assert 'familiarity' in ds_.sa.keys()
    assert len(np.unique(ds_.sa.familiarity)) == 2
    assert np.sum(ds_.sa.familiarity == 1) == np.sum(ds_.sa.familiarity == 0)
    assert targets_ == 'familiarity'

    for targets in ['fam-identity', 'fam-orientation', 'str-identity',
                    'str-orientation']:
        ds_, targets_ = _filter_targets(ds, targets)
        assert ds_.nsamples == nsamples / 2
        assert targets_ == targets.replace('fam-', '').replace('str-', '')

