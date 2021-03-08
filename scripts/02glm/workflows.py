"""Module containing workflows for hyperface"""
import sys
import os
from six import iteritems
from nipype.interfaces.afni import (Deconvolve, Remlfit, ConvertDset,
                                    BlurToFWHM, Despike, TProject)
from nipype.interfaces.afni.utils import OneDToolPy
from nipype.interfaces.base import isdefined
from nipype.interfaces import Function, IdentityInterface, Merge
from nipype.pipeline import Node, MapNode, Workflow
from famfaceangles.interfaces import TimingToolPy
from famfaceangles.workflows import (init_makeortvec_mapnode,
                                     init_afnipreprocess_surf_wf,
                                     init_datasource_wf,
                                     init_afnipreprocess_wf)

# This is the python path used by TimingTool and AFNI's python functions
PY27_PATH = os.environ.get('PY27_PATH', 'python2')

SPACES = {
    'fsaverage6': 'fsaverage6',
    'MNI': 'MNI152NLin2009cAsym',
    'T1w': 'T1w',
    'hpalfsaverage6': 'hpalfsaverage6',
}


def basename(list_of_paths):
    import os.path as op
    return [op.basename(path) for path in list_of_paths]


def init_afnimodel_wf(blurfwhm=4.0, do_remlfit=False,
                      out_ext='gii', name='afnimodel', njobs=1):
    """
    Workflow running smoothing + Deconvolve + REMLFit

    Parameters
    ----------
    blurfwhm: float or None
        amount of spatial smoothing to apply prior to computing the glm model
    do_remlfit : bool
        whether to run also remlfit
    out_ext: 'gii' | 'nii.gz' | 'nii' | 'niml.dset'
        output extension
    name
    njobs

    Returns
    -------
    wf

    **Inputs**

        run_files
            list of run files
        stim_times
            stimuli onset in AFNI format, that is a list of tuples
            (int_stim, stim_file, model)
        stim_label
            list of stimuli labels, same length as stim_times; tuples of
            form (int_stim, stim_label)
        ortvec
            a file containing additional regressors as a matrix
        censor
            a 1D file containing TRs to censor
        gltsym
            list of strings, symbolic notation for general linear test,
            e.g. ('+face -house')
        glt_label
            list of contrast labels, e.g. (1, 'facevshouse')
        mask
            mask file


    **Outputs**

        betas
            gifti/nifti file containing the betas (depends on input format)
        matrix
            design matrix used by REMLfit

    """

    wf = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['run_files',
                                               'stim_times',
                                               'stim_label',
                                               'ortvec',
                                               'censor',
                                               'gltsym',
                                               'glt_label',
                                               'mask',
                                               ]),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['betas_deconvolve',
                                                'betas_remlfit',
                                                'matrix']),
                      name='outputnode')

    """
    ============================
    setup blurring 
    ============================
    """
    smooth = MapNode(BlurToFWHM(),
                     name='smooth' + '{}mm'.format(blurfwhm).replace('.', ''),
                     iterfield=['in_file'])
    if blurfwhm:
        if out_ext in ['gii', 'niml.dset']:
            raise ValueError("Smoothing on surface not yet supported")
        # TODO: extend this in case the input is gifti. now it assumes it's nii
        smooth.inputs.fwhm = blurfwhm
        smooth.inputs.outputtype = 'NIFTI_GZ'

    """
    ============================
    setup deconvolve and remlfit 
    ============================
    """
    # set up deconvolve
    # TODO: make it customizable
    deconvolve = Node(Deconvolve(
        num_threads=njobs,
        polort=3,
        tout=True,
        fout=True,
        nofdr=False,
        out_file='Decon.' + out_ext,
        #force_TR=1.25,
    ), name='deconvolve')

    remlfit = Node(Remlfit(
        tout=True,
        nofdr=False,
        num_threads=njobs,
        out_file='Remlfit.' + out_ext
    ), name='remlfit')

    # smoothing
    if blurfwhm:
        wf.connect([
            (inputnode, smooth, [('run_files', 'in_file'),
                                 ('mask', 'mask')]),
            (smooth, deconvolve, [('out_file', 'in_files')]),
            (smooth, remlfit, [('out_file', 'in_files')])
        ])
    else:
        wf.connect([
            (inputnode, deconvolve, [('run_files', 'in_files')]),
        ])

    wf.connect([
        (inputnode, deconvolve, [('stim_times', 'stim_times'),
                                 ('stim_label', 'stim_label'),
                                 ('gltsym', 'gltsym'),
                                 ('glt_label', 'glt_label'),
                                 ('ortvec', 'ortvec'),
                                 ('censor', 'censor'),
                                 ('mask', 'mask')]),
        (deconvolve, outputnode, [('x1D', 'matrix'),
                                  ('out_file', 'betas_deconvolve')])
    ])

    if do_remlfit:
        # setup remlfit
        wf.connect([
            (inputnode, remlfit, [('run_files', 'in_files')]),
            (inputnode, remlfit, [('mask', 'mask')]),
            (deconvolve, remlfit, [('x1D', 'matrix')]),
            (remlfit, outputnode, [('out_file', 'betas_remlfit')]),
        ])

    return wf


def init_glm_localizer_wf(subject, space='fsaverage6',
                          model='BLOCK(1.5,1)',
                          name='glmlocalizer_wf', njobs=1):
    """
    GLM pipeline for localizer data.

    Parameters
    ----------
    subject
    space : 'fsaverage6' | 'MNI' | 'T1w'
        which normalized files to compute GLM on
    model : str
        model specification passed on to 3dDeconvolve
    name
    njobs

    Returns
    -------
    wf workflow


    **Inputs**

        data_dir
            directory containing BIDS dataset
        fmriprep_dir
            directory of derivatives containing the fmriprep data
        mask
            mask to be used for analysis
        hemifield
            'L' or 'R'; which hemifield to process


    **Outputs**

        betas_deconvolve
            file containing the betas estimated from 3dDeconvolve
        betas_remlfit
            file containing the betas estimated from 3dREMLfit
    """

    if space not in SPACES:
        raise ValueError("I don't know of space {0}, needs to be one "
                         "of {1}".format(space, SPACES.keys()))
    space = SPACES[space]

    wf = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['data_dir',
                                               'fmriprep_dir',
                                               'mask',
                                               'hemifield']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['betas_deconvolve',
                                                'betas_remlfit',
                                                'betas_niml']),
                      name='outputnode')

    # datasource wf
    datasource_wf = init_datasource_wf(subject, 'localizer', space,
                                       name='datasource')
    # setup nodes for nuisance regressors
    motion = init_makeortvec_mapnode(which='motion', name='motion')

    compcor = init_makeortvec_mapnode(which='compcor', name='compcor')

    afnistim_wf = init_afnistim_localizer_wf(
        name='afnistim_localizer',
        model=model)

    # preprocessing
    afnipreproc_wf = init_afnipreprocess_surf_wf()
    wf.connect([
        (inputnode, afnipreproc_wf, [('mask', 'inputnode.mask')]),
        (datasource_wf, afnipreproc_wf, [('outputnode.run_files',
                                          'inputnode.run_file')]),
        (motion, afnipreproc_wf, [('ortvec', 'inputnode.motion_params')]),
        (compcor, afnipreproc_wf,
         [('ortvec', 'inputnode.nuisance_regressors')])
    ])

    # stack the censor file
    def vstack(fns):
        import numpy as np
        import os.path as op
        fnout = op.abspath('censor_vstacked.txt')
        # XXX: this assumes I already know these are column vectors
        censors = [np.loadtxt(c)[:, None] for c in fns]
        stack = np.vstack(censors).astype(int)
        np.savetxt(fnout, stack, fmt='%d')
        return fnout

    censor_stack = Node(
        Function(function=vstack,
                 input_names=['fns'],
                 output_names=['out_file']),
        name='censor_stack')
    wf.connect(afnipreproc_wf, 'outputnode.censor', censor_stack, 'fns')

    # AFNI model
    out_ext = 'niml.dset'

    afnimodel_wf = init_afnimodel_wf(blurfwhm=None,
                                     out_ext=out_ext,
                                     njobs=njobs,
                                     do_remlfit=True,
                                     name='afnimodel')

    # Add contrasts
    #afnimodel_wf.inputs.inputnode.gltsym = [
    #    '+4*faces -objects -scenes -bodies -scrambled_objects']
    afnimodel_wf.inputs.inputnode.gltsym = ['+faces -objects']
    afnimodel_wf.inputs.inputnode.glt_label = [(1, 'faces_vs_objects')]

    def enumerate_tuples(list_of_tuples):
        return [tuple([i] + list(tup)[1:])
                for i, tup in enumerate(list_of_tuples, 1)]

    def mylen(iterable):
        return len(iterable)
    """
    link everything together
    """
    wf.connect([
        # sort files
        (inputnode, datasource_wf,
         [('data_dir', 'inputnode.data_dir'),
          ('fmriprep_dir', 'inputnode.fmriprep_dir'),
          ('hemifield', 'inputnode.hemi')]),
        # make stim times
        (datasource_wf, afnistim_wf,
         [('outputnode.event_files', 'inputnode.event_files')]),
        # pass number of runs before the strangers
        # make confound matrix / motion
        (datasource_wf, motion,
         [('outputnode.confound_files', 'in_file')]),
        # make confound matrix / compcor
        (datasource_wf, compcor,
         [('outputnode.confound_files', 'in_file')]),
        # pass preprocessed file to afnimodel
        (afnipreproc_wf, afnimodel_wf,
         [('outputnode.preproc_file', 'inputnode.run_files')]),
        # pass stimulus times
        (afnistim_wf, afnimodel_wf,
         [('outputnode.stim_times', 'inputnode.stim_times'),
          ('outputnode.stim_label', 'inputnode.stim_label')]),
        (inputnode, afnimodel_wf, [('mask', 'inputnode.mask')]),
        (censor_stack, afnimodel_wf,
         [('out_file', 'inputnode.censor')]),
        # pass output
        (afnimodel_wf, outputnode, [
            ('outputnode.betas_deconvolve', 'betas_deconvolve'),
            ('outputnode.betas_remlfit', 'betas_remlfit')]),
    ])

    return wf


def init_glm_localizer_nodatasource_wf(model='BLOCK(1.5,1)',
                                       name='glmlocalizer_wf', njobs=1):
    """
    GLM pipeline for localizer data without datasource

    Parameters
    ----------
    model : str
        model specification passed on to 3dDeconvolve
    name
    njobs

    Returns
    -------
    wf workflow


    **Inputs**

        run_files
        event_files
        confound_files


    **Outputs**

        betas_deconvolve
            file containing the betas estimated from 3dDeconvolve
        betas_remlfit
            file containing the betas estimated from 3dREMLfit
    """
    wf = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['run_files',
                                               'event_files',
                                               'confound_files']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['betas_deconvolve',
                                                'betas_remlfit',
                                                'betas_niml']),
                      name='outputnode')

    # datasource wf
    # setup nodes for nuisance regressors
    motion = init_makeortvec_mapnode(which='motion', name='motion')

    compcor = init_makeortvec_mapnode(which='compcor', name='compcor')

    afnistim_wf = init_afnistim_localizer_wf(
        name='afnistim_localizer',
        model=model)

    # preprocessing
    afnipreproc_wf = init_afnipreprocess_surf_wf()
    wf.connect([
        #(inputnode, afnipreproc_wf, [('mask', 'inputnode.mask')]),
        (inputnode, afnipreproc_wf, [('run_files', 'inputnode.run_file')]),
        (motion, afnipreproc_wf, [('ortvec', 'inputnode.motion_params')]),
        (compcor, afnipreproc_wf,
         [('ortvec', 'inputnode.nuisance_regressors')])
    ])

    # stack the censor file
    def vstack(fns):
        import numpy as np
        import os.path as op
        fnout = op.abspath('censor_vstacked.txt')
        # XXX: this assumes I already know these are column vectors
        censors = [np.loadtxt(c)[:, None] for c in fns]
        stack = np.vstack(censors).astype(int)
        np.savetxt(fnout, stack, fmt='%d')
        return fnout

    censor_stack = Node(
        Function(function=vstack,
                 input_names=['fns'],
                 output_names=['out_file']),
        name='censor_stack')
    wf.connect(afnipreproc_wf, 'outputnode.censor', censor_stack, 'fns')

    # AFNI model
    out_ext = 'niml.dset'

    afnimodel_wf = init_afnimodel_wf(blurfwhm=None,
                                     out_ext=out_ext,
                                     njobs=njobs,
                                     do_remlfit=True,
                                     name='afnimodel')

    # Add contrasts
    afnimodel_wf.inputs.inputnode.gltsym = ['+face -object']
    afnimodel_wf.inputs.inputnode.glt_label = [(1, 'faces_vs_objects')]

    def enumerate_tuples(list_of_tuples):
        return [tuple([i] + list(tup)[1:])
                for i, tup in enumerate(list_of_tuples, 1)]

    def mylen(iterable):
        return len(iterable)
    """
    link everything together
    """
    wf.connect([
        # make stim times
        (inputnode, afnistim_wf,
         [('event_files', 'inputnode.event_files')]),
        # pass number of runs before the strangers
        # make confound matrix / motion
        (inputnode, motion,
         [('confound_files', 'in_file')]),
        # make confound matrix / compcor
        (inputnode, compcor,
         [('confound_files', 'in_file')]),
        # pass preprocessed file to afnimodel
        (afnipreproc_wf, afnimodel_wf,
         [('outputnode.preproc_file', 'inputnode.run_files')]),
        # pass stimulus times
        (afnistim_wf, afnimodel_wf,
         [('outputnode.stim_times', 'inputnode.stim_times'),
          ('outputnode.stim_label', 'inputnode.stim_label')]),
        #(inputnode, afnimodel_wf, [('mask', 'inputnode.mask')]),
        (censor_stack, afnimodel_wf,
         [('out_file', 'inputnode.censor')]),
        # pass output
        (afnimodel_wf, outputnode, [
            ('outputnode.betas_deconvolve', 'betas_deconvolve'),
            ('outputnode.betas_remlfit', 'betas_remlfit')]),
    ])

    return wf


def init_afnistim_localizer_wf(model='BLOCK(1.5,1)',
                               tr=None,
                               name='afnistim_localizer_wf',):
    """
    Generate AFNI stims for each individual target category, across runs.

    Parameters
    ----------
    model: str
        symbolic notation of the model to use (see 3dDeconvolve's help)
    tr : float or None
        repetition time
        If set, stim times are truncated to the nearest TR using
        timing_tool.py in AFNI
    name : str
        name of the workflow

    Returns
    -------
    wf : workflow


    ** Inputs **

        event_files : list of event files

    ** Outputs **

        stim_times : list of stimulus files in AFNI format
        stim_label : list of stim labels
    """

    wf = Workflow(name)
    inputnode = Node(IdentityInterface(fields=['event_files']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['stim_times', 'stim_label']),
                      name='outputnode')
    make_afnistim = Node(
        Function(function=make_afni_stimtimes_localizer,
                 input_names=['in_files', 'start'],
                 output_names=['stim_file']),
        name='make_afnistim')

    # add model specification
    def add_model_str(stim_file, model):
        return [(i, f, model) for i, f in stim_file]
    add_model = Node(
        Function(function=add_model_str,
                 input_names=['stim_file', 'model'],
                 output_names=['stim_file']),
        name='add_model')
    add_model.inputs.model = model

    """
    add stim labels for afni
    """
    def get_stimlabels(stim_file):
        import os.path as op
        stim_labels = []
        for sf in stim_file:
            lbl = op.basename(sf).replace('.txt', '')
            stim_labels.append(lbl)
        return stim_labels

    make_stimlabel = Node(Function(function=get_stimlabels,
                                   input_names=['stim_file'],
                                   output_names=['stim_label']),
                          name='get_stimlabels')

    def enumeratelist(iterable, start=1):
        return list(enumerate(iterable, start))

    if tr is not None:
        truncate_times = MapNode(
            TimingToolPy(truncate_times=True, tr=tr,
                         py27_path=PY27_PATH),
            iterfield=['in_file', 'out_file'],
            name='truncate_times')
        wf.connect([
            (make_afnistim, truncate_times,
             [('stim_file', 'in_file'),
              (('stim_file', basename), 'out_file')]),
            (truncate_times, add_model,
             [(('out_file', enumeratelist), 'stim_file')])
        ])
    else:
        wf.connect([
            (make_afnistim, add_model,
             [(('stim_file', enumeratelist), 'stim_file')]),
        ])
    wf.connect([
        (inputnode, make_afnistim, [('event_files', 'in_files')]),
        (make_afnistim, make_stimlabel, [('stim_file', 'stim_file')]),
        (add_model, outputnode, [('stim_file', 'stim_times')]),
        (make_stimlabel, outputnode,
         [(('stim_label', enumeratelist), 'stim_label')])
    ])
    return wf


def make_afni_stimtimes_localizer(in_files, start=0):
    """Given event files, return stim times for AFNI for every exemplar

    Parameters
    ----------
    in_files: list of list
    start: int
        if we need to assume there are `start` runs before these ones. it
        will append `start` asterisks (*) to the stimulus files
    """
    import pandas as pd
    from collections import defaultdict
    from os.path import basename, abspath
    from six import iteritems

    def make_dict_onsets(filename):
        events = pd.read_csv(filename, sep='\t')
        onsets = defaultdict(list)
        for row in events.itertuples():
            key = row.stim_type
            if key != 'fixation':
                onsets[key].append('{0:.2f}'.format(row.onset))
        onsets = dict(onsets)
        return onsets

    onsets = dict()
    for f in in_files:
        on = make_dict_onsets(f)
        # format them
        for k, o in iteritems(on):
            on[k] = ' '.join(o)
        onsets[f] = on

    # get unique stim labels across all runs
    stim_labels = []
    for ifile in in_files:
        stim_labels += list(onsets[ifile].keys())
    stim_labels = sorted(list(set(stim_labels)))

    # create stim files
    stim_files = dict()
    for sl in stim_labels:
        stim_files[sl] = [onsets[f].get(sl, '* *') for f in in_files]

    # save
    header = '\n'.join(['* *'] * start) if start else ''
    fns = []
    for sl in stim_labels:
        fns.append(abspath(sl) + '.txt')
        with open(fns[-1], 'w') as f:
            f.write(header)
            f.write('\n')
            f.write('\n'.join(stim_files[sl]))
    return fns
