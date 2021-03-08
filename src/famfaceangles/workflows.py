"""Module containing nipype workflows"""
import sys
import os
from six import iteritems
from nipype.interfaces.afni import (Deconvolve, Remlfit, ConvertDset,
                                    BlurToFWHM, Despike, TProject)
from nipype.interfaces.afni.utils import OneDToolPy
from nipype.interfaces.base import isdefined
from nipype.interfaces import Function, IdentityInterface, Merge
from nipype.pipeline import Node, MapNode, Workflow
from .interfaces import TimingToolPy

# This is the python path used by TimingTool and AFNI's python functions
PY27_PATH = os.environ.get('PY27_PATH', 'python2')


def make_afni_stimtimes_exemplar(in_files, start=0, per_run=False):
    """Given event files, return stim times for AFNI for every exemplar

    Parameters
    ----------
    in_files: list of list
    start: int
        if we need to assume there are `start` runs before these ones. it
        will append `start` asterisks (*) to the stimulus files
    per_run: bool
        whether to get the stimulus times for each run instead of across runs
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from os.path import basename, abspath
    from six import iteritems

    def make_dict_onsets(filename):
        events = pd.read_csv(filename, sep='\t')
        onsets = defaultdict(list)
        columns = events.columns
        for row in events.itertuples():
            # model the first trial separately
            if row.Index == 0:
                key = 'first_trial'
                onsets[key].append("{0:.2f}".format(row.onset)
                                   if row.stim_type == 'face' else '* *')
            elif row.stim_type == 'face':
                key = '_'.join([row.identity, row.orientation])
                onsets[key].append("{0:.2f}".format(row.onset))
                # also store button press
                if 'response' in columns:
                    if row.response == 'same':
                        rt = "{0:.2f}".format(row.onset + row.reaction_time)
                        onsets['same_response'].append(rt)
        onsets = dict(onsets)
        return onsets

    onsets = dict()
    for f in in_files:
        on = make_dict_onsets(f)
        keep_keys = []
        # format them
        for k, o in iteritems(on):
            # do not store empty stimtimes
            if list(set(o)) != ['* *']:
                on[k] = ' '.join(o)
                keep_keys.append(k)
        onsets[f] = {k: on[k] for k in keep_keys}

    # get stim labels
    stim_labels = sorted(list(onsets[in_files[0]].keys()))

    # create stim files
    stim_files = dict()
    if per_run:
        #  we need to keep the run information, and add extra '* *' for every
        #  run we skip
        for irun, f in enumerate(in_files, 1):
            onsets_run = onsets[f]
            for sl, on in iteritems(onsets_run):
                lbl = sl + '_run{0:02d}'.format(irun)
                stim_files[lbl] = ['* *' for _ in range(irun-1)] + \
                                  [on]
    else:
        for sl in stim_labels:
            stim_files[sl] = []
            for f in in_files:
                if sl in onsets[f]:
                    toadd = onsets[f][sl]
                else:
                    toadd = '* *'
                stim_files[sl].append(toadd)

    # save
    header = '\n'.join(['* *'] * start) + '\n' if start else ''
    fns = []
    for lbl, stim_times in iteritems(stim_files):
        fns.append(abspath(lbl) + '.txt')
        with open(fns[-1], 'w') as f:
            f.write(header)
            f.write('\n'.join(stim_times))
    return fns


def bidsgrabber(base_dir, query):
    from bids.grabbids import BIDSLayout
    try:
        layout = BIDSLayout(base_dir, extensions=['derivatives'])
    except TypeError:  # newever version of pybids
        layout = BIDSLayout((base_dir, ['bids', 'derivatives']))
    files = layout.get(**query)
    if len(files) == 0:
        raise ValueError("Query returned 0 files")
    return [f.filename for f in files]


def write_filelist(run_files, event_files, confound_files):
    """Write the list of files used as input to disk, just to be sure"""
    import os
    fn = os.path.abspath('report.txt')
    f = open(fn, 'w')
    towrite = {'Runs': run_files,
               'Events': event_files,
               'Confounds': confound_files}
    for what in ['Runs', 'Events', 'Confounds']:
        f.write(what + '\n')
        f.write('=' * len(what) + '\n')
        for file_fn in towrite[what]:
            f.write(2*' ' + os.path.basename(file_fn) + '\n')
    f.close()
    return fn


def write_filelist_run(run_files):
    """Write the list of files used as input to disk, just to be sure"""
    import os
    fn = os.path.abspath('report.txt')
    f = open(fn, 'w')
    towrite = {'Runs': run_files}
    for what in ['Runs']:
        f.write(what + '\n')
        f.write('=' * len(what) + '\n')
        for file_fn in towrite[what]:
            f.write(2*' ' + os.path.basename(file_fn) + '\n')
    f.close()
    return fn


def init_datasource_wf(subject, task, space, name='datasource_wf'):
    """
    Workflow for grabbing the necessary data

    Parameters
    ----------
    subject
    task
    space
    name

    Returns
    -------
    wf


    **Inputs**

        data_dir
            directory containing BIDS dataset
        fmriprep_dir
            directory of derivatives containing the fmriprep data
        hemifield
            'L' or 'R'; which hemifield to process


    **Outputs**

        run_files
            list of run files
        event_files
            list of event files associated to the run files
        confound_files
            list of confound files (e.g., generated by fmriprep)
    """
    wf = Workflow(name=name)
    inputnode = Node(IdentityInterface(fields=['data_dir',
                                               'fmriprep_dir',
                                               'hemi']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['run_files',
                                                'event_files',
                                                'confound_files']),
                      name='outputnode')
    """
    =========================== 
    Setup data grabber for BIDS
    ===========================
    """
    grab_events = Node(Function(function=bidsgrabber,
                                input_names=['base_dir', 'query'],
                                output_names=['files']),
                       name='grab_events')
    grab_derivs = Node(Function(function=bidsgrabber,
                                input_names=['base_dir', 'query'],
                                output_names=['files']),
                       name='grab_derivs')
    grab_confs = Node(Function(function=bidsgrabber,
                               input_names=['base_dir', 'query'],
                               output_names=['files']),
                      name='grab_confs')
    # we query the events from data, and we query the giftis/niftis from
    # derivatives
    surface = True
    if 'hpal' in space and 'fsaverage6' in space:
        out_ext = 'niml.dset'
    elif 'fsaverage' in space:
        out_ext = 'gii'
    else:
        surface = False
        out_ext = 'nii.gz'
    grab_events.inputs.query = dict(subject=subject,
                                    extensions='.tsv',
                                    task=task)
    grab_derivs.inputs.query = dict(subject=subject,
                                    extensions=out_ext,
                                    task=task,
                                    space=space)
    if 'fsaverage' not in space:
        # do not get brainmasks
        grab_derivs.inputs.query['type'] = 'preproc'

    grab_confs.inputs.query = dict(subject=subject,
                                   extensions='.tsv',
                                   task=task,
                                   type='confounds')
    """
    ====================================
    filter files for only one hemisphere
    ====================================
    """

    def filter_hemisphere(in_files, hemi):
        return list(filter(lambda x: hemi in x, in_files))

    filterhemisphere = Node(Function(function=filter_hemisphere,
                                     input_names=['in_files', 'hemi'],
                                     output_names=['files']),
                            name='filter_hemi')

    """
    sort files by run number so that they all match
    """

    def sort_byrun(run_files, event_files, confound_files):
        def runnr(file):
            import re
            r = re.compile(r'run-(\d{2})')
            n = r.findall(file)
            if n:
                return int(n[0])
            else:
                raise ValueError("Did not find run information on "
                                 "filename {0}".format(file))

        run_files = sorted(run_files, key=runnr)
        event_files = sorted(event_files, key=runnr)
        confound_files = sorted(confound_files, key=runnr)
        return run_files, event_files, confound_files

    sort_files = Node(Function(
        function=sort_byrun,
        input_names=['run_files', 'event_files', 'confound_files'],
        output_names=['run_files', 'event_files', 'confound_files']),
        name='sort_files')

    wf.connect([
        (inputnode, grab_events, [('data_dir', 'base_dir')]),
        (inputnode, grab_derivs, [('fmriprep_dir', 'base_dir')]),
        (inputnode, grab_confs, [('fmriprep_dir', 'base_dir')])

    ])

    if surface:
        wf.connect([
            (inputnode, filterhemisphere, [('hemi', 'hemi')]),
            (grab_derivs, filterhemisphere, [('files', 'in_files')]),
            (filterhemisphere, sort_files, [('files', 'run_files')])
        ])
    else:
        wf.connect(grab_derivs, 'files', sort_files, 'run_files')

    wf.connect([
        (grab_events, sort_files, [('files', 'event_files')]),
        (grab_confs, sort_files, [('files', 'confound_files')]),
        (sort_files, outputnode, [('run_files', 'run_files'),
                                  ('event_files', 'event_files'),
                                  ('confound_files', 'confound_files')])
    ])

    return wf


def init_makeortvec_mapnode(which='motion', name='makeortvec'):
    """
    Given input confound files from fmriprep, generate a matrix containing
    nuisance regressors

    Parameters
    ----------
    which: str ('motion' | 'compcor' | 'all')
        if 'motion', returns the 6 motion parameters
        if 'compcor', returns the aCompCor components
        if 'all', returns both 'motion' and 'compcor'
    name: str
        name of the node

    Returns
    -------
    make_matconfounds: Node
    """
    # pass them in order as for AFNI, i.e., "roll pitch yaw dS  dL  dP"
    motion_names = ['RotX', 'RotY', 'RotZ', 'X', 'Y', 'Z']
    compcor_names = ['aCompCor0' + str(i) for i in range(6)]
    if which == 'motion':
        names = motion_names
    elif which == 'compcor':
        names = compcor_names
    else:
        names = motion_names + compcor_names

    def make_matconfounds(in_file, confounds):
        import pandas as pd
        import numpy as np
        import os

        mat_confounds = np.array(
            pd.read_csv(in_file, sep='\t')[confounds].fillna(0))
        fn = os.path.abspath('mat_confounds.txt')
        np.savetxt(fn, mat_confounds)
        return fn

    make_matconfounds = MapNode(
        Function(function=make_matconfounds,
                 input_names=['in_file', 'confounds'],
                 output_names=['ortvec']),
        iterfield=['in_file'],
        name=name)
    make_matconfounds.inputs.confounds = names
    return make_matconfounds


def basename(list_of_paths):
    import os.path as op
    return [op.basename(path) for path in list_of_paths]


def init_afnistim_exemplar_wf(model='TENT(2.5,17,7)',
                              tr=1.25,
                              per_run=False,
                              prefix='',
                              name='afnistim_exemplar_wf',):
    """
    Generate AFNI stims for each individual target category, across runs.

    Parameters
    ----------
    model: str
        symbolic notation of the model to use (see 3dDeconvolve's help)
    tr : float
        repetition time
        If set, stim times are truncated to the nearest TR using
        timing_tool.py in AFNI
    per_run : bool
        whether to get stimulus times for each run instead of across runs
    prefix: str
        prefix to add to the stimulus files
    name : str
        name of the workflow

    Returns
    -------
    wf : workflow


    ** Inputs **

        event_files : list of event files
        start_run : integer indicating how many previous runs to consider
                    (used to add as many * as necessary to the stim files)

    ** Outputs **

        stim_times : list of stimulus files in AFNI format
        stim_label : list of stim labels
    """

    wf = Workflow(name)
    inputnode = Node(IdentityInterface(fields=['event_files',
                                               'start_run']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['stim_times', 'stim_label']),
                      name='outputnode')
    make_afnistim = Node(
        Function(function=make_afni_stimtimes_exemplar,
                 input_names=['in_files', 'start', 'per_run'],
                 output_names=['stim_file']),
        name='make_afnistim')
    make_afnistim.inputs.per_run = per_run

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
    def get_stimlabels(stim_file, prefix):
        import re
        import os.path as op
        # XXX: this is experiment-specific
        # r = re.compile('(id\d_a\d)')
        stim_labels = []
        pf = prefix + '_' if prefix else ''
        for sf in stim_file:
            # # XXX: this depends on make_afni_stimtimes_exemplar
            # if 'first_trial' in sf or 'same_response' in sf:
            #     stim_labels.append(pf + op.basename(sf).replace('.txt', ''))
            # else:
            #     stim_labels.append(pf + r.findall(sf)[0])
            stim_labels.append(pf + op.basename(sf).replace('.txt', ''))
        return stim_labels

    make_stimlabel = Node(Function(function=get_stimlabels,
                                   input_names=['stim_file', 'prefix'],
                                   output_names=['stim_label']),
                          name='get_stimlabels')
    make_stimlabel.inputs.prefix = prefix

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
        (inputnode, make_afnistim, [('event_files', 'in_files'),
                                    ('start_run', 'start')]),
        (make_afnistim, make_stimlabel, [('stim_file', 'stim_file')]),
        (add_model, outputnode, [('stim_file', 'stim_times')]),
        (make_stimlabel, outputnode,
         [(('stim_label', enumeratelist), 'stim_label')])
    ])
    return wf


def init_afnipreprocess_wf(blurfwhm=4.0, name='afnipreprocess'):
    """
    Workflow running preprocessing for a single run, that is
        - Compute which TRs to censors based on motion parameters
        - Despike
        - Bandpass filtering and removal of nuisance regressors
        - Blurring

    Parameters
    ----------
    blurfwhm: float
        amount of spatial smoothing; if 0 no spatial smoothing applied
    name

    Returns
    -------
    wf


    **Inputs**

        run_file
            run file
        motion_params
            motion parameters in matrix form
        mask (optional)
            mask to use
        nuisance_regressors (optional)
            additional nuisance regressors to remove

    **Outputs**

        preproc_file
            preprocessed_file
        censor
            1D censor file
    """
    def basename(listfn, sfx=''):
        def _basename(fn):
            import os
            if fn.endswith('.nii.gz'):
                ext = '.nii.gz'
            elif fn.endswith('.gii'):
                ext = '.gii'
            elif fn.endswith('.niml.dset'):
                ext = '.niml.dset'
            fn = fn.replace(ext, '')
            fn = os.path.basename(fn)
            if sfx:
                fn += '_' + sfx
            return os.path.basename(fn) + ext

        listfn = [_basename(f) for f in listfn]
        return listfn

    wf = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['run_file',
                                               'motion_params',
                                               'mask',
                                               'nuisance_regressors',
                                               ]),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['preproc_file',
                                                'censor']),
                      name='outputnode')

    censor = MapNode(OneDToolPy(
        censor_motion=(1.2, 'censor'),
        censor_prev_TR=True,
        py27_path=PY27_PATH),
        iterfield=['in_file'],
        name='censor')
    wf.connect(inputnode, 'motion_params', censor, 'in_file')

    despike = MapNode(Despike(
        outputtype='NIFTI_GZ',
        num_threads=2),
        iterfield=['in_file', 'out_file'],
        name='despike')
    wf.connect([
        (inputnode, despike, [('run_file', 'in_file'),
                              (('run_file', basename, 'ds'), 'out_file')])
    ])

    def merge_mat(motion_params, nuisance_regressors=None):
        import numpy as np
        import os
        mp = np.loadtxt(motion_params)
        if nuisance_regressors is not None:
            nr = np.loadtxt(nuisance_regressors)
            mpnr = np.hstack((mp, nr))
        else:
            mpnr = mp
        fn = os.path.abspath('merged_regr.txt')
        np.savetxt(fn, mpnr)
        return fn

    merge_regressors = MapNode(Function(
        function=merge_mat,
        input_names=['motion_params', 'nuisance_regressors'],
        output_names=['out_file']),
        name='merge_regressors',
        iterfield=['motion_params', 'nuisance_regressors']
    )

    tproject = MapNode(TProject(
        outputtype='NIFTI_GZ',
        bandpass=(0.00667, 99999)),
        iterfield=['in_file', 'out_file', 'ort'],
        name='tproject')

    wf.connect([
        (inputnode, merge_regressors,
         [('motion_params', 'motion_params'),
          ('nuisance_regressors', 'nuisance_regressors')]),
        (merge_regressors, tproject, [('out_file', 'ort')])])

    wf.connect([
        (inputnode, tproject, [('mask', 'mask')]),
        (despike, tproject, [('out_file', 'in_file'),
                             (('out_file', basename, 'tp'), 'out_file')]),
    ])

    if blurfwhm:
        smooth = MapNode(BlurToFWHM(
            fwhm=blurfwhm,
            outputtype='NIFTI_GZ'),
            iterfield=['in_file', 'out_file'],
            name='smooth')
        wf.connect([
            (tproject, smooth, [('out_file', 'in_file'),
                                (('out_file', basename, 'bl'), 'out_file')])
        ])
        wf.connect(smooth, 'out_file', outputnode, 'preproc_file')
    else:
        wf.connect(tproject, 'out_file', outputnode, 'preproc_file')
    wf.connect(censor, 'out_file', outputnode, 'censor')
    return wf


def init_afnipreprocess_surf_wf(name='afnipreprocess'):
    """
    Workflow running preprocessing for a single run on surface, that is
        - Compute which TRs to censors based on motion parameters
        - Bandpass filtering and removal of nuisance regressors

    Parameters
    ----------
    name

    Returns
    -------
    wf


    **Inputs**

        run_file
            run file
        motion_params
            motion parameters in matrix form
        mask (optional)
            mask to use
        nuisance_regressors (optional)
            additional nuisance regressors to remove

    **Outputs**

        preproc_file
            preprocessed_file
        censor
            1D censor file
    """
    def basename(listfn, sfx=''):
        def _basename(fn):
            import os
            ext = ''
            if fn.endswith('.nii.gz'):
                ext = '.nii.gz'
            elif fn.endswith('.gii'):
                ext = '.gii'
            elif fn.endswith('.niml.dset'):
                ext = '.niml.dset'
            fn = fn.replace(ext, '')
            fn = os.path.basename(fn)
            if sfx:
                fn += '_' + sfx
            return os.path.basename(fn) + ext

        listfn = [_basename(f) for f in listfn]
        return listfn

    wf = Workflow(name=name)

    inputnode = Node(IdentityInterface(fields=['run_file',
                                               'motion_params',
                                               'nuisance_regressors',
                                               'mask'
                                               ]),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['preproc_file',
                                                'censor']),
                      name='outputnode')

    censor = MapNode(OneDToolPy(
        censor_motion=(1.2, 'censor'),
        censor_prev_TR=True,
        py27_path=PY27_PATH),
        iterfield=['in_file'],
        name='censor')
    wf.connect(inputnode, 'motion_params', censor, 'in_file')

    def merge_mat(params):
        """Input must be a list, where first element is motion parameters
        and second element is (optional) other nuisance regressors"""
        if isinstance(params, str):
            params = [params]
        if len(params) == 1:
            motion_params = params[0]
            nuisance_regressors = None
        else:
            motion_params, nuisance_regressors = params
        import numpy as np
        import os
        mp = np.loadtxt(motion_params)
        if nuisance_regressors is not None:
            nr = np.loadtxt(nuisance_regressors)
            mpnr = np.hstack((mp, nr))
        else:
            mpnr = mp
        fn = os.path.abspath('merged_regr.txt')
        np.savetxt(fn, mpnr)
        return fn

    def zipelements(a, b=None):
        if b is None:
            return a
        else:
            return list(zip(a, b))
    zipnode = Node(Function(
        function=zipelements,
        input_names=['a', 'b'],
        output_names=['zipped']),
        name='zip')

    merge_regressors = MapNode(Function(
        function=merge_mat,
        input_names=['params'],
        output_names=['out_file']),
        name='merge_regressors',
        iterfield=['params']
    )

    tproject = MapNode(TProject(
        outputtype='NIFTI_GZ',
        bandpass=(0.00667, 99999)),
        iterfield=['in_file', 'out_file', 'ort'],
        name='tproject')

    wf.connect([
        (inputnode, zipnode,
         [('motion_params', 'a'), 
          ('nuisance_regressors', 'b')]),
        (zipnode, merge_regressors, [('zipped', 'params')]),
        (merge_regressors, tproject, [('out_file', 'ort')]),
        (inputnode, tproject, [('mask', 'mask')]),
        (inputnode, tproject, [('run_file', 'in_file'),
                               (('run_file', basename, 'tp'), 'out_file')]),
    ])
    wf.connect(tproject, 'out_file', outputnode, 'preproc_file')
    wf.connect(censor, 'out_file', outputnode, 'censor')
    return wf


def init_afnimodel_wf(blurfwhm=4.0, do_remlfit=False, TR=1.25,
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
    TR : float
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
        nofdr=True,
        out_file='Decon.' + out_ext,
        force_TR=TR,
    ), name='deconvolve')

    remlfit = Node(Remlfit(
        tout=True,
        nofdr=True,
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


SPACES = {
    'fsaverage6': 'fsaverage6',
    'MNI': 'MNI152NLin2009cAsym',
    'T1w': 'T1w',
    'hpalfsaverage6': 'hpalfsaverage6',
}


def make_afni_stimtimes_category(in_files, key='face'):
    """Given event files, return stim times for AFNI for every exemplar"""
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from os.path import basename, abspath

    def make_dict_onsets(filename):
        events = pd.read_csv(filename, sep='\t')
        onsets = defaultdict(list)
        for row in events.itertuples():
            # skip the first repeated trial
            if row.stim_type == 'face' and row.Index != 0:
                onsets[key].append(row.onset)
        onsets = dict(onsets)
        return onsets

    onsets = dict()
    for f in in_files:
        onsets[f] = make_dict_onsets(f)

    # get stim labels
    stim_labels = sorted(list(onsets[in_files[0]].keys()))

    # create stim files
    stim_files = dict()
    for sl in stim_labels:
        stim_files[sl] = np.array([onsets[f][sl] for f in in_files])
    # save
    fns = []
    for sl in stim_labels:
        fns.append(abspath(sl) + '.txt')
        np.savetxt(fns[-1], stim_files[sl], fmt='%.2f')
    return list(zip(range(1, len(fns) + 1), fns))


def init_afnistim_category_wf(model='GAM', name='afnistim_category_wf'):
    """
    Generate AFNI stims for each individual category (fam/str), across runs.

    Parameters
    ----------
    model: str
        symbolic notation of the model to use (see 3dDeconvolve's help)
    name : str
        name of the workflow

    Returns
    -------
    wf : workflow


    ** Inputs **

        familiar_event_files : list of event files for familiar faces
        stranger_event_files : list of event files for stranger faces

    ** Outputs **

        stim_times : list of stimulus files in AFNI format
        stim_label : list of stim labels
    """

    wf = Workflow(name)
    inputnode = Node(IdentityInterface(fields=['familiar_event_files',
                                               'stranger_event_files']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['stim_times', 'stim_label']),
                      name='outputnode')
    make_afnistim_familiar = Node(
        Function(function=make_afni_stimtimes_category,
                 input_names=['in_files'],
                 output_names=['stim_file']),
        name='make_afnistim_familiar')
    make_afnistim_stranger = Node(
        Function(function=make_afni_stimtimes_category,
                 input_names=['in_files'],
                 output_names=['stim_file']),
        name='make_afnistim_stranger')

    # add model specification
    def _merge_stim_add_model(familiar_stim_file, stranger_stim_file, model):
        import numpy as np
        import os.path as op
        # we need to load both stim files and add '*' to match the number of
        #  runs
        familiar_stim = np.loadtxt(familiar_stim_file[0][1])
        stranger_stim = np.loadtxt(stranger_stim_file[0][1])
        nfamiliar_runs = familiar_stim.shape[0]

        fam_fn = op.abspath('fam_stim.txt')
        str_fn = op.abspath('str_stim.txt')
        np.savetxt(fam_fn, familiar_stim, fmt='%.2f')
        zerostim = '\n'.join(['*']*nfamiliar_runs)
        np.savetxt(str_fn, stranger_stim, header=zerostim, comments='')

        out = []
        for i, f in enumerate([fam_fn, str_fn], 1):
            out.append((i, f, model))
        return out

    merge_stim_add_model = Node(
        Function(function=_merge_stim_add_model,
                 input_names=['familiar_stim_file',
                              'stranger_stim_file',
                              'model'],
                 output_names=['stim_file']),
        name='merge_stim_add_model')
    merge_stim_add_model.inputs.model = model

    # manually add stim labels
    outputnode.inputs.stim_label = [(1, 'familiar'), (2, 'stranger')]

    wf.connect([
        (inputnode, make_afnistim_familiar,
         [('familiar_event_files', 'in_files')]),
        (inputnode, make_afnistim_stranger,
         [('stranger_event_files', 'in_files')]),
        (make_afnistim_familiar, merge_stim_add_model,
         [('stim_file', 'familiar_stim_file')]),
        (make_afnistim_stranger, merge_stim_add_model,
         [('stim_file', 'stranger_stim_file')]),
        (merge_stim_add_model, outputnode,
         [('stim_file', 'stim_times')])
    ])
    return wf


def init_glm_wf(subject, space='fsaverage6', smooth=4.0,
                model='TENT(2.5,17.5,7)', do_remlfit=False,
                name='glm_wf', njobs=1):
    """
    GLM pipeline with TENT functions applying standard GLM analysis.

    Parameters
    ----------
    subject
    space : 'fsaverage6' | 'MNI' | 'T1w' | 'hpalfsaverage6' |
            'hpalsidXXXXfsaverage6'
        which normalized files to compute GLM on
    smooth : float or None
        amount of spatial smoothing
    model : str
        model specification passed on to 3dDeconvolve
    do_remlfit: bool
        whether to run remlfit as well
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
        matrix_img
            png image of the design matrix
        report
            txt files with input files
    """

    #if space not in SPACES:
    #    raise ValueError("I don't know of space {0}, needs to be one "
    #                     "of {1}".format(space, SPACES.keys()))
    space = SPACES.get(space, space)

    wf = Workflow(name=name)

    smooth = 0 if 'fsaverage6' in space else smooth

    inputnode = Node(IdentityInterface(fields=['data_dir',
                                               'fmriprep_dir',
                                               'mask',
                                               'hemifield']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['betas_deconvolve',
                                                'betas_remlfit',
                                                #'betas_niml',
                                                'matrix_img',
                                                'report']),
                      name='outputnode')

    # datasource wf
    datasource_familiar_wf = init_datasource_wf(subject, 'fam1back', space,
                                                name='datasource_familiar')
    datasource_stranger_wf = init_datasource_wf(subject, 'str1back', space,
                                                name='datasource_stranger')
    # setup nodes for nuisance regressors
    motion_fam = init_makeortvec_mapnode(which='motion', name='motion_fam')
    motion_str = init_makeortvec_mapnode(which='motion', name='motion_str')
    motion_merge = Node(Merge(2), name='motion_merge')
    wf.connect(motion_fam, 'ortvec', motion_merge, 'in1')
    wf.connect(motion_str, 'ortvec', motion_merge, 'in2')

    compcor_fam = init_makeortvec_mapnode(which='compcor', name='compcor_fam')
    compcor_str = init_makeortvec_mapnode(which='compcor', name='compcor_str')
    compcor_merge = Node(Merge(2), name='compcor_merge')
    wf.connect(compcor_fam, 'ortvec', compcor_merge, 'in1')
    wf.connect(compcor_str, 'ortvec', compcor_merge, 'in2')

    afnistim_fam_wf = init_afnistim_exemplar_wf(
        name='afnistim_fam',
        model=model,
        prefix='fam')
    afnistim_str_wf = init_afnistim_exemplar_wf(
        name='afnistim_str',
        model=model,
        prefix='str')
    afnistim_time_merge = Node(Merge(2), name='afnistim_time_merge')
    wf.connect(afnistim_fam_wf, 'outputnode.stim_times',
               afnistim_time_merge, 'in1')
    wf.connect(afnistim_str_wf, 'outputnode.stim_times',
               afnistim_time_merge, 'in2')
    afnistim_label_merge = Node(Merge(2), name='afnistim_label_merge')
    wf.connect(afnistim_fam_wf, 'outputnode.stim_label',
               afnistim_label_merge, 'in1')
    wf.connect(afnistim_str_wf, 'outputnode.stim_label',
               afnistim_label_merge, 'in2')

    # make contrast
    def _makecontrast(fam_stim_label, str_stim_label, model):
        """Generate symbolic glt contrasts to be passed"""
        fams = ['+' + lbl for _, lbl in fam_stim_label]
        strs = ['-' + lbl for _, lbl in str_stim_label]
        famstrs = fams + strs
        gltsyms = []
        gltlabels = []
        # fam
        gltsyms.append(' '.join(fams))
        gltlabels.append('fam')
        # str
        gltsyms.append(' '.join(['+' + lbl for _, lbl in str_stim_label]))
        gltlabels.append('str')
        # fam vs str
        gltsyms.append(' '.join(famstrs))
        gltlabels.append('fam_vs_str')
        # subset fam vs str
        if 'TENT' in model:
            # sth of the form 'TENT(2.5,17.5,7)'
            ntents = int(model.split(',')[-1].replace(')', ''))
            subset = '1..{}'.format(ntents - 1)
            gltsyms.append(' '.join([lbl + '[' + subset + ']'
                                     for lbl in famstrs]))
            gltlabels.append('Subfam_vs_str')
            # subset fam vs str per tent
            gltsyms.append(' '.join([lbl + '[[' + subset + ']]'
                                     for lbl in famstrs]))
            gltlabels.append('SubTfam_vs_str')

        gltlabels = list(enumerate(gltlabels, 1))
        return gltsyms, gltlabels

    makecontrast = Node(
        Function(function=_makecontrast,
                 input_names=['fam_stim_label', 'str_stim_label', 'model'],
                 output_names=['gltsym', 'glt_label']),
        name='makecontrast')
    makecontrast.inputs.model = model

    # merge run files
    runfile_merge = Node(Merge(2), name='runfile_merge')
    wf.connect(datasource_familiar_wf, 'outputnode.run_files',
               runfile_merge, 'in1')
    wf.connect(datasource_stranger_wf, 'outputnode.run_files',
               runfile_merge, 'in2')

    # preprocessing
    if 'fsaverage6' in space:
        afnipreproc_wf = init_afnipreprocess_surf_wf()
    else:
        afnipreproc_wf = init_afnipreprocess_wf(blurfwhm=smooth)
    wf.connect([
        (inputnode, afnipreproc_wf, [('mask', 'inputnode.mask')]),
        (runfile_merge, afnipreproc_wf, [('out', 'inputnode.run_file')]),
        (motion_merge, afnipreproc_wf, [('out', 'inputnode.motion_params')]),
        (compcor_merge, afnipreproc_wf,
         [('out', 'inputnode.nuisance_regressors')])
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
    out_ext = 'niml.dset' if 'fsaverage6' in space else 'nii.gz'
    afnimodel_wf = init_afnimodel_wf(blurfwhm=None,
                                     out_ext=out_ext,
                                     do_remlfit=do_remlfit,
                                     njobs=njobs,
                                     name='afnimodel')

    # Add contrasts
    # wf.connect([
    #     (afnistim_fam_wf, makecontrast,
    #      [('outputnode.stim_label', 'fam_stim_label')]),
    #     (afnistim_str_wf, makecontrast,
    #      [('outputnode.stim_label', 'str_stim_label')]),
    #     (makecontrast, afnimodel_wf, [('gltsym', 'inputnode.gltsym'),
    #                                   ('glt_label', 'inputnode.glt_label')])
    # ])

    # XXX: pass these outside
    plotmatrix = init_plotmatrix_node()

    """
    add a report with the files used for the GLM to make sure it's what we want
    """
    report = Node(Function(function=write_filelist_run,
                           input_names=['run_files'],
                                        #'event_files',
                                        #'confound_files'],
                           output_names=['report']),
                  name='report')

    """
    convert giftis to niml dset datasets for later use
    """
    # convert2niml = Node(ConvertDset(), name='convert2niml')
    # convert2niml.inputs.out_type = 'niml_asc'

    def change_ext(in_file):
        import os
        newext = '.niml.dset'
        return os.path.splitext(os.path.basename(in_file))[0] + newext

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
        (inputnode, datasource_familiar_wf,
         [('data_dir', 'inputnode.data_dir'),
          ('fmriprep_dir', 'inputnode.fmriprep_dir'),
          ('hemifield', 'inputnode.hemi')]),
        (inputnode, datasource_stranger_wf,
         [('data_dir', 'inputnode.data_dir'),
          ('fmriprep_dir', 'inputnode.fmriprep_dir'),
          ('hemifield', 'inputnode.hemi')]),
        # make stim times
        (datasource_familiar_wf, afnistim_fam_wf,
         [('outputnode.event_files', 'inputnode.event_files')]),
        (datasource_stranger_wf, afnistim_str_wf,
         [('outputnode.event_files', 'inputnode.event_files')]),
        # pass number of runs before the strangers
        (datasource_familiar_wf, afnistim_str_wf,
         [(('outputnode.event_files', mylen), 'inputnode.start_run')]),
        # make confound matrix / motion
        (datasource_familiar_wf, motion_fam,
         [('outputnode.confound_files', 'in_file')]),
        (datasource_stranger_wf, motion_str,
         [('outputnode.confound_files', 'in_file')]),
        # make confound matrix / compcor
        (datasource_familiar_wf, compcor_fam,
         [('outputnode.confound_files', 'in_file')]),
        (datasource_stranger_wf, compcor_str,
         [('outputnode.confound_files', 'in_file')]),
        # pass preprocessed file to afnimodel
        (afnipreproc_wf, afnimodel_wf,
         [('outputnode.preproc_file', 'inputnode.run_files')]),
        # pass stimulus times
        (afnistim_time_merge, afnimodel_wf,
         [(('out', enumerate_tuples), 'inputnode.stim_times')]),
        (afnistim_label_merge, afnimodel_wf,
         [(('out', enumerate_tuples), 'inputnode.stim_label')]),
        (inputnode, afnimodel_wf, [('mask', 'inputnode.mask')]),
        (censor_stack, afnimodel_wf,
         [('out_file', 'inputnode.censor')]),
        # plot design matrix
        (afnimodel_wf, plotmatrix, [('outputnode.matrix', 'matrix')]),
        # save report
        (afnipreproc_wf, report, [('outputnode.preproc_file', 'run_files')]),
        # pass output
        (afnimodel_wf, outputnode, [
            ('outputnode.betas_deconvolve', 'betas_deconvolve'),
            ('outputnode.betas_remlfit', 'betas_remlfit')]),
        (plotmatrix, outputnode, [('matrix_img', 'matrix_img')]),
        (report, outputnode, [('report', 'report')])
    ])

    # convert to niml if using surfaces
    #if space == 'fsaverage6':
    #    wf.connect([
    #        # convert to niml
    #        (afnimodel_wf, convert2niml, [
    #            ('outputnode.betas_remlfit', 'in_file'),
    #            (('outputnode.betas_remlfit', change_ext), 'out_file')]),
    #        # save output
    #        (convert2niml, outputnode, [('out_file', 'betas_niml')]),
    #    ])

    return wf


def init_glmrun_wf(subject, task='fam1back', space='fsaverage6', smooth=None,
                   model='BLOCK(1.6,1)', do_remlfit=False, per_run=True,
                   TR=1.25, name='glmrun_wf', njobs=1):
    """
    GLM pipeline with BLOCK function applying standard GLM analysis
    estimating betas within each run separately. It differs from
    `init_glm_wf` in the following ways:

        1. It computes betas within each run (instead of across the entire
        experiment) using a standard gamma HRF (one beta for each stimulus)
        2. It computes the GLM within familiar / stranger stimuli only. This is
        done to reduce runtime since a giganormous matrix needs to be inverted
        otherwise.

    Parameters
    ----------
    subject
    task: 'fam1back' | 'str1back'
        which task to run
    space : 'fsaverage6' | 'MNI' | 'T1w' | 'hpalfsaverage6' |
            'hpalsidXXXXfsaverage6'
        which normalized files to compute GLM on
    smooth : float or None
        amount of spatial smoothing
    model : str
        model specification passed on to 3dDeconvolve
    do_remlfit: bool
        whether to run remlfit as well
    per_run: bool
        if True, betas are estimated within each run; otherwise, across all
        runs
    TR : float
        repetition time
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
        matrix_img
            png image of the design matrix
        report
            txt files with input files
    """
    space = SPACES.get(space, space)

    wf = Workflow(name=name)

    smooth = 0 if 'fsaverage' in space else smooth

    inputnode = Node(IdentityInterface(fields=['data_dir',
                                               'fmriprep_dir',
                                               'mask',
                                               'hemifield']),
                     name='inputnode')
    outputnode = Node(IdentityInterface(fields=['betas_deconvolve',
                                                'betas_remlfit',
                                                #'betas_niml',
                                                'matrix_img',
                                                'report']),
                      name='outputnode')

    # datasource wf
    datasource_wf = init_datasource_wf(subject, task, space, name='datasource')
    # setup nodes for nuisance regressors
    motion = init_makeortvec_mapnode(which='motion', name='motion')
    compcor = init_makeortvec_mapnode(which='compcor', name='compcor')

    # get stim times for each run
    afnistim_wf = init_afnistim_exemplar_wf(
        name='afnistim',
        model=model,
        prefix=task[:3],
        per_run=per_run,
        tr=TR)

    # preprocessing
    if 'fsaverage' in space:
        afnipreproc_wf = init_afnipreprocess_surf_wf()
    else:
        afnipreproc_wf = init_afnipreprocess_wf(blurfwhm=smooth)
    # XXX: if we're using hpal data, it's likely that we are inputting files
    # as niml.dset. so we need to convert them to gifti or tproject doesn't
    # work
    if 'hpal' in space:
        convert2gii = MapNode(ConvertDset(),
                          iterfield=['in_file', 'out_file'],
                          name='convert')
        convert2gii.inputs.out_type = 'gii'

        def rename(listfiles):
            import os.path as op
            return [op.basename(l.replace('niml.dset', 'gii')) for l in listfiles]

        wf.connect([
            (datasource_wf, convert2gii,
             [('outputnode.run_files', 'in_file'),
              (('outputnode.run_files', rename), 'out_file')]),
            (convert2gii, afnipreproc_wf, [
                ('out_file', 'inputnode.run_file')])
        ])
    else:
        wf.connect([
            (datasource_wf, afnipreproc_wf, [('outputnode.run_files',
                                              'inputnode.run_file')])
        ])

    wf.connect([
        (inputnode, afnipreproc_wf, [('mask', 'inputnode.mask')]),
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
    out_ext = 'niml.dset' if 'fsaverage' in space else 'nii.gz'
    afnimodel_wf = init_afnimodel_wf(blurfwhm=None,
                                     out_ext=out_ext,
                                     do_remlfit=do_remlfit,
                                     njobs=njobs,
                                     TR=TR,
                                     name='afnimodel')

    """
    add a report with the files used for the GLM to make sure it's what we want
    """
    report = Node(Function(function=write_filelist_run,
                           input_names=['run_files'],
                           #'event_files',
                           #'confound_files'],
                           output_names=['report']),
                  name='report')

    # Change extension so we get niml.dset files after deconvolve
    def change_ext(in_file):
        import os
        newext = '.niml.dset'
        return os.path.splitext(os.path.basename(in_file))[0] + newext

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
        # save report
        (afnipreproc_wf, report, [('outputnode.preproc_file', 'run_files')]),
        # pass output
        (afnimodel_wf, outputnode, [
            ('outputnode.betas_deconvolve', 'betas_deconvolve'),
            ('outputnode.betas_remlfit', 'betas_remlfit')]),
        (report, outputnode, [('report', 'report')])
    ])

    return wf


def init_plotmatrix_node():
    """
    plot design matrix
    """

    def plot_matrix(matrix):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
        import numpy as np
        import os

        mat = np.loadtxt(matrix)
        mat_ = np.apply_along_axis(rankdata, 0, mat) / len(mat)
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        ax.matshow(mat_.T)
        fn = os.path.abspath('matrix.png')
        plt.tight_layout()
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        return fn

    plotmatrix = Node(Function(function=plot_matrix,
                               input_names=['matrix'],
                               output_names=['matrix_img']),
                      name='plotmatrix')
    return plotmatrix


def init_glm_localizer_wf(subject, space='fsaverage6', smooth=None,
                          model='BLOCK(1.5,1)',
                          name='glmlocalizer_wf', njobs=1):
    """
    GLM pipeline for localizer data.

    Parameters
    ----------
    subject
    space : 'fsaverage6' | 'MNI' | 'T1w'
        which normalized files to compute GLM on
    smooth : float or None
        amount of spatial smoothing
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
        matrix_img
            png image of the design matrix
        report
            txt files with input files
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
                                                'betas_niml',
                                                'matrix_img',
                                                'report']),
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
    if 'fsaverage6' in space:
        afnipreproc_wf = init_afnipreprocess_surf_wf()
    else:
        afnipreproc_wf = init_afnipreprocess_wf(blurfwhm=smooth)
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
    out_ext = 'niml.dset' if 'fsaverage6' in space else 'nii.gz'
    afnimodel_wf = init_afnimodel_wf(blurfwhm=None,
                                     out_ext=out_ext,
                                     njobs=njobs,
                                     do_remlfit=True,
                                     name='afnimodel')

    # Add contrasts
    afnimodel_wf.inputs.inputnode.gltsym = ['+face -object']
    afnimodel_wf.inputs.inputnode.glt_label = [(1, 'face_vs_object')]

    # XXX: pass these outside
    plotmatrix = init_plotmatrix_node()

    """
    add a report with the files used for the GLM to make sure it's what we want
    """
    report = Node(Function(function=write_filelist_run,
                           input_names=['run_files'],
                           #'event_files',
                           #'confound_files'],
                           output_names=['report']),
                  name='report')

    """
    convert giftis to niml dset datasets for later use
    """
    #convert2niml = Node(ConvertDset(), name='convert2niml')
    #convert2niml.inputs.out_type = 'niml_asc'

    #def change_ext(in_file):
    #    import os
    #    newext = '.niml.dset'
    #    return os.path.splitext(os.path.basename(in_file))[0] + newext

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
        # plot design matrix
        (afnimodel_wf, plotmatrix, [('outputnode.matrix', 'matrix')]),
        # save report
        (afnipreproc_wf, report, [('outputnode.preproc_file', 'run_files')]),
        # pass output
        (afnimodel_wf, outputnode, [
            ('outputnode.betas_deconvolve', 'betas_deconvolve'),
            ('outputnode.betas_remlfit', 'betas_remlfit')]),
        (plotmatrix, outputnode, [('matrix_img', 'matrix_img')]),
        (report, outputnode, [('report', 'report')])
    ])

    # convert to niml if using surfaces
    # if space == 'fsaverage6':
    #     wf.connect([
    #         # convert to niml
    #         (afnimodel_wf, convert2niml, [
    #             ('outputnode.betas_remlfit', 'in_file'),
    #             (('outputnode.betas_remlfit', change_ext), 'out_file')]),
    #         # save output
    #         (convert2niml, outputnode, [('out_file', 'betas_niml')]),
    #
    #     ])

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

    # get stim labels
    stim_labels = sorted(list(onsets[in_files[0]].keys()))

    # create stim files
    stim_files = dict()
    for sl in stim_labels:
        stim_files[sl] = [onsets[f][sl] for f in in_files]

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
