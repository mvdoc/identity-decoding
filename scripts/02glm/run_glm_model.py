#!/usr/bin/env python
"""Script to run GLM for famfaceangles"""
import argparse
from nipype import config, logging
from famfaceangles.workflows import init_glm_wf, init_glmrun_wf
from nipype.pipeline import Workflow, Node
from nipype.interfaces import IdentityInterface, DataSink
from os.path import abspath

# set debug mode
config.set('execution', 'crashfile_format', 'txt')
config.enable_debug_mode()


def setup_workflow(subject, space, smooth, mask, model,
                   data_dir, fmriprep_dir, work_dir, output_dir,
                   task=None, estimate_within_run=False, njobs=1,
                   TR=1.25):
    """
    Setup workflow for glm

    Parameters
    ----------
    subject
    space
    smooth
    mask
    model
    data_dir
    fmriprep_dir
    work_dir
    output_dir
    task
    estimate_within_run
    njobs
    TR

    Returns
    -------
    wf
    """
    # use absolute paths
    data_dir = abspath(data_dir)
    fmriprep_dir = abspath(fmriprep_dir)
    work_dir = abspath(work_dir)
    output_dir = abspath(output_dir)

    # glm workflow
    if estimate_within_run:
        glm_wf = init_glmrun_wf(subject, space=space, smooth=smooth, task=task,
                                model=model, do_remlfit=True, TR=TR, njobs=njobs)
    else:
        glm_wf = init_glm_wf(subject, space=space, smooth=smooth,
                             model=model, njobs=njobs)
        task = 'both'

    # setup workflow
    glmtype = model.split('(')[0].lower()
    wf_name = 'glm{0}-{1}-{2}'.format(glmtype, subject, task)
    wf = Workflow(name=wf_name)

    # setup inputnode
    inputnode = Node(IdentityInterface(
        fields=['data_dir', 'fmriprep_dir', 'mask', 'hemifield']),
                     name='inputnode')
    if 'fsaverage6' in space:
        inputnode.iterables = ('hemifield', ('L', 'R'))
    inputnode.inputs.data_dir = data_dir
    inputnode.inputs.fmriprep_dir = fmriprep_dir

    # pass mask
    if mask is not None:
        inputnode.inputs.mask = mask
        wf.connect(inputnode, 'mask', glm_wf, 'inputnode.mask')

    # datasink
    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = 'sub-{0}'.format(subject)

    base_fn = 'sub-{0}_task-{1}_'.format(subject, task)
    datasink.inputs.substitutions = [
        ('matrix.png', 'glm-{}_matrix.png'.format(glmtype)),
        ('Decon', 'deconvolve-' + glmtype),
        ('Remlfit', 'remlfit-' + glmtype),
        ('_hemifield_L/', 'hemi-L_'),
        ('_hemifield_R/', 'hemi-R_'),
        ('_workdirs/', base_fn + 'space-{0}_'.format(space)),
        ('_workdir_hydra_deleteme/', base_fn + 'space-{0}_'.format(space))
    ]

    wf.base_dir = work_dir
    wf.connect([
        (inputnode, glm_wf, [('data_dir', 'inputnode.data_dir'),
                             ('fmriprep_dir', 'inputnode.fmriprep_dir'),
                             ('hemifield', 'inputnode.hemifield')]),
        (glm_wf, datasink, [('outputnode.betas_deconvolve',
                             '@betas_deconvolve'),
                            ('outputnode.betas_remlfit', '@betas_remlfit'),
                            #('outputnode.betas_niml', '@betas_niml'),
                            ('outputnode.matrix_img', '@matrix_img'),
                            ('outputnode.report', '@report')])
    ])
    return wf


def check_parse(p):
    # check for task and estimate-within-run
    if p.estimate_within_run and p.task is None:
        raise ValueError("You need to specify a task when "
                         "--estimate-within-run is set")


def main():
    parsed = parse_args()
    check_parse(parsed)
    njobs = parsed.njobs
    wf = setup_workflow(subject=parsed.subject,
                        space=parsed.space,
                        smooth=parsed.smooth,
                        mask=parsed.mask,
                        model=parsed.model,
                        data_dir=parsed.data_dir,
                        fmriprep_dir=parsed.fmriprep_dir,
                        work_dir=parsed.work_dir,
                        output_dir=parsed.output_dir,
                        task=parsed.task,
                        estimate_within_run=parsed.estimate_within_run,
                        njobs=njobs,
                        TR=parsed.tr)
    if njobs == 1:
        wf.run()
    else:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': njobs})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', '-s', type=str,
                        help='subject id (without sid)',
                        required=True)
    parser.add_argument('--space', '-p', type=str,
                        help='target space',
                        required=False, choices=['fsaverage6', 'MNI', 'T1w',
                                                 'hpalfsaverage6',
                                                 'hpalsid000005fsaverage6'],
                        default='fsaverage6')
    parser.add_argument('--smooth',  type=float,
                        help='amount of smoothing to perform prior to GLM ('
                             'default None)',
                        required=False, default=None)
    parser.add_argument('--mask',  type=str,
                        help='mask to use',
                        required=False, default=None)
    parser.add_argument('--model',  type=str,
                        help='AFNI model specification',
                        required=False, default='TENT(2.5,17,7)')
    parser.add_argument('--task',  type=str,
                        help='task to process. required only if '
                             '--estimate-within-run is set. Either fam1back '
                             'or str1back',
                        required=False, default=None,
                        choices=('fam1back', 'str1back', 'identity'))
    parser.add_argument('--estimate-within-run',
                        help='If this flag is set, then the betas are '
                             'estimated within each run, instead of across '
                             'runs. If set, it requires to specify the task '
                             'with the --task flag.',
                        action='store_true', default=False)
    parser.add_argument('--data-dir', '-d', type=str,
                        help='data directory in BIDS format',
                        required=True)
    parser.add_argument('--fmriprep-dir', '-f', type=str,
                        help='fmriprep directory (contains sub-*)',
                        required=True)
    parser.add_argument('--work-dir', '-w', type=str,
                        help='working directory',
                        required=True)
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory',
                        required=True)
    parser.add_argument('--njobs', '-n', type=int,
                        help='number of jobs',
                        required=True)
    parser.add_argument('--tr', type=float,
                        help='repetition time',
                        required=False, default=1.25)
    return parser.parse_args()


if __name__ == '__main__':
    main()
