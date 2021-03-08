#!/usr/bin/env python
"""Script to run GLM for hyperface between subject (after hpal)"""
import argparse
from nipype import config, logging
from famfaceangles.workflows import init_glm_localizer_wf
from nipype.pipeline import Workflow, Node
from nipype.interfaces import IdentityInterface, DataSink
from os.path import abspath
from workflows import init_glm_localizer_nodatasource_wf

# set debug mode
config.set('execution', 'crashfile_format', 'txt')


def setup_workflow(run_files, event_files, confound_files,
                   work_dir, output_dir, output_template,
                   njobs=1):
    """
    Setup workflow for glm

    Parameters
    ----------
    run_files
    event_files
    confound_files
    work_dir
    output_dir
    njobs

    Returns
    -------
    wf
    """
    # use absolute paths
    output_dir = abspath(output_dir)

    # setup workflow
    wf_name = 'glm-{0}-bwsj'.format('localizer')
    wf = Workflow(name=wf_name)

    # glm workflow
    glm_wf = init_glm_localizer_nodatasource_wf(njobs=njobs)

    # setup inputnode
    inputnode = Node(IdentityInterface(
        fields=['run_files', 'event_files', 'confound_files']),
        name='inputnode')
    inputnode.inputs.run_files = run_files
    inputnode.inputs.event_files = event_files
    inputnode.inputs.confound_files = confound_files

    # datasink
    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = 'bwsj'

    # base_fn = 'sub-{0}_task-{1}_'.format(subject, 'localizer')
    datasink.inputs.substitutions = [
        ('matrix.png', 'glm_matrix.png'),
        ('Decon', 'deconvolve-block'),
        ('Remlfit', 'remlfit-block'),
        ('_hemifield_L/', 'hemi-L_'),
        ('_hemifield_R/', 'hemi-R_'),
        ('_workdirs/', output_template),
    ]

    wf.base_dir = work_dir
    wf.connect([
        (inputnode, glm_wf, [('run_files', 'inputnode.run_files'),
                             ('event_files', 'inputnode.event_files'),
                             ('confound_files', 'inputnode.confound_files')]),
        (glm_wf, datasink, [('outputnode.betas_deconvolve',
                             '@betas_deconvolve'),
                            ('outputnode.betas_remlfit', '@betas_remlfit'),
                            ('outputnode.betas_niml', '@betas_niml'),
                            ])
    ])
    return wf


def main():
    parsed = parse_args()
    njobs = parsed.njobs
    rf = parsed.run_files
    ef = parsed.event_files
    cf = parsed.confound_files
    lens = set((len(rf), len(ef), len(cf)))
    if len(lens) != 1:
        raise ValueError("Got different number of files")

    wf = setup_workflow(run_files=parsed.run_files,
                        event_files=parsed.event_files,
                        confound_files=parsed.confound_files,
                        work_dir=parsed.work_dir,
                        output_dir=parsed.output_dir,
                        output_template=parsed.output_template,
                        njobs=njobs)
    if njobs == 1:
        wf.run()
    else:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': njobs})


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mask',  type=str,
                        # help='mask to use',
                        # required=False, default=None)
    # parser.add_argument('--model',  type=str,
    #                     help='AFNI model specification',
    #                     required=False, default='TENT(2.5,17,7)')
    parser.add_argument('--run-files', '-r', type=str, nargs='+',
                        help='run files (need to match order of event files '
                             'and confound files)',
                        required=True)
    parser.add_argument('--event-files', '-e', type=str, nargs='+',
                        help='event files (need to match order of run files '
                             'and confound files)',
                        required=True)
    parser.add_argument('--confound-files', '-c', type=str, nargs='+',
                        help='confound files (need to match order of run '
                             'files and event files)',
                        required=True)
    parser.add_argument('--work-dir', '-w', type=str,
                        help='working directory',
                        required=True)
    parser.add_argument('--output-dir', '-o', type=str,
                        help='output directory',
                        required=True)
    parser.add_argument('--output-template', type=str,
                        help='output template filename to save the results',
                        required=True)
    parser.add_argument('--njobs', '-n', type=int,
                        help='number of jobs',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
