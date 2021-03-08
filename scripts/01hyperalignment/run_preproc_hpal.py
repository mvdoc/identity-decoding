#!/usr/bin/env python
"""Script to preprocess HPAL data"""
import argparse
from famfaceangles.workflows import bidsgrabber, init_makeortvec_mapnode
from nipype import config
from nipype import Node, Workflow, Function, MapNode
from nipype.interfaces import IdentityInterface
from nipype.interfaces.io import DataSink
from nipype.interfaces.afni import TProject
import os.path as op
config.set('execution', 'remove_unnecessary_outputs', 'false')


def init_wf(subject, task, fmriprep_dir, output_dir, space,
            bandpass=(0.00667, 0.1), name='hpalpreprocess'):
    wf = Workflow(name)

    space2ext = {
        'T1w': '.nii.gz',
        'fsaverage6': '.gii'
    }
    grab_data = Node(
        Function(function=bidsgrabber,
                 input_names=['base_dir', 'query'],
                 output_names=['files']), name='grab_data')
    grab_data.inputs.base_dir = fmriprep_dir
    query = dict(modality='func', extensions=space2ext[space],
                 task=task, space=space, subject=subject)
    if space == 'T1w':
        query['type'] = 'preproc'
    grab_data.inputs.query = query

    def get_confounds(runfiles):
        import re
        m = re.compile(r'space-.*')
        confoundfiles = [
            m.sub('confounds.tsv', f) for f in runfiles
        ]
        return confoundfiles
    grab_conf = Node(
        Function(function=get_confounds,
                 input_names=['runfiles'],
                 output_names=['files']), name='grab_conf')
    #grab_conf = Node(
    #    Function(function=bidsgrabber,
    #             input_names=['base_dir', 'query'],
    #             output_names=['files']), name='grab_conf')
    #grab_conf.inputs.base_dir = fmriprep_dir
    #grab_conf.inputs.query = dict(
    #    modality='func', type='confounds',
    #    extensions='.tsv', task=task, subject=subject)

    makeortvec = init_makeortvec_mapnode(which='all')
    tproject = MapNode(TProject(
        outputtype='NIFTI_GZ',
        bandpass=bandpass),
        iterfield=['in_file', 'out_file', 'ort'],
        name='tproject')

    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = 'sub-{0}'.format(subject)
    datasink.inputs.regexp_substitutions = [
        ('_workdirs/_tproject[0-9]*', '')
    ]

    def basename(listfn, sfx='', change_ext=''):
        def _basename(fn):
            import os
            ext = 'nii.gz' if fn.endswith('nii.gz') else 'gii'
            fn = fn.replace(ext, '')
            fn = os.path.basename(fn)
            if sfx:
                fn += sfx
            if change_ext:
                ext = change_ext
            return os.path.basename(fn) + '.' + ext

        listfn = [_basename(f) for f in listfn]
        return listfn

    newext = 'niml.dset' if space == 'fsaverage6' else ''
    wf.connect([
        (grab_data, grab_conf, [('files', 'runfiles')]),
        (grab_conf, makeortvec, [('files', 'in_file')]),
        (makeortvec, tproject, [('ortvec', 'ort')]),
        (grab_data, tproject, [('files', 'in_file'),
                               (('files', basename, 'tp', newext),
                                'out_file')]),
        (tproject, datasink, [('out_file', '@tprojected')])
    ])
    return wf


def run(subject, task, space, fmriprep_dir, output_dir,
        work_dir, bandpass=(0.00667, 0.1), nproc=1):
    wf = init_wf(subject=subject,
                 task=task,
                 space=space,
                 fmriprep_dir=fmriprep_dir,
                 output_dir=output_dir,
                 bandpass=bandpass)
    wf.base_dir = op.join(work_dir, 'sub-{}'.format(subject))

    if nproc == 1:
        wf.run()
    else:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': nproc})


def main():
    p = parse_args()
    run(subject=p.subject,
        task=p.task,
        space=p.space,
        fmriprep_dir=op.abspath(p.fmriprep_dir),
        output_dir=op.abspath(p.output_dir),
        work_dir=op.abspath(p.work_dir),
        nproc=p.njobs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', '-s', type=str,
                        help='subject id (without sid)',
                        required=True)
    parser.add_argument('--task', '-t', type=str,
                        help='task',
                        required=True, choices=['movie'])
    parser.add_argument('--space', type=str,
                        help='space',
                        default='T1w',
                        required=False, choices=['T1w', 'fsaverage6'])
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
    return parser.parse_args()


if __name__ == '__main__':
    main()
