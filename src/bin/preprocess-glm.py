#!/usr/bin/env python
"""Quick standardized preprocessing prior to GLM."""

import argparse
import os.path as op
from nipype.pipeline import Workflow, Node
from nipype.interfaces import DataSink
from famfaceangles.workflows import init_afnipreprocess_surf_wf, \
    init_afnipreprocess_wf, init_makeortvec_mapnode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--nproc', '-n', type=int,
                        help='number of procs to use',
                        default=1)
    parser.add_argument('--inputs', '-i', type=str, nargs='+',
                        help='input files')
    parser.add_argument('--confounds', '-c', type=str, nargs='+',
                        help='confounds files')
    parser.add_argument('outdir', type=str,
                        help='output directory')

    return parser.parse_args()


def is_surface(filename):
    return True if filename.endswith(("gii", "niml.dset")) else False


def get_ext(file):
    if "niml.dset" in file:
        return "niml.dset"
    elif "nii.gz" in file:
        return "nii.gz"
    else:
        return file.split('.')[-1]


def setup_workflow(infile, confounds, outdir):
    wf = Workflow("preprocess")
    preproc_wf = init_afnipreprocess_surf_wf() if is_surface(infile) \
        else init_afnipreprocess_wf()
    motion = init_makeortvec_mapnode('motion', name='motion')
    compcor = init_makeortvec_mapnode('compcor', name='compcor')

    motion.inputs.in_file = confounds
    compcor.inputs.in_file = confounds
    preproc_wf.inputs.inputnode.run_file = infile

    infile_ext = get_ext(infile[0])

    sinker = Node(DataSink(), name='sinker')
    sinker.inputs.base_directory = op.abspath(outdir)
    subs = [('_tproject{}/'.format(i), '') for i in range(len(infile))]
    subs += [
        ('_censor{}/'.format(i), infile[i].replace(infile_ext, '_censor.1D'))
        for i in range(len(infile))]
    sinker.inputs.substitutions = subs

    wf.connect([
        (motion, preproc_wf, [('ortvec', 'inputnode.motion_params')]),
        (compcor, preproc_wf, [('ortvec', 'inputnode.nuisance_regressors')]),
        (preproc_wf, sinker, [('outputnode.preproc_file', '@preproc'),
                              ('outputnode.censor', '@censor')])
    ])
    return wf


def main():
    p = parse_args()
    wf = setup_workflow(p.input, p.confounds, p.outdir)
    nproc = p.nproc
    if nproc > 1:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': nproc})
    else:
        wf.run()


if __name__ == '__main__':
    main()
