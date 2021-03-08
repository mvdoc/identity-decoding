"""Module containing custom nipype interfaces"""
import os
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine,
                                    Directory, TraitedSpec,
                                    traits, isdefined, File, InputMultiPath,
                                    Undefined, Str)
from nipype.interfaces.afni.base import (AFNIPythonCommandInputSpec,
                                         AFNIPythonCommand,
                                         AFNICommandInputSpec,
                                         AFNICommandOutputSpec)


class TimingToolPyInputSpec(AFNIPythonCommandInputSpec):
    in_file = File(
        desc='specify a stimulus timing file to load',
        exist=True,
        argstr='-timing %s')
    tr = traits.Float(
        desc='specify the time resolution in 1D output (in seconds)',
        argstr='-tr %f')
    truncate_times = traits.Bool(
        desc="""truncate times to multiples of the TR
        All stimulus times will be truncated to the largest multiple of the TR
        that is less than or equal to each respective time.  That is to say,
        shift each stimulus time to the beginning of its TR.

        This is particularly important when stimulus times are at a constant
        offset into each TR and at the same time using TENT basis functions
        for regression (in 3dDeconvolve, say).  The shorter the (non-zero)
        offset, the more correlated the first two tent regressors will be,
        possibly leading to unpredictable results.
        
        This option requires -tr.""",
        argstr='-truncate_times')
    out_file = File(
        desc='newly generated stimulus time',
        argstr='-write_timing %s',
        position=-1)


class TimingToolPyOutputSpec(AFNICommandOutputSpec):
    out_file = File(desc='newly generated stimulus time')


class TimingToolPy(AFNIPythonCommand):
    """
     This program is used for manipulating and evaluating stimulus timing files
    """

    _cmd = 'timing_tool.py'

    input_spec = TimingToolPyInputSpec
    output_spec = TimingToolPyOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.join(os.getcwd(),
                                               self.inputs.out_file)
        return outputs

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if not isdefined(self.inputs.out_file):
            self.inputs.out_file = 'tr_stimtime.txt'

        return super(TimingToolPy, self)._parse_inputs(skip)
