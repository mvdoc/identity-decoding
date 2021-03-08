#!/bin/bash

FSDIR=../../derivatives/freesurfer/fsaverage6/SUMA
HERE=$PWD
NODE=1644825

cd $FSDIR
for HEMI in lh rh; do
  3dcalc -a "$HEMI".aparc.a2009s.annot.niml.dset -expr 1-equals\(a,"$NODE"\) -prefix "$HEMI".maskmedial.niml.dset
done
cd $HERE
