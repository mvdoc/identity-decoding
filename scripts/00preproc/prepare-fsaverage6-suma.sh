#!/bin/bash -ex

IMGNAME="poldracklab_fmriprep_1.0.3-2018-01-04-8c1973f461fd.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
FS_LICENSE=$BASEDIR/scripts/00preproc/license.txt

SUBJ=fsaverage6
FSPATH=$BASEDIR/derivatives/freesurfer/$SUBJ
export FS_LICENSE

if [ ! -d "$FSPATH/mri/orig" ]; then
  mkdir -p "$FSPATH/mri/orig"
fi

singularity exec \
  -B /idata \
  -c \
  "$IMG" \
  xvfb-run @SUMA_Make_Spec_FS -NIFTI -no_ld \
    -inflate 10 -inflate 20 \
    -inflate 30 -inflate 40 -inflate 50 -inflate 60 -inflate 70 \
    -inflate 80 -inflate 90 -inflate 100 -inflate 120 -inflate 150 \
    -inflate 200 \
    -sid $SUBJ -fspath $FSPATH
