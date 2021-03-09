#!/bin/bash -e

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
FMRIPREPDIR=$BASEDIR/derivatives/fmriprep
FSDIR=$BASEDIR/derivatives/freesurfer
GLMDIR=$BASEDIR/derivatives/glm-blockrun-nosmooth
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=16

programname=$0
function usage {
    echo "usage: $programname subjectid target task hemi [deconvolve|remlfit]"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

SID=$1
TARG=$2
TASK=$3
HEMI=$4
DECON=$5

if [ $HEMI == L ]; then
  H=lh
else
  H=rh
fi

if [ -z "$DECON" ]; then
    DECON=deconvolve
fi
OUTDIR=$BASEDIR/derivatives/slclf-blockrun-"$DECON"-fsaverage6

INPUT="$GLMDIR"/sub-"$SID"/sub-"$SID"_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_"$DECON"-block.niml.dset
MASK="$FSDIR"/fsaverage6/SUMA/"$H".maskmedial.niml.dset

CMD="$BASEDIR/scripts/02mvpa/run_sl.py \
     -i $INPUT \
     -t $TARG \
     -f $FSDIR \
     --mask $MASK \
     -o $OUTDIR \
     -n $NCORES"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -e \
  "$IMG" \
  "$PYTHON_WRAP" "$CMD"
