#!/bin/bash -ex
# THIS SCRIPT RUNS BETWEEN SUBJECT DECODING
# ON DATA THAT HAS BEEN ONLY ANATOMICALLY ALIGNED

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
# BASEDIR=/data/famface_angles

IMG=$BASEDIR/singularity/$IMGNAME
FMRIPREPDIR=$BASEDIR/derivatives103-budapest/fmriprep
FSDIR=$BASEDIR/derivatives103-budapest/freesurfer
GLMDIR=$BASEDIR/derivatives103-budapest/glm-blockrun-nosmooth
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=16

programname=$0
function usage {
    echo "usage: $programname target task hemi [deconvolve|remlfit] [permute]"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TARG=$1
TASK=$2
HEMI=$3
DECON=$4
PERMUTE=$5

if [ $HEMI == L ]; then
  H=lh
else
  H=rh
fi

if [ -z "$DECON" ]; then
    DECON=deconvolve
fi

OUTDIR=$BASEDIR/derivatives/slclfbwsbj-blockrun-"$DECON"-fsaverage6

INPUT="$GLMDIR"/sub-*/sub-*_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_"$DECON"-block.niml.dset
OUTPUT="$OUTDIR"/bwsbj_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_target-"$TARG"_sl

if [ ! -z $PERMUTE ]; then
    printf -v P "%03d" $PERMUTE
    OUTPUT="$OUTPUT"_"$P"ip
    CMD="--permute $PERMUTE"
fi
OUTPUT="$OUTPUT".niml.dset

CMD="$BASEDIR/scripts/02mvpa/run_sl_bwsbj.py \
     -i $INPUT \
     -t $TARG \
     -o $OUTPUT \
     -n $NCORES \
     $CMD"


singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$PYTHONWRAP" "$CMD"
