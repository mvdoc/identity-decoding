#!/bin/bash -ex

# This script runs the GLM using BLOCK function with betas estimated within
# each run.

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/glm)
DATADIR=$BASEDIR/data
FMRIPREPDIR=$BASEDIR/derivatives/fmriprep
PYTHON_WRAP=$BASEDIR/scripts/python36
NCORES=16

# $1 is subject
# $2 is space
# $3 is task
# $4 is smoothing, if any

if [ ! -z "$4" ]; then
  CMD="--smooth $4"
  SFX="$4"mm
else
  SFX="nosmooth"
fi

MODEL="'BLOCK(1.6,1)'"
MODELSFX='blockrun'
OUTDIR=$BASEDIR/derivatives/glm-"$MODELSFX"
WORKDIR="$WORKDIR"-"$MODELSFX"-"$2"-"$SFX"
OUTDIR="$OUTDIR"-"$SFX"
TMPDIR=$WORKDIR/tmp

if [ ! -d "$WORKDIR" ]; then
   echo "Creating $WORKDIR"
   mkdir -p "$WORKDIR"
fi

if [ ! -d "$TMPDIR" ]; then
   echo "Creating $TMPDIR"
   mkdir -p "$TMPDIR"
fi

if [ ! -d "$OUTDIR" ]; then
   echo "Creating $OUTDIR"
   mkdir -p "$OUTDIR"
fi

MASK=$BASEDIR/derivatives/unionmask/sub-"$1"/sub-"$1"_space-"$2"_unionbrainmask.nii.gz

if [ -f "$MASK" ] && [ "$2" != "fsaverage6" ] ; then
  echo "Using mask $MASK"
  CMD="$CMD --mask $MASK"
fi

CMD="$BASEDIR/scripts/01glm/run_glm_model.py \
       -d $DATADIR \
       -f $FMRIPREPDIR \
       -w $WORKDIR \
       -o $OUTDIR \
       -s $1 \
       -p $2 \
       --task $3 \
       --estimate-within-run \
       -n $NCORES \
       --model $MODEL \
       $CMD"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -e \
  "$IMG" \
  "$PYTHON_WRAP $CMD"
