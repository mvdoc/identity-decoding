#!/bin/bash -ex

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/glm)
DATADIR=$BASEDIR/data
FMRIPREPDIR=$BASEDIR/derivatives/fmriprep
PYTHON_WRAP=$BASEDIR/scripts/python36
NCORES=24

# $1 is subject
# $2 is space
# $3 is smoothing, if any

SUBJ=$1
SPACE=$2
SMOOTH=$3

if [ ! -z "$3" ]; then
  CMD="--smooth $3"
  SFX="$3"mm
else
  SFX="nosmooth"
fi

MODELSFX='localizer'
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

if [ $SPACE != fsaverage6 ]; then
    MASK=$BASEDIR/derivatives/unionmask/sub-"$1"/sub-"$1"_space-"$2"_unionbrainmask.nii.gz
fi

if [ -f "$MASK" ]; then
  echo "Using mask $MASK"
  CMD="$CMD --mask $MASK"
fi

CMD="$BASEDIR/scripts/01glm/run_glm_localizer.py \
       -d $DATADIR \
       -f $FMRIPREPDIR \
       -w $WORKDIR \
       -o $OUTDIR \
       -s $1 \
       -p $2 \
       -n $NCORES \
       $CMD"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -e \
  "$IMG" \
  "$PYTHON_WRAP $CMD"
