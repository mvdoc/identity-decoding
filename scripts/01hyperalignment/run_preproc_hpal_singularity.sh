#!/bin/bash -ex
IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/hpalpreproc-budapest)
TMPDIR=$WORKDIR/tmp
DATADIR=$BASEDIR/data
OUTDIR=$BASEDIR/derivatives103-budapest/hpal-preproc
FS_LICENSE=$BASEDIR/scripts/00preproc/license.txt
PYTHONWRAP=$BASEDIR/scripts/python36

NCORES=8

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

CMD="$BASEDIR/scripts/00preproc/run_preproc_hpal.py \
    -s $1 --task movie \
    --fmriprep-dir $BASEDIR/derivatives103-budapest/fmriprep \
    --work-dir $WORKDIR \
    --output-dir $OUTDIR \
    --njobs $NCORES"

if [ ! -z "$2" ]; then
    CMD="$CMD --space $2"
fi

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -e \
  "$IMG" \
  $PYTHONWRAP $CMD
