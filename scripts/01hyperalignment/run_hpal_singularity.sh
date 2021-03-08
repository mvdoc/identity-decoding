#!/bin/bash -ex
set -eu

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/hpalpreproc)
TMPDIR=$WORKDIR/tmp
DATADIR=$BASEDIR/derivatives103-budapest/hpal-preproc
OUTDIR=$BASEDIR/derivatives103-budapest/hpal
FS_LICENSE=$BASEDIR/scripts/00preproc/license.txt
PYTHONWRAP=$BASEDIR/scripts/python27

NCORES=24

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

CMD="$BASEDIR/scripts/00preproc/run_hpal.py \
    --input-dir $DATADIR \
    --hemi $1 \
    --fsdir $BASEDIR/derivatives/freesurfer \
    -n $NCORES \
    -o $OUTDIR"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -B "$TMPDIR":/tmp \
  -e \
  "$IMG" \
  $PYTHONWRAP $CMD
