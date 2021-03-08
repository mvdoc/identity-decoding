#!/bin/bash -ex

IMGNAME="poldracklab_fmriprep_1.0.3-2018-01-04-8c1973f461fd.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/nobackup_fmriprep103)
TMPDIR=$WORKDIR/tmp
DATADIR=$BASEDIR/data
OUTDIR=$BASEDIR/derivatives103
FS_LICENSE=$BASEDIR/scripts/00preproc/license.txt

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

singularity run  \
  -B "$WORKDIR":/work \
  -B "$DATADIR":/data:ro \
  -B "$OUTDIR":/out \
  -B /idata:/idata \
  -e \
  "$IMG" \
  /data /out participant \
  --bold2t1w-dof 9 \
  --output-space fsaverage6 T1w template \
  --nthreads "$NCORES" \
  --omp-nthreads 8 \
  --fs-license-file "$FS_LICENSE" \
  --participant-label "$1" \
  -w /work
