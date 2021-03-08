#!/bin/bash -ex

#IMGNAME="neurodocker.img"
IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/glm)
DATADIR=$BASEDIR/data
FMRIPREPDIR=$BASEDIR/derivatives103-budapest/fmriprep
HPALDIR=$BASEDIR/derivatives103-budapest/hpal-func
PYTHON_WRAP=$BASEDIR/scripts/python36
EVENTDIR=$BASEDIR/derivatives103-budapest/localizer_events
NCORES=16


HEMI="$1"
MODELSFX='localizer-bwsj5'
OUTDIR=$BASEDIR/derivatives103-budapest/glm-"$MODELSFX"
WORKDIR="$WORKDIR"-locbw"$HEMI"ffa
OUTDIR="$OUTDIR"
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

CMD="$BASEDIR/scripts/01glm/run_glm_localizer_bwsj.py \
       --run-files $HPALDIR/sub-*005/*/*/*localizer*$HEMI*.niml.dset \
       --confound-files $HPALDIR/sub-*005/*/*/*localizer*.tsv \
       --event-files $EVENTDIR/sub-*005_*localizer*_events.tsv \
       -w $WORKDIR \
       -o $OUTDIR \
       --output-template bwsj_task-localizer_space-hpalsid000005fsaverage_hemi-"$HEMI"_ \
       -n $NCORES \
       $CMD"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -e \
  "$IMG" \
  $PYTHON_WRAP $CMD
