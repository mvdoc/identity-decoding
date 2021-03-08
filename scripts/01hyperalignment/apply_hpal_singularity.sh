#!/bin/bash -ex
IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/hpalfunc)
TMPDIR=$WORKDIR/tmp
FMRIPREPDIR=$BASEDIR/derivatives/fmriprep
FSDIR=$BASEDIR/derivatives103-budapest/freesurfer
MAPPERDIR=$BASEDIR/derivatives103-budapest/hpal
OUTDIR=$BASEDIR/derivatives103-budapest/hpal-func
PYTHONWRAP=$BASEDIR/scripts/python27

NCORES=16

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

SUBJ=sub-"$1"
HEMI="$2"

if [ $HEMI == L ]; then
    HEMIFS=lh
else
    HEMIFS=rh
fi

CMD="$BASEDIR/scripts/00preproc/apply_hpal.py \
    --inputs $FMRIPREPDIR/$SUBJ/ses-*first/func/*"$HEMI".func.gii \
    --mask $FSDIR/fsaverage6/SUMA/"$HEMIFS".maskmedial.niml.dset \
    --mapper $MAPPERDIR/$SUBJ/"$SUBJ"_task-movie_space-fsaverage6_hemi-"$HEMI"_target-hpal_mapper.h5 \
    --reverse $MAPPERDIR/sub-sid000005/sub-sid000005_task-movie_space-fsaverage6_hemi-"$HEMI"_target-hpal_mapper.h5 \
    --output-dir $OUTDIR \
    --nproc $NCORES"

singularity run  \
  -B /idata \
  -B /dartfs-hpc \
  -B $TMPDIR:/tmp \
  -e \
  "$IMG" \
  $PYTHONWRAP $CMD
