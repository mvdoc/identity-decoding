#!/bin/bash -ex

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
FMRIPREPDIR=$BASEDIR/derivatives103-budapest/fmriprep
FSDIR=$BASEDIR/derivatives103-budapest/freesurfer
GLMDIR=$BASEDIR/derivatives103-budapest/glm-blockrun-hpal-nosmooth
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=16

programname=$0
function usage {
    echo "usage: $programname target hemi [roi] [n_permute]"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TARG=$1
HEMI=$2
ROI=$3
PERMUTE=$4

if [ $HEMI == L ]; then
  H=lh
else
  H=rh
fi

if [[ "$TARG" == "fam-"* ]]; then
  TASK=fam1back
else
  TASK=str1back
fi

if [ -z "$ROI" ]; then
    ROI=all
fi

DECON=deconvolve
OUTDIR=$BASEDIR/derivatives103-budapest/roi-bwsj-v2-blockrun-"$DECON"-hpalsid000005fsaverage6-nofeatsel

if [ ! -d "$OUTDIR" ]; then
    mkdir $OUTDIR
fi

INPUT="$GLMDIR"/sub-'*'/sub-'*'_task-"$TASK"_space-hpalsid000005fsaverage6_hemi-"$HEMI"_"$DECON"-block.niml.dset
ROIMASK="$BASEDIR"/derivatives103-budapest/rois/"$H".core-and-extended-rois-bws.niml.dset
OUTPUT="$OUTDIR"/bwsbj_roi-"$ROI"_task-"$TASK"_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-"$TARG"

if [ ! -z $PERMUTE ]; then
    printf -v P "%03d" $PERMUTE
    OUTPUT="$OUTPUT"_npermute-"$P"
    CMD="--n_permute $PERMUTE"
fi
OUTPUT="$OUTPUT".joblib

CMD="$BASEDIR/scripts/02mvpa/run_bwsj_roi_v2.py \
     --input $INPUT \
     --target $TARG \
     --ds_roi_file $ROIMASK \
     --roi $ROI \
     --nproc $NCORES \
     --output $OUTPUT \
     --no-feature-selection \
     $CMD"

echo "$CMD"

singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$PYTHONWRAP" "$CMD"
