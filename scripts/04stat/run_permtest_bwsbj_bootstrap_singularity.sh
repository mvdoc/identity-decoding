#!/bin/bash -ex

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
FMRIPREPDIR=$BASEDIR/derivatives103-budapest/fmriprep
FSDIR=$BASEDIR/derivatives103-budapest/freesurfer
OUTDIR=$BASEDIR/derivatives103/slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=16

TARG=$1
HEMI=$2

if [[ $TARG == *fam* ]]; then
    TASK="fam1back"
else
    TASK="str1back"
fi

INPUT="$OUTDIR"/bwsbj_task-"$TASK"_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-"$TARG"_sl_000ip.niml.dset
PERMUTED="$OUTDIR"/bwsbj_task-"$TASK"_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-"$TARG"_sl_*ip.niml.dset
OUTPUT="$OUTDIR"/bwsbj_task-"$TASK"_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-"$TARG"_sl


CMD="$BASEDIR/scripts/03stat/run_permtest_bootstrap.py \
     -i $INPUT \
     -p $PERMUTED \
     --prefix $OUTPUT \
     -n $NCORES \
     --nbootstraps 10000 \
     $CMD"


singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$PYTHONWRAP" "$CMD"
