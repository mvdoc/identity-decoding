#!/bin/bash -ex

IMGNAME="neurodocker.img"
# BASEDIR=/idata/DBIC/castello/famface_angles
BASEDIR=/backup/users/contematto/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
OUTDIR=$BASEDIR/derivatives103/slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=16

HEMI=$1

INPUT1="$OUTDIR"/bwsbj_task-fam1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-fam-identity_sl_000ip.niml.dset
INPUT2="$OUTDIR"/bwsbj_task-str1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-str-identity_sl_000ip.niml.dset
PERMUTED1="$OUTDIR"/bwsbj_task-fam1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-fam-identity_sl_*ip.niml.dset
PERMUTED2="$OUTDIR"/bwsbj_task-str1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-str-identity_sl_*ip.niml.dset
OUTPUT="$OUTDIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity-clipped_sl


CMD="$BASEDIR/scripts/03stat/run_permtest_bootstrap.py \
     --input $INPUT1 \
     --input2 $INPUT2 \
     --permuted $PERMUTED1 \
     --permuted2 $PERMUTED2 \
     --prefix $OUTPUT \
     -n $NCORES \
     --tail 0 \
     --nbootstraps 10000 \
     --clip-min-value 0.25 \
     $CMD"


singularity run  \
-e \
-B /backup \
"$IMG" \
"$PYTHONWRAP" "$CMD"