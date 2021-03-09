#!/bin/bash -ex

IMGNAME="neurodocker.img"
# BASEDIR=/idata/DBIC/castello/famface_angles
BASEDIR=/backup/users/contematto/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
OUTDIR=$BASEDIR/derivatives103/slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6
PYTHONWRAP="$BASEDIR"/scripts/python27
NCORES=4

HEMI=$1

INPUT="$OUTDIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_000ip.niml.dset
PERMUTED="$OUTDIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_*ip.niml.dset
OUTPUT="$OUTDIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl


CMD="$BASEDIR/scripts/03stat/run_permtest_bootstrap.py \
     -i $INPUT \
     -p $PERMUTED \
     --prefix $OUTPUT \
     -n $NCORES \
     --tail 1 \
     --nbootstraps 10000 \
     $CMD"


singularity run  \
-e \
"$IMG" \
"$PYTHONWRAP" "$CMD"
