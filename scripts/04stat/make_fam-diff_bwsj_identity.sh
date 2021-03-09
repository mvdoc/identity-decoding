#!/bin/bash
# setup singularity wrapper
IMGNAME="neurodocker.img"
#BASEDIR="/idata/DBIC/castello/famface_angles"
BASEDIR="/backup/users/contematto/famface_angles"
IMG=$BASEDIR/singularity/$IMGNAME

sing () {
    singularity run  \
    -B /backup \
    -e \
    "$IMG" \
    "$@"
}

DERIVDIR="$BASEDIR/derivatives103"
SLTYPE="slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6"

DERIVDIR="$DERIVDIR"/"$SLTYPE"

cd $DERIVDIR

sing \
'
BASEDIR="/backup/users/contematto/famface_angles"
DERIVDIR="$BASEDIR/derivatives103"
SLTYPE="slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6"
DERIVDIR="$DERIVDIR"/"$SLTYPE"

cd $DERIVDIR

for HEMI in L R; do
    for PERM in `seq -w 0 50`; do
        3dcalc \
            -a "$DERIVDIR"/bwsbj_task-fam1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-fam-identity_sl_0"$PERM"ip.niml.dset \
            -b "$DERIVDIR"/bwsbj_task-str1back_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-str-identity_sl_0"$PERM"ip.niml.dset \
            -expr 'a-b' \
            -prefix "$DERIVDIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_0"$PERM"ip.niml.dset
    done
done
'

