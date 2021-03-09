#!/bin/bash -ex

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
SCRIPTDIR="$BASEDIR/scripts/03stat"

TARG=$1
HEMI=$2
NITER=10000

CMD="cd $SCRIPTDIR && octave --eval run_tfce_fsaverage6_cosmo\(\'$TARG\',\'$HEMI\',$NITER,1,\'fsaverage6\',\'blockrun-deconvolve\'\)"

singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$CMD"
