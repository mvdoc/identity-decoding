#!/bin/bash -ex

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
SCRIPTDIR="$BASEDIR/scripts/03stat"

FAM=$1
TARG=$2
HEMI=$3

CMD="cd $SCRIPTDIR && octave --eval run_tfce_cvrsa_fsaverage6_cosmo\(\'$FAM\',\'$TARG\',\'$HEMI\',10000\)"
#CMD="cd $SCRIPTDIR && octave --eval run_tfce_cvrsa_fsaverage6_cosmo\(\'$FAM\',\'$TARG\',\'$HEMI\',10\)"

singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$CMD"
