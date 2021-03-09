#!/bin/bash -e

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
SCRIPTDIR="$BASEDIR/scripts/03stat"

TARG="$1"
TASK="$2"

CMD="$SCRIPTDIR/make_thresholded_avg_cvrsa_fsaverage6.sh $TARG $TASK"

singularity run  \
-B /idata \
-B /dartfs-hpc \
-e \
"$IMG" \
"$CMD"
