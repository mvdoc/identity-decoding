#!/bin/bash
set -e

IMGNAME="neurodocker.img"
BASEDIR=/idata/DBIC/castello/famface_angles
IMG=$BASEDIR/singularity/$IMGNAME
WORKDIR=$(readlink -f $BASEDIR/_workdirs/glm)
PYTHONWRAP="$BASEDIR"/scripts/python27

programname=$0
function usage {
    echo "usage: $programname TYPE (roi | sl) subjectid task [id+or+mi+ge | id+or+mi+ge+fam | fam | vgg-maxpool-layer-{1,..,5}  ] [0 .. 99] "
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TYPE="$1"
SID=sub-"$2"
TASK=$3
PREDTYPE=$4
PERM=$5

if [ -z "$PREDTYPE" ]; then
    PREDTYPE="id+or+mi+ge"
fi
if [ -z "$PERM" ]; then
    PERM=0
fi

printf -v P "%03d" $PERM

RSADIR=$BASEDIR/derivatives/"$TYPE"cvrsa-blockrun-deconvolve-fsaverage6
OUTDIR=$BASEDIR/derivatives/"$TYPE"cvrsa-target-blockrun-deconvolve-fsaverage6
PREDDIR=$BASEDIR/derivatives/target-rdms

COMMON_PRED="$PREDDIR"/rdm_id+or+mi+ge.tsv

neurodocker() {
  singularity exec -B /idata -c $IMG "$@"
}

for HEMI in L R; do
    INPUT="$RSADIR"/"$SID"/"$SID"_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_cvrsa_metric-zcorrelation_"$TYPE"
    OUTPUT="$OUTDIR"/"$SID"/"$SID"_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_cvrsa_metric-zcorrelation_"$TYPE"_target-"$PREDTYPE"
    if [ $PERM != 0 ]; then
        INPUT="$INPUT"_"$P"perm
        OUTPUT="$OUTPUT"_"$P"perm
    fi
    INPUT="$INPUT".niml.dset
    OUTPUT="$OUTPUT".niml.dset

    CMD="$PYTHONWRAP $BASEDIR/scripts/02mvpa/run_rsa_target.py \
            --ds-rsa $INPUT \
            -o $OUTPUT \
            --predictors "

    if [ $PREDTYPE != "fam" ] && [[ $PREDTYPE != *"vgg"* ]]; then
            CMD="$CMD $COMMON_PRED"
    fi

    if [ "$TASK" == "fam1back" ] && [ "$PREDTYPE" == "id+or+mi+ge+fam" ] || [ "$PREDTYPE" == "fam" ]; then
        CMD="$CMD $PREDDIR/$SID/${SID}_rdm_famscore.tsv"
    fi

    if [[ "$PREDTYPE" == *"vgg"* ]]; then
        PRED="$PREDTYPE".tsv
        if [ "$TASK" == "fam1back" ]; then
            PRED=rdm_fam_"$PRED"
        else
            PRED=rdm_str_"$PRED"
        fi
        CMD="$CMD $PREDDIR/$PRED"
    fi

    neurodocker $CMD
done
  
