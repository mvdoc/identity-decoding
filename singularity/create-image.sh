#!/bin/bash
IMG="neurodocker.img"
TMPDIR="./tmpdir"

if [ ! -e $IMG ]; then
  singularity create --size 30000 $IMG
fi

if [ ! -d $TMPDIR ]; then
  mkdir -p $TMPDIR
fi

export SINGULARITY_TMPDIR=$TMPDIR
export SINGULARITY_CACHEDIR=$TMPDIR
singularity bootstrap $IMG Singularity-neurodocker
