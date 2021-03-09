#!/bin/bash
set -ex

# BASEDIR="/idata/DBIC/castello/famface_angles"
BASEDIR="/backup/users/contematto/famface_angles"
DERIVDIR="$BASEDIR/derivatives"
SLTYPE="slclf-blockrun-deconvolve"
TARG="$1"
SUBSET="$2"
SPACE="$3"

if [ -z "$SPACE" ]; then
    SPACE="fsaverage6"
fi
SLTYPE="$SLTYPE"-"$SPACE"
SLDIR="$DERIVDIR"/"$SLTYPE"
SFX=group
if [ ! -z "$SUBSET" ]; then
    SFX="$SFX"-"$SUBSET"
    # HACK to use another directory for input files
    SLDIR="$SLDIR"h
fi
OUTDIR="$SLDIR"/"$SFX"

HERE="$PWD"

AVERAGER="$BASEDIR"/scripts/utils/mean_niml.py
SPLITTER="$BASEDIR"/scripts/utils/split_hemi.py

if [[ $TARG == *"fam"* ]]; then
    TASK=fam1back
else
    TASK=str1back
fi

for HEMI in L R; do
    echo "01. Averaging input nimls"
    echo 
    OUTFN="$OUTDIR"/group_task-"$TASK"_space-"$SPACE"_hemi-"$HEMI"_target-"$TARG"_acc.niml.dset
    python $AVERAGER -i "$SLDIR"/*/*hemi-"$HEMI"*target-"$TARG"_sl.niml.dset --how intersect -o "$OUTFN"

    echo "02. Averaging group average across folds"
    echo
    OUTFN2="${OUTFN/acc/acc_avg}"
    cd "$OUTDIR"
    3dTstat -mean -prefix "$(basename $OUTFN2)" "$OUTFN"
    # mv "$(basename $OUTFN2)" "$OUTFN2"
    
    # we need to pad only if we don't have applied hyperalignment
    # NO now we're using the mask, so no need to pad
    #if [ $SPACE != hpalfsaverage6 ]; then
    #    echo "03. Padding TFCE dataset to max number of nodes"
    #    TFCE="$OUTDIR"/group_task-"$TASK"_space-"$SPACE"_hemi-"$HEMI"_target-"$TARG"_tfce10000p.niml.dset 
    #    TFCEPAD="${TFCE/0p/0p_pad}"
    #    ConvertDset -input "$TFCE" -o_niml -o $TFCEPAD -pad_to_node d:"$OUTFN2"
    #else
        TFCEPAD="$OUTDIR"/group_task-"$TASK"_space-"$SPACE"_hemi-"$HEMI"_target-"$TARG"_tfce10000p.niml.dset 
    #fi

    echo "04. Thresholding group avg with TFCE values"
    echo
    OUTFN3="${OUTFN2/avg/avg_thr}"
    3dcalc -a "$OUTFN2" -b "$TFCEPAD" -expr 'a*step(b-1.96)' -prefix "$OUTFN3"
done
