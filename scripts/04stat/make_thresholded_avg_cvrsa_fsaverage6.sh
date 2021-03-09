#!/bin/bash -e
set -e
set -x

# BASEDIR="/idata/DBIC/castello/famface_angles"
# DERIVDIR="$BASEDIR/derivatives"
# XXX: temporarily changed this because of permission issues
BASEDIR="/dartfs-hpc/rc/home/j/f0015tj/tmp-famface"
DERIVDIR="$BASEDIR/derivatives103"
# SLTYPE="slcvrsa-target-blockrun-deconvolve-fsaverage6"
# XXX: changed to the following directory to have only subjects with hpal
SLTYPE="slcvrsa-target-blockrun-deconvolve-fsaverage6-hpalsubs"
SLDIR="$DERIVDIR"/"$SLTYPE"
# OUTDIR="$SLDIR"/group
OUTDIR="$SLDIR"/grouph
TARG="$1"
TASK="$2"
HERE="$PWD"

BASEDIR="/idata/DBIC/castello/famface_angles"
AVERAGER="$BASEDIR"/scripts/utils/mean_niml.py
SPLITTER="$BASEDIR"/scripts/utils/split_hemi.py

if [ "$TARG" == "id+or+mi+ge" ]; then
    PREDS=(identity orientation mirror gender) 
elif [ "$TARG" == "id+or+mi+ge+fam" ]; then
    PREDS=(famscore) 
elif [ "$TARG" == "vgg16-maxpool-layer-1" ]; then
    PREDS=("$TARG")
else
    echo "Invalid target: $TARG"
    exit 1
fi

echo "01. Extract individual samples"
# extract bucket for every subject
for fn in "$SLDIR"/sub-*/*task-"$TASK"_*target-"$TARG".niml.dset; do
    cd "$(dirname $fn)"
    if [ "$TARG" == "vgg16-maxpool-layer-1" ]; then
        OUT="${fn/$TARG/${TARG}_pred-${TARG}}"
	if [ ! -f "$OUT" ]; then
           ln -s "$fn" "$OUT"
	fi
    else
        for PRED in "${PREDS[@]}"; do
            OUT="${fn/$TARG/${TARG}_pred-${PRED}}"
            if [ ! -f "$OUT" ]; then
                3dbucket -prefix "$OUT" "$fn"["$PRED"]
            fi
        done 
    fi
done

cd $OUTDIR
for HEMI in L R; do
    for PRED in ${PREDS[@]}; do
        echo "01. Averaging input nimls ($PRED)"
        # OUTFN="$OUTDIR"/group_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_cvrsa_metric-zcorrelation_sl_target-"$TARG"_pred-"$PRED"_avg.niml.dset
        OUTFN="$OUTDIR"/group_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_cvrsa_metric-zcorrelation_sl_target-"$TARG"_avg.niml.dset
        python $AVERAGER -i "$SLDIR"/sub-*/*task-"$TASK"_*hemi-"$HEMI"_*pred-"$PRED".niml.dset --how intersect -o "$OUTFN"
	# 3dMean -prefix "$OUTFN" -i "$SLDIR"/sub-*/*task-"$TASK"_*hemi-"$HEMI"_*pred-"$PRED".niml.dset
        INFN="$OUTFN"

        echo "02. Thresholding group avg with TFCE values ($PRED)"
        echo
        OUTFN="${INFN/avg/avg_thr}"
        TFCE="$OUTDIR"/group_task-"$TASK"_space-fsaverage6_hemi-"$HEMI"_cvrsa_metric-zcorrelation_sl_target-"$TARG"_pred-"$PRED"_tfce10000p.niml.dset
        3dcalc -a "$INFN" -b "$TFCE" -expr 'a*step(b-1.96)' -prefix "$OUTFN"
        INFN="$OUTFN"
    done
done
cd "$HERE"
echo
echo "Done."
