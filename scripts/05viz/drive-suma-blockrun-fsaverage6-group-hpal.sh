#!/bin/bash -x
TARG="$1"
SPACE=fsaverage6

BASEDIR="$HOME/Documents/projects/famface-angles"
DERIVDIR="$BASEDIR"/derivatives
SLDIR=slclf-blockrun-deconvolve-
GROUP=group-hpal


if [ -z "$SPACE" ]; then
  SPACE=fsaverage6
fi

if [[ $TARG == *"fam"* ]]; then
    TASK=fam1back
else
    TASK=str1back
fi

if [[ $TARG == *"identity"* ]]; then
  RANGE="0.25 0.35"
else  # orientation
  RANGE="0.20 0.65"
fi

export DYLD_LIBRARY_PATH=/opt/X11/lib/flat_namespace
#OUT="img/"$SLDIR"img_$TARG"_"$SPACE"_"$GROUP"
OUT="img_new/"$SLDIR"img_$TARG"_"$SPACE"_"$GROUP"
SURFDIR="$DERIVDIR"/freesurfer/fsaverage6/SUMA
DATADIR="$DERIVDIR"/"$SLDIR""$SPACE"/"$GROUP"
VIEWDIR="views"
CMAPDIR="cmaps"

mkdir "$OUT"

for HEMI in rh lh; do
  if [ $HEMI == rh ]; then
    HEMILBL=R
  else
    HEMILBL=L
  fi
  SPEC="$SURFDIR"/fsaverage6_"$HEMI".spec
  suma -niml -spec "$SPEC" &
  DriveSuma -com viewer_cont -load_view "$VIEWDIR"/"$HEMI"_lateral_white.niml.vvs
  sleep 3
  DriveSuma -com surf_cont \
    -switch_surf inf_200 \
    -load_dset "$DATADIR"/group_task-"$TASK"_space-"$SPACE"_hemi-"$HEMILBL"_target-"$TARG"_acc_avg_thr.niml.dset \
    -load_cmap "$CMAPDIR"/plasma_r.1D.cmap \
    -I_range "$RANGE"
  DriveSuma -com viewer_cont -key a
  sleep 3
  # DriveSuma -com viewer_cont -viewer_size 1280 1024
  sleep 1
  DriveSuma -com viewer_cont -key 'r'
  DriveSuma -com viewer_cont -load_view "$VIEWDIR"/"$HEMI"_medial_white.niml.vvs
  DriveSuma -com viewer_cont -key 'down'
  sleep 1
  DriveSuma -com viewer_cont -key 'r'
  DriveSuma -com viewer_cont -load_view "$VIEWDIR"/"$HEMI"_ventral_white.niml.vvs
  DriveSuma -com viewer_cont -key 'down'
  sleep 1
  DriveSuma -com viewer_cont -key 'r'
  sleep 1
  DriveSuma -com recorder_cont -save_as "$OUT"/"$HEMI".png -save_all
  # kill suma
  DriveSuma -com kill_suma

  # convert images 
  for f in "$OUT"/"$HEMI"*; do
    convert -trim $f ${f%.*}_trim.png
  done
  # append top and bottom
  convert "$OUT"/"$HEMI".0000{0,1}_trim.png -background white -gravity center -append "$OUT"/"$HEMI".top.png
  # append to ventral
  if [ $HEMI == lh ]; then
    FIRST="$OUT"/"$HEMI".00002_trim.png
    SECOND="$OUT"/"$HEMI".top.png
  else
    FIRST="$OUT"/"$HEMI".top.png
    SECOND="$OUT"/"$HEMI".00002_trim.png
  fi
  convert $FIRST $SECOND -background white -gravity center +append "$OUT"/"$HEMI".final.png
done

# make final image
convert "$OUT"/lh.final.png "$OUT"/rh.final.png -background white -gravity center +append "$OUT"/mh.final.png
