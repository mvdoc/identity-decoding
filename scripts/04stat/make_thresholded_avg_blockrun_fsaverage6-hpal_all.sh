#!/bin/bash
TARGETS=(fam-identity str-identity)
for T in ${TARGETS[*]}; do
    # bash make_thresholded_avg_blockrun_fsaverage6_singularity.sh "$T" hpal fs6hp
    bash make_thresholded_avg_blockrun_fsaverage6-hpal.sh "$T" hpal fsaverage6
done
