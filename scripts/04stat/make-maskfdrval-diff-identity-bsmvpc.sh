#!/bin/bash

# Make a mask of nodes that surpass the threshold. Actually they are -1 or 1 depending on the sign
DATADIR=/Users/mvdoc/projects/famface-angles/derivatives/slclfbwsbj-blockrun-deconvolve-hpalsid000005fsaverage6


for HEMI in L R; do
  3dcalc \
    -a "$DATADIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_fdrpval.niml.dset \
    -b "$DATADIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_avgfdrthr.niml.dset \
    -prefix "$DATADIR"/bwsbj_space-hpalsid000005fsaverage6_hemi-"$HEMI"_target-diff-identity_sl_maskfdrthr.niml.dset \
    -expr 'ispositive(a-cdf2stat(0.05,24,0,0,0))*(step(b)-step(-b))'
done

# values in a are already -log10
# cdf2stat(0.05,24,0,0,0) equals to -log10(0.05), that is alpha of 0.05
# step(b)-step(-b) should correspond to the sign function (+1 if positive, -1 if negative)
