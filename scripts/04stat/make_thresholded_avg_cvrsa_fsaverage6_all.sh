#!/bin/bash

# bash make_thresholded_avg_cvrsa_fsaverage6_singularity.sh id+or+mi+ge fam1back
# bash make_thresholded_avg_cvrsa_fsaverage6_singularity.sh id+or+mi+ge str1back
# bash make_thresholded_avg_cvrsa_fsaverage6_singularity.sh id+or+mi+ge+fam fam1back
bash make_thresholded_avg_cvrsa_fsaverage6_singularity.sh vgg16-maxpool-layer-1 fam1back
bash make_thresholded_avg_cvrsa_fsaverage6_singularity.sh vgg16-maxpool-layer-1 str1back
