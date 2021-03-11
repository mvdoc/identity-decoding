# Shared neural codes for visual and semantic information about familiar others in a common representational space

This repository contains the code for the analyses reported in *Shared neural codes for visual and semantic information about familiar others in a common representational space* by Matteo Visconti di Oleggio Castello, James V. Haxby, & M. Ida Gobbini.

**DISCLAIMER** These scripts are shared in a format that is suitable for archival and review. All analyses were run inside a singularity container (shared in the current repository) on a local cluster and on [Discovery, Dartmouth's HPC cluster](https://rc.dartmouth.edu/index.php/discovery-overview/). The paths listed in these scripts need to be modified in order to run the scripts on a different system.

**If you have any questions related to the code, please open an issue in this repository.**

## Data

The raw data and its derivatives will be released on OpenNeuro after publication. We will update this section with more information on how to download the data.

## Repository structure

* [`singularity`](singularity) contains code to generate the singularity image that was used to run all analyses
* [`src`](src) contains a python package (`famfaceangles`) containing various general functions used in the analysis scripts
* [`scripts`](scripts)  contains the scripts used for the analyses reported in the manuscript

In the following sections we describe each file in detail.

### singularity

This folder contains the following files

* `Singularity-neurodocker`: a singularity definition file for the image used in all analyses
* `create-image.sh`: a bash script to generate the singularity image. Note that the syntax used in this script is for singularity versions 2.X. New versions of singularity will need a different syntax, and they have not been tested with this definition file.

### src

This folder contains the python package `famfaceangles` with helper functions used in the analysis scripts. It can be installed as any other python package (e.g., `pip install -e src`)

### scripts

This folder contains the following scripts

#### Preprocessing
* [`00preproc/run-fmriprep103-singularity.sh`](scripts/00preproc/run-fmriprep103-singularity.sh) calls fmriprep to preprocess the data.
* [`00preproc/prepare-fsaverage6-suma.sh`](scripts/00preproc/prepare-fsaverage6-suma.sh) prepares the *fsaverage6* surfaces to be used with SUMA.
* [`00preproc/make-maskmedial-fsaverage6.sh`](scripts/00preproc/make-maskmedial-fsaverage6.sh) creates a mask in NIML format to remove medial nodes in *fsaverage6*.

#### Hyperalignment

* [`01hyperalignment/run_preproc_hpal.py`](scripts/01hyperalignment/run_preproc_hpal.py) preprocesses the data from *The Grand Budapest Hotel* to be used for hyperalignment.
* [`01hyperalignment/run_preproc_hpal_singularity.sh`](scripts/01hyperalignment/run_preproc_hpal_singularity.sh) shows an example call with singularity.
* [`01hyperalignment/run_hpal.py`](scripts/01hyperalignment/run_hpal.py) runs the hyperalignment algorithm.
* [`01hyperalignment/run_hpal_singularity.sh`](scripts/01hyperalignment/run_hpal_singularity.sh) shows an example call with singularity.
* [`01hyperalignment/apply_hpal.py`](scripts/01hyperalignment/apply_hpal.py) applies the hyperalignment transformations to the input data.
* [`01hyperalignment/apply_hpal_singularity.sh`](scripts/01hyperalignment/apply_hpal_singularity.sh) shows an example call with singularity.

#### GLM

* [`02glm/run_glm_model.py`](scripts/02glm/run_glm_model.py) runs a GLM model for the face perception experiment using the specified model.
* [`02glm/run_glm_blockrun_singularity.sh`](scripts/02glm/run_glm_blockrun_singularity.sh) shows an example call with singularity on anatomically-aligned data and a BLOCK model estimated within each run.
* [`02glm/run_glm_blockrun_hpal_singularity.sh`](scripts/02glm/run_glm_blockrun_hpal_singularity.sh) shows an example call with singularity on hyperaligned data and a BLOCK model estimated within each run.
* [`02glm/run_glm_localizer_bwsj.py`](scripts/02glm/run_glm_localizer_bwsj.py) runs the GLM model for the hyperaligned localizer data.
* [`02glm/run_glm_localizer_bwsj_singularity.sh`](scripts/02glm/run_glm_localizer_bwsj_singularity.sh) shows an example call with singularity.
* [`02glm/workflows.py`](scripts/02glm/workflows.py) contains additional functions and Nipype workflows required to run the GLM models.

#### MVPA

Between-subject searchlight decoding

* [`03mvpa/run_sl_bwsbj.py`](scripts/03mvpa/run_sl_bwsbj.py) runs between-subject whole-brain searchlight decoding.
* [`03mvpa/run_sl_bwsbj_singularity.sh`](scripts/03mvpa/run_sl_bwsbj_singularity.sh) shows an example call with singularity on hyperaligned data.
* [`03mvpa/run_sl_bwsbj_fsaverage6_singularity.sh`](scripts/03mvpa/run_sl_bwsbj_fsaverage6_singularity.sh) shows an example call with singularity on anatomically-aligned data.

Between-subject ROI decoding

* [`03mvpa/run_bwsj_roi_v2.py`](scripts/03mvpa/run_bwsj_roi_v2.py) runs between-subject decoding analyses within manually defined ROIs.
* [`03mvpa/run_bwsj_roi_v2_singularity.sh`](scripts/03mvpa/run_bwsj_roi_v2_singularity.sh) shows an example call with singularity.
* [`03mvpa/run_sl_roi.py`](scripts/03mvpa/run_sl_roi.py) contains some additional functions needed for ROI decoding.

Within-subject searchlight decoding

* [`03mvpa/run_sl.py`](scripts/03mvpa/run_sl.py) runs within-subject whole-brain searchlight decoding.
* [`03mvpa/run_sl_blockrun_singularity.sh`](scripts/03mvpa/run_sl_blockrun_singularity.sh) shows an example call with singularity.
* [`03mvpa/run_sl_blockrun_permutation_singularity.sh`](scripts/03mvpa/run_sl_blockrun_permutation_singularity.sh) shows an example call with singularity to generate permuted maps.

Cross-validated RSA

* [`03mvpa/run_sl_cvrsa.py`](scripts/03mvpa/run_sl_cvrsa.py) runs within-subject searchlight cross-validated RSA.
* [`03mvpa/run_sl_cvrsa_blockrun_singularity.sh`](scripts/03mvpa/run_sl_cvrsa_blockrun_singularity.sh) shows an example call with singularity.
* [`03mvpa/run_rsa_target.py`](scripts/03mvpa/run_rsa_target.py) runs model-based RSA by comparing the cross-validated brain RDMs with model RDMs.
* [`03mvpa/run_rsa_target_singularity.sh`](scripts/03mvpa/run_rsa_target_singularity.sh) shows an example call with singularity.

#### Statistics

Permutation testing for between-subject MVPC

* [`04stat/run_permtest_bootstrap.py`](scripts/04stat/run_permtest_bootstrap.py) runs permutation testing with bootstrapping.
* [`04stat/run_permtest_bwsbj_bootstrap_singularity.sh`](scripts/04stat/run_permtest_bwsbj_bootstrap_singularity.sh) shows an example call with singularity.
* [`04stat/make_fam-diff_bwsj_identity.sh`](scripts/04stat/make_fam-diff_bwsj_identity.sh) creates difference maps (familiar - visual) from precomputed accuracy maps.
* [`04stat/run_permtest_famdiff_bwsbj_bootstrap_singularity.sh`](scripts/04stat/run_permtest_famdiff_bwsbj_bootstrap_singularity.sh) runs permutation testing on the familiar - visual difference maps.
* [`04stat/make-maskfdrval-diff-identity-bsmvpc.sh`](scripts/04stat/make-maskfdrval-diff-identity-bsmvpc.sh) makes a mask that highlights significant nodes for the familiar - visual difference map.

Threshold-Free Cluster Enhancement for within-subject MVPC and RSA

* [`04stat/run_tfce_fsaverage6_cosmo.m`](scripts/04stat/run_tfce_fsaverage6_cosmo.m) runs the CoSMoMVPA TFCE code for within-subject MVPC.
* [`04stat/run_tfce_fsaverage6_cosmo_blockrun_hpalsubjs_singularity.sh`](scripts/04stat/run_tfce_fsaverage6_cosmo_blockrun_hpalsubjs_singularity.sh) shows an example call with singularity.
* [`04stat/make_thresholded_avg_blockrun_fsaverage6-hpal.sh`](scripts/04stat/make_thresholded_avg_blockrun_fsaverage6-hpal.sh) creates thresholded maps based on the TFCE values.
* [`04stat/make_thresholded_avg_blockrun_fsaverage6-hpal_all.sh`](scripts/04stat/make_thresholded_avg_blockrun_fsaverage6-hpal_all.sh) calls the previous script for all comparisons of interest.
* [`04stat/run_tfce_cvrsa_fsaverage6_cosmo.m`](scripts/04stat/run_tfce_cvrsa_fsaverage6_cosmo.m) runs the CoSMoMVPA TFCE code for within-subject cross-validated RSA.
* [`04stat/run_tfce_cvrsa_fsaverage6_cosmo_singularity.sh`](scripts/04stat/run_tfce_cvrsa_fsaverage6_cosmo_singularity.sh) shows an example call with singularity.
* [`04stat/make_thresholded_avg_cvrsa_fsaverage6.sh`](scripts/04stat/make_thresholded_avg_cvrsa_fsaverage6.sh) creates thresholded maps based on the TFCE values.
* [`04stat/make_thresholded_avg_cvrsa_fsaverage6_singularity.sh`](scripts/04stat/make_thresholded_avg_cvrsa_fsaverage6_singularity.sh) shows an example with singularity.
* [`04stat/make_thresholded_avg_cvrsa_fsaverage6_all.sh`](scripts/04stat/make_thresholded_avg_cvrsa_fsaverage6_all.sh) calls the previous script for all comparisons of interest.

#### Visualization

* [`05viz/drive-suma-blockrun-fsaverage6-group-hpal.sh`](scripts/05viz/drive-suma-blockrun-fsaverage6-group-hpal.sh) shows an example call to `DriveSuma` to generate surface plots.
