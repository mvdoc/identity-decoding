function run_tfce_cvrsa_fsaverage6_cosmo(familiarity, target, hemi, niter)
% Run TFCE using CoSMoMVPA for target RDM and niter iterations.
%
% familiarity: 'fam' | 'str'
% if familiarity is 'diff', then the paired constrast fam - str will be computed
% target: 'identity' | 'orientation' | 'mirror' | 'gender' | 'famscore'
% Example:
% run_tfce_cosmo('fam', 'identity', 'L', 10000)

pkg load parallel

addpath('/idata/DBIC/castello/famface_angles/3rd/CoSMoMVPA/mvpa');
cosmo_set_path();
addpath('/idata/DBIC/castello/famface_angles/3rd/AFNI/src/matlab');
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/surfing'));
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/misc'));
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/toolbox_fast_marching'));
addpath('/idata/DBIC/castello/famface_angles/3rd/gifti-1.6');

%% Directories and params -- need to be hardcoded :(
derivdir = '/idata/DBIC/castello/famface_angles/derivatives';
sldir = [derivdir '/' 'slcvrsa-target-blockrun-deconvolve-fsaverage6'];
outdir = [sldir '/' 'group'];

hpalsubs = 1;

if hpalsubs
    fprintf("Running on only subjects with hyperalignment data!\n")
    outdir = [outdir '-hpal'];
    subjects = {'sub-sid000005', 'sub-sid000007', 'sub-sid000009', 'sub-sid000010', 'sub-sid000013', 'sub-sid000024', 'sub-sid000025', 'sub-sid000030', 'sub-sid000034', 'sub-sid000050', 'sub-sid000055', 'sub-sid000114', 'sub-sid000134', 'sub-sid000278'};
else
    fprintf("Running on all subjects\n")
    subjects = {'sub-sid000005', 'sub-sid000007', 'sub-sid000009', 'sub-sid000010', 'sub-sid000013', 'sub-sid000014', 'sub-sid000024', 'sub-sid000025', 'sub-sid000028', 'sub-sid000030', 'sub-sid000034', 'sub-sid000035', 'sub-sid000037', 'sub-sid000050', 'sub-sid000055', 'sub-sid000114', 'sub-sid000134', 'sub-sid000278'};
end
outdir
nsubjects = length(subjects);

switch familiarity
    case 'fam'
        task = 'fam1back';
    case 'str'
        task = 'str1back';
end

if strcmp(target, 'famscore')
    if strcmp(task, 'fam1back')
        % fulltarget = 'id+or+mi+ge+fam';
        fulltarget = 'fam';
    else
        error('famscore is valid only for familiar faces')
    end
else
    if strcmp(target, 'vgg16-maxpool-layer-1')
        fulltarget = target;
    else
        fulltarget = 'id+or+mi+ge';
    end
end

orig_tmpl = '%s_task-%s_space-fsaverage6_hemi-%s_cvrsa_metric-zcorrelation_sl_target-%s.niml.dset';
perm_tmpl = '%s_task-%s_space-fsaverage6_hemi-%s_cvrsa_metric-zcorrelation_sl_target-%s_%03dperm.niml.dset';

hemi2sfx = struct('L', 'lh', 'R', 'rh');

% XXX: RSA results are already masked, no need to apply mask
mask = [derivdir '/' 'freesurfer/fsaverage6/SUMA/' hemi2sfx.(hemi) '.maskmedial.niml.dset'];
mask_ds = cosmo_surface_dataset(mask);
mask_idx = find(mask_ds.samples);
n_mask_idx = numel(mask_idx);

%% Step 1. Load surface
surf_fn = [derivdir '/' 'freesurfer/fsaverage6/SUMA/' hemi2sfx.(hemi) '.pial.gii'];
[vertices, faces] = surfing_read(surf_fn);
nvertices = size(vertices, 1);

%% Step 2. Load original and permuted data
h0_mean = 0;

if strcmp(familiarity, 'diff')
    error('Difference not implemented yet')
    famsfx = 'diff';
    familiarity = {'fam1back', 'str1back'};
else
    famsfx = familiarity;
    familiarity = {task};
end
nfam = numel(familiarity);
% store masks to reuse later
masks = cell(1, nsubjects);
orig_data = cell(1, nsubjects * nfam);
idx = 1;
for isub = 1:nsubjects
    this_sub = subjects{isub};
    fprintf('Loading original data for %s\n', this_sub);
    for ifam = 1:nfam
        orig_fn = [sldir '/' this_sub '/' sprintf(orig_tmpl, this_sub, familiarity{ifam}, hemi, fulltarget)];
        fprintf('  %s\n', orig_fn);
        ds = cosmo_surface_dataset(orig_fn);
        % extract samples with the predictors we want
        if length(ds.sa.labels) > 1
            mask_predictors = strcmp(target, ds.sa.labels);
            ds = cosmo_slice(ds, mask_predictors, 1);
        end
        nsamples = size(ds.samples, 1);
        ds.sa.chunks = ones([nsamples, 1]) * isub;
        ds.sa.targets = ones([nsamples, 1]) * ifam;
        % NO NEED TO APPLY MASK WITH RSA -- ALREADY MASKED
        %% XXX: apply common mask on fsaverage6 to remove medial wall
        %% need to change fdim values as well and reset fa.node_indices
        %ds = cosmo_slice(ds, mask_idx, 2);
        %% need to change fdim values as well and reset fa.node_indices
        %ds.fa.node_indices = 1:n_mask_idx;
        %ds.a.fdim.values{1} = mask_idx;
        % Fisher-transform correlations
        ds.samples = atanh(ds.samples);
        orig_data{idx} = ds;
        idx = idx + 1;
    end
end

% store latest number of samples to store number of folds
nfolds = nsamples;
orig_data = cosmo_stack(orig_data);

MAX_PERM = 50;
null_data = cell(1, MAX_PERM);

for iperm = 1:MAX_PERM
    perm_data = cell(1, nsubjects * nfam);
    idx = 1;
    for isub = 1:nsubjects
        this_sub = subjects{isub};
        fprintf('Loading permuted data for %s, perm %03d\n', this_sub, iperm);
        for ifam = 1:nfam
            perm_fn = [sldir '/' this_sub '/' sprintf(perm_tmpl, this_sub, familiarity{ifam}, hemi, fulltarget, iperm)];
            fprintf('  %s\n', perm_fn);
            ds = cosmo_surface_dataset(perm_fn);
            % extract samples with the predictors we want
            if length(ds.sa.labels) > 1
                mask_predictors = strcmp(target, ds.sa.labels);
                ds = cosmo_slice(ds, mask_predictors, 1);
            end
            % add chunk information
            nsamples = size(ds.samples, 1);
            ds.sa.targets = ones([nsamples, 1]) * ifam;
            ds.sa.chunks = ones([nsamples, 1]) * isub;
            % Fisher-transform correlations
            ds.samples = atanh(ds.samples);
            perm_data{idx} = ds;
            idx = idx + 1;
        end
    end
    null_data{iperm} = cosmo_stack(perm_data);
end

%% Step 4. Put everything together and run TFCE
nbr = cosmo_cluster_neighborhood(orig_data, 'vertices', vertices, 'faces', faces);

opt = struct();
opt.cluster_stat = 'tfce';
opt.niter = niter;
if nfam == 1
    opt.h0_mean = h0_mean;
end
opt.seed = 42;
opt.null = null_data;
%opt.nproc = 48;
opt.nproc = 16;

fprintf('Using the following options\n');
cosmo_disp(opt);

z_ds = cosmo_montecarlo_cluster_stat(orig_data, nbr, opt);

%% Step 5. Store results
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fnout_tmpl = 'group_task-%s_space-fsaverage6_hemi-%s_cvrsa_metric-zcorrelation_sl_target-%s_pred-%s_tfce%dp';
fnout = sprintf(fnout_tmpl, task, hemi, fulltarget, target, niter);
fnout_ = [outdir '/' fnout];

try
    fprintf('Saving in %s\n', fnout_)
    cosmo_map2surface(z_ds, [fnout_ '.niml.dset']);
    save([fnout_ '.mat'], '-struct', 'z_ds');
catch
    fprintf('Couldnt save in %s, saving locally\n', fnout_)
    cosmo_map2surface(z_ds, [fnout '.niml.dset']);
    save([fnout '.mat'], '-struct', 'z_ds');
end
end
