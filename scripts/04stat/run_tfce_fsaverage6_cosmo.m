function run_tfce_fsaverage6_cosmo(target, hemi, niter, hpalsubs, space, glmtype)
% Run TFCE using CoSMoMVPA for target classification, on hemisphere hemi, and niter iterations.
%
% Example:
% run_tfce_cosmo('fam-identity', 'L', 10000)
%
% To run it on the subset of subjects with hyperalignment data:
% run_tfce_cosmo('fam-identity', 'L', 10000, 1)
%
% To run it on the subset of subjects with hyperalignment on hyperaligned data:
% run_tfce_cosmo('fam-identity', 'L', 10000, 1, 'hpalfsaverage6')

if nargin == 3
    hpalsubs = 0;
    space = 'fsaverage6';
    glmtype = 'tent7';
elseif nargin == 4
    space = 'fsaverage6';
    glmtype = 'tent7';
elseif nargin == 5
    glmtype = 'tent7';
end

pkg load parallel

addpath('/idata/DBIC/castello/famface_angles/3rd/CoSMoMVPA/mvpa');
cosmo_set_path();
addpath('/idata/DBIC/castello/famface_angles/3rd/AFNI/src/matlab');
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/surfing'));
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/misc'));
addpath(genpath('/idata/DBIC/castello/famface_angles/3rd/surfing/toolbox_fast_marching'));
addpath('/idata/DBIC/castello/famface_angles/3rd/gifti-1.6');

%% Directories and params -- need to be hardcoded :(
% derivdir = '/idata/DBIC/castello/famface_angles/derivatives';
fsdir = '/idata/DBIC/castello/famface_angles/derivatives';
%% XXX: TEMPORARY FIX WHILE I CAN'T CHANGE PERMISSIONS ON DISCOVERY
derivdir = '/dartfs-hpc/rc/home/j/f0015tj/tmp-famface/derivatives103';
hemi2sfx = struct('L', 'lh', 'R', 'rh');
% mask = [derivdir '/' 'freesurfer/fsaverage6/SUMA/' hemi2sfx.(hemi) '.maskmedial.niml.dset']
mask = [fsdir '/' 'freesurfer/fsaverage6/SUMA/' hemi2sfx.(hemi) '.maskmedial.niml.dset']

if strcmp(space, 'hpalsid000005fsaverage6')
    derivdir = [derivdir '103-budapest'];
    fprintf("Setting derivatives directory to %s \n", derivdir);
end
% sldir = [derivdir '/' 'slclf-tent7-fsaverage6'];
sldir = [derivdir '/' 'slclf-' glmtype '-', space];
outdir = [sldir '/' 'group'];
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

% orig_tmpl = '%s_task-both_space-fsaverage6_hemi-%s_target-%s_sl.niml.dset';
% perm_tmpl = '%s_task-both_space-fsaverage6_hemi-%s_target-%s_sl_100perm.niml.dset';
if strcmp(glmtype, 'tent7') || strcmp(glmtype, 'block-deconvolve')
    task = 'both';
else
    task = [target(1:3) '1back'];
end
orig_tmpl = ['%s_task-' task '_space-' space '_hemi-%s_target-%s_sl.niml.dset'];
perm_tmpl = ['%s_task-' task '_space-' space '_hemi-%s_target-%s_sl_100perm.niml.dset'];


mask_ds = cosmo_surface_dataset(mask);
mask_idx = find(mask_ds.samples);
n_mask_idx = numel(mask_idx);

%% Step 1. Load surface
surf_fn = [fsdir '/' 'freesurfer/fsaverage6/SUMA/' hemi2sfx.(hemi) '.pial.gii'];
[vertices, faces] = surfing_read(surf_fn);
nvertices = size(vertices, 1);

%% Step 2. Load original and permuted data
switch target
    case {'fam-identity', 'str-identity'}
        h0_mean = 1/4;
    case {'fam-orientation', 'str-orientation', 'fam-orientation-male', 'fam-orientation-female', 'str-orientation-male', 'str-orientation-female'}
        h0_mean = 1/5;
    case {'fam-identity-male', 'fam-identity-female', 'str-identity-male', 'str-identity-female', 'fam-gender', 'str-gender'}
        h0_mean = 1/2;
    otherwise
        error('target %s invalid', target)
end

% store masks to reuse later
masks = cell(1, nsubjects);
orig_data = cell(1, nsubjects);
for isub = 1:nsubjects
    this_sub = subjects{isub};
    fprintf('Loading original data for %s\n', this_sub);
    orig_fn = [sldir '/' this_sub '/' sprintf(orig_tmpl, this_sub, hemi, target)];
    ds = cosmo_surface_dataset(orig_fn);
    nsamples = size(ds.samples, 1);
    ds.sa.chunks = ones([nsamples, 1]) * isub;
    ds.sa.targets = ones([nsamples, 1]);
    ds = cosmo_average_samples(ds);
    % We need to extend each dataset in case we don't have all nodes
    % ds = extend_dataset_features(ds, nvertices, h0_mean);
    % We'll take the intersection to be more conservative
    % masks{isub} = ismember(ds.a.fdim.values{1}, mask_idx);
    % ds = cosmo_slice(ds, masks{isub}, 2);

    % XXX: apply common mask on fsaverage6 to remove medial wall
    % only if it's not already removed
    % need to change fdim values as well and reset fa.node_indices
    if n_mask_idx < numel(ds.fa.node_indices)
        ds = cosmo_slice(ds, mask_idx, 2);
        ds.fa.node_indices = 1:n_mask_idx;
        ds.a.fdim.values{1} = mask_idx;
    end
    orig_data{isub} = ds;
end
% store latest number of samples to store number of folds
nfolds = nsamples;
orig_data = cosmo_stack(orig_data);

perm_data = cell(1, nsubjects);
for isub = 1:nsubjects
    this_sub = subjects{isub};
    fprintf('Loading permuted data for %s\n', this_sub);
    perm_fn = [sldir '/' this_sub '/' sprintf(perm_tmpl, this_sub, hemi, target)];
    ds = cosmo_surface_dataset(perm_fn);
    nsamples = size(ds.samples, 1);
    % average across folds. input has nfolds * 100, with folds changing fastest
    ds.sa.chunks = sort(repmat(1:100, [1, nfolds]))';
    ds.sa.targets = ones([nsamples, 1]);
    ds = cosmo_average_samples(ds);
    nsamples = size(ds.samples, 1);
    ds.sa.chunks = ones([nsamples, 1]) * isub;
    ds.sa.perm = (1:100)';
    % We need to extend each dataset in case we don't have all nodes
    % ds = extend_dataset_features(ds, nvertices, h0_mean);
    % We'll take the intersection to be more conservative
    % ds = cosmo_slice(ds, masks{isub}, 2);
    if n_mask_idx < numel(ds.fa.node_indices)
        ds = cosmo_slice(ds, mask_idx, 2);
        % need to change fdim values as well and reset fa.node_indices
        ds.fa.node_indices = 1:n_mask_idx;
        ds.a.fdim.values{1} = mask_idx;
    end
    perm_data{isub} = ds;
end

% now we need to store it as a cell of cosmo datasets for null
null_data = cell(1, 100);
for ip = 1:100
    ds = cellfun(@(x) cosmo_slice(x, ip), perm_data, 'UniformOutput', false);
    ds = cosmo_stack(ds);
    null_data{ip} = ds;
end


%% Step 4. Put everything together and run TFCE
nbr = cosmo_cluster_neighborhood(orig_data, 'vertices', vertices, 'faces', faces);

opt = struct();
opt.cluster_stat = 'tfce';
opt.niter = niter;
opt.h0_mean = h0_mean;
opt.seed = 42;
opt.null = null_data;
%opt.nproc = 40;
opt.nproc = 24;

fprintf('Using the following options\n');
cosmo_disp(opt);

z_ds = cosmo_montecarlo_cluster_stat(orig_data, nbr, opt);

%% Step 5. Store results
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

fnout = ['group_task-' task '_space-' space '_hemi-%s_target-%s_tfce%dp'];
fnout = sprintf(fnout, hemi, target, niter);
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

%% XXX: DO NOT USE THIS -- BUG
function ds = extend_dataset_features(ds, nvertices, h0_mean)
    % impute h0_mean so it will be 0 when it gets subtracted
    nsamples = size(ds.samples, 1);
    tmp_data = ones([nsamples, nvertices]) * h0_mean;
    % THIS IS A BUG! IN COSMOMVPA THE ACTUAL INDICES ARE STORED IN ds.a.fdim.values
    % AND ds.fa.node_indices is simply 1 : n_node_indices
    tmp_data(:, ds.fa.node_indices) = ds.samples;
    ds.samples = tmp_data;
    ds.fa.node_indices = 1:nvertices;
    ds.a.fdim.values = {ds.fa.node_indices};
    cosmo_check_dataset(ds)
end
