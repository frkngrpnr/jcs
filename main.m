% This file reads the extracted features, optimizes the model and produces
% the test set predictions for ChaLearn LAP First Impressions and Job Candidate Screening coopetition.  
% Based on the options set in section 1, section 2 carries out feature
% extraction, which can take up to 10 days depending on your computer's configuration. 
% See the readme.txt for links to extracted features (to be placed under
% ./data/features) and the vggfer.mat (our FER fine-tuned VGG-Face DCNN model)

%% 1. Initialization
opts = struct;
opts.base_path = pwd; % determine full path of this directory
addpath(genpath(opts.base_path)); % add all subdirectories to MATLAB path
bss = '/'; % backslash symbol. '\' for Windows, '/' for Linux.
opts.bss = bss;
opts.align_faces = 0; % note that you need the IntraFace library to use the face alignment code.
opts.extract_features = 0; % if 1, features will be extracted (for this, you need VLFeat>=0.9.20 and MatConvNet>=1.0.beta18 installed). if 0, saved features will be loaded.
opts.video_path = [opts.base_path bss 'data' bss 'videos' bss 'raw'];
opts.alignment_path = [opts.base_path bss 'data' bss 'videos' bss 'aligned'];
opts.frame_features_path = [opts.base_path bss 'data' bss 'features' bss 'frame'];
opts.output_path= [opts.base_path opts.bss 'data' opts.bss 'output'] ;
opts.write_preds=1; % writes the predictions to predictions.pkl over a csv file
% setting this to 1 requires that python command is working !!!
%load('gt_train_struct')
%load('gt_trval')
load('anotations_train')
load('anotations_val')
load('test_filenames')
%load('val_filenames')
dimensions=cell(1,6);
for i=1:6
    dimensions{1,i}=anotations_train{1, i}.dimension;
end

use_val = 0;
Nfold=8; % number of fold used to optimize parameters within labeled training set
%% 2. Data preparation
%% 2.1. Face alignment

if opts.align_faces
    fprintf('%s.m: aligning faces..\n',mfilename);
    [alignment] = LAPFI_align_w_IntraFace(opts.video_path, opts.alignment_path);
end

if opts.extract_features
    fprintf('%s.m: extracting features..\n',mfilename);
    % Extract LGBP-TOP
    lgbptop = LAPFI_extract_video_TOP_features(opts.alignment_path, 'LGBPTOP');
    % Extract deep face features from VGG-Face fine tuned on FER-2013:
    load('vggfer.mat')
    LAPFI_extract_frame_features(opts.alignment_path, [opts.frame_features_path bss 'vggfer33'], 'CNN_VGGFER', 33, net);
    vggfer33fun = LAPFI_encode_frame_features([opts.frame_features_path bss 'vggfer33'], 'FUN');
    vggfer33fun = LAPFI_attach_labels(vggfer33fun);
    % Extract deep scene features from VGG-VeryDeep-19 network (available at http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat):
    net = load('imagenet-vgg-verydeep-19.mat');
    vd19 = LAPFI_extract_scene_features(opts.video_path, [opts.frame_features_path bss 'vd19'], 'CNN_VGGVD19', 39, net);
    vd19 = LAPFI_attach_labels(vd19);
else % load features
    % to reduce memory complexity we will load feature structs one by one
end
%% 3. Optimize the models and estimate val labels

% Step 3.1- FF(OS_IS13,vd19)-> ELM
ff_audio_scene

% Step 3.2 FF(lgbptop,vggfer33fun)->ELM
ff_face_feats

% Step 3.3: Stack predictions of 3.1 and 3.2. subsystem to RF
rf_score_fusion

% We are done! Time to write predictions in submission format
if (opts.write_preds)
    prepare_submission(scores_elm,dimensions,test_filenames);
end