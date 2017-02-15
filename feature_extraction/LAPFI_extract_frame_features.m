function [no_output] = LAPFI_extract_frame_features(inpath, outpath, feature_name, feature_params, extra_params)
% This code extracts frame-level features from aligned videos.
%% Parameters:
% inpath is a folder with .mat files that contain the aligned videos.
% outpath will have .mat files with the same names, containing frame-level descriptors
%% Input validation
if nargin<5, extra_params={}; end
if nargin<3, feature_params = []; end
bss = '\';
feature_name = upper(feature_name);

if strcmp(feature_name(1:3), 'CNN')
    extra_params.layers{end}.type = 'softmax';
    extra_params = vl_simplenn_tidy(extra_params);
    use_gpu = 1;
    if use_gpu
        extra_params = vl_simplenn_move(extra_params, 'gpu');
        extra_params.use_gpu = 1;
    end
end

%% Initilization
no_output=42;
options = struct('feature_name', feature_name, 'histeq', 0, 'encoding_name', 'FUN');
    options.feature_params = feature_params;
sure_outpath_exists = 0;
files = dir([inpath bss '*.mat']);
N = numel(files);
fset = struct;
fset.meta.feature_options = options;
fset.label = zeros(N,1);
fset.filter = ones(N,1);
fset.filename = cell(N, 1);
for i=1:N
    waitbar(i/N);
    %% check if video already processed, i.e. outpath already has filename:
    if exist([outpath bss files(i).name], 'file')
        if mod(i,50)==0, fprintf('(%d. skipped video) %s already processed.\n',i,files(i).name); end
        continue;
    end
    tv=tic();
    %% Load video
    load([inpath bss files(i).name]);
    filename_withoutmat = strrep(files(i).name, '.mat', '');
    fset.filename{i,:} = filename_withoutmat;
    %% Construct 3D video:
    F = size(V.data, 1); % number of frames in this video
    PV = V; % PV = processed video. the variable that will be saved into the mat file
    PV.data = [];
    PV.meta.feature_options = options;
    for f=1:F
        %fprintf('frame %d/%d of video %d\n',f,F,i);
        img = vec2square(V.data(f,:));
        %% preprocessing
        if options.histeq
            img = histeq(img/255)*255;
        end
        desc = extract_image_features(img, feature_name, feature_params, extra_params);
        if f==1, PV.data = zeros(F, numel(desc)); end % preallocate
        PV.data(f,:) = desc;
    end % for each frame
    % save frame level features, i.e. the struct PV:
    if sure_outpath_exists == 0, mkdir(outpath); sure_outpath_exists=1; end
    save([outpath bss files(i).name], 'PV');
    %% Extract and save features    
    if i==1, fset.data = zeros(N, numel(desc)); end
    fset.data(i,:) = desc;    
    if i==1 || mod(i,25)==0
        fprintf('processed %s in %f seconds (file %d/%d)\n',files(i).name,toc(tv),i,N); 
    end    
end

end % main function

%% Helpers
function desc = extract_image_features(img, featureName, gridParameter, extraParameters)
featureName = upper(featureName); % careful. this makes all the lettes uppercase
    if strcmp(featureName(1:3),'CNN') == 1 % gridParameter is the output layer ID(s), 
        %% Preprocess image
        %img2net = img2vggface(img);
        if size(img,3)==3
            img2net = single(img);
            for i=1:3
                img2net(:,:,i) = img2net(:,:,i) - extraParameters.meta.normalization.averageImage(i); 
            end
            if size(img2net,1)~=224 || size(img2net,2)~=224
                img2net = imresize(img2net, [224 224]);
            end
        elseif size(img,3)==1
            img2net = single(zeros(size(img)));
            for i=1:3
                img2net(:,:,i) = img - extraParameters.meta.normalization.averageImage(i); 
            end
            if size(img2net,1)~=224 || size(img2net,2)~=224
                img2net = imresize(img2net, [224 224]);
            end
        end
        %% Extract feature vector        
        if isfield(extraParameters, 'use_gpu') && extraParameters.use_gpu, img2net = gpuArray(img2net); end
        resp = vl_simplenn(extraParameters, img2net);
        desc = resp(gridParameter).x;
        if isfield(extraParameters, 'use_gpu') && extraParameters.use_gpu, desc = gather(desc); end
    else
        error(['Unknown feature: ' featureName]);
    end
    desc = reshape(desc,1,[]);
end
