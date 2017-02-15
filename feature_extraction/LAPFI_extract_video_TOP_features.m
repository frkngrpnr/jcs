function [fset] = LAPFI_extract_video_TOP_features(inpath, feature_name, feature_params)
% inpath folder will have .mat files with aligned videos in them
% for LGBP-TOP, you need to have the VLFeat library installed.
%% Input validation
extra_params={}; if nargin<3, feature_params = []; end
feature_name = upper(feature_name);
bss = '\';
if strcmp(feature_name, 'LGBPTOP'), extra_params = gaborFilterBank(3,6,9,9); end
options = struct('feature_name', feature_name, 'histeq', 0);

files = dir([inpath bss '*.mat']);
N = numel(files);
fset = struct;
fset.meta.feature_options = options;
fset.label = zeros(N,1);
fset.filter = ones(N,1);
fset.filename = cell(N, 1);
for i=1:N
    waitbar(i/N);
    %if mod(i,100)==0, fprintf('video %d/%d\n',i,N); end
    %% Load video
    load([inpath bss files(i).name]);
    tv = tic();
    filename_withoutmat = strrep(files(i).name, '.mat', '');
    fset.filename{i,:} = filename_withoutmat;
    %close all, plot_video(V); pause
    %% Construct 3D video:
    F = size(V.data, 1); % number of frames in this video
    if F<3 % hope that this is not the first video:
    desc = zeros(1, size(fset.data,2));
    fset.filter(i,:) = 0;
    else
    %vid3d = zeros([64,64,F]); %zeros([alignment.meta.image_size(1:2), F]);
    %if strcmp(feature_name, 'LBPTOP') || strcmp(feature_name, 'LGBPTOP') || strcmp(feature_name, 'LPQTOP'), vid3d=zeros(64,64,F); end
    vid3d=zeros(64,64,F);
    for f=1:F
        %img = vec2square(V.data(f,:));
        img = reshape(V.data(f,:),[64 64 3]);
        %% preprocessing
        if options.histeq
            img = histeq(img/255)*255;
        end
%         if strcmp(feature_name, 'LBPTOP') || strcmp(feature_name, 'LGBPTOP') || strcmp(feature_name, 'LPQTOP'), img=imresize(img,[64,64]); end
%         if strcmp(feature_name, 'LBPTOP') || strcmp(feature_name, 'LGBPTOP') || strcmp(feature_name, 'LPQTOP'), img=mean(img,3); end
        img=imresize(img,[64,64]);
        img=mean(img,3);
    %fprintf('%s vs %s\n',mat2str(size(img)), mat2str(size(vid3d)));
    
        vid3d(:,:,f) = img;
        %img = imresize(img, alignment.meta.image_size(1:2)); % will already be resized
        %desc = extract_TOP_features(vid3d, feature_name, feature_params, extraParameters);        
    end % for each frame
    desc = extract_TOP_features(vid3d, feature_name, feature_params, extra_params);
    end % if enough frames
    %% Extract and save features
    
    if i==1, fset.data = zeros(N, numel(desc)); end
    if i==1, fprintf('feature size = %d for %s[%s]\n',numel(desc),feature_name, mat2str(feature_params)); end
    if mod(i,100)==1, fprintf('video %d/%d processed in %f seconds\n',i,N,toc(tv)); end
    fset.data(i,:) = desc;
end

end % main function

%% Helpers
function desc = extract_TOP_features(vid, featureName, gridParameter, extraParameters)
    if strcmp(featureName,'LBPTOP')
        desc = video_to_lbp_top_2014(vid);
    elseif strcmp(featureName,'LGBPTOP')
        desc = video_to_lgbp_top_2015(vid, extraParameters);
    elseif strcmp(featureName,'LPQTOP')
        desc = video_to_lpq_top(vid,gridParameter);
    elseif strcmp(featureName,'HOGTOP')
        desc = hog_top(vid,gridParameter);
    else
        error(['Unknown feature: ' featureName]);
    end
    desc = reshape(desc,1,[]);
end

