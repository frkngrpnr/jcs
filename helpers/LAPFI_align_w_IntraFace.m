function [alignment] = LAPFI_align_w_IntraFace(inpath, outpath)
%% This code uses the IntraFace library in order to align faces.
% inpath should contain videos, outpath will have .mat files that contain
% aligned videos.

% alignment will be a struct with fields:
% data : Nx1 cell
% label : Nx1

alignment.meta = struct;
alignment.meta.dataset = 'LAP-FI';
alignment.meta.alignment = 'IF';
alignment.meta.image_size = [64,64,3];
alignment.meta.class_names = cell(1,1);

%vi = 1; returnAt = -1;
bss='\';
videos = dir([inpath bss '*.mp4']);
for vi=1:numel(videos)
    
    tv=tic();
    
    %% first, check if mat file already exists in outpath
    %namewoext = videos(vi).name; namewoext(numel(namewext)-4:end)=[];
    %namewext = videos(vi).name; %namewext(numel(namewext)-4:end)=[];
    %fprintf('checking %s.mp4.. comment this out later.\n', namewoext);
    if exist([outpath bss videos(vi).name '.mat'], 'file')
        %fprintf('%s already aligned.\n', videos(vi).name);
        continue;
    end
    %% mat file does not exist. continue:
    videopath = [inpath bss videos(vi).name];
    %if rand>0.9, disp(['Processing ' num2str(vi) 'th video']); end % in emotion #' num2str(e-2) ':' videopath]); end
    fprintf('Processing %s (video %d/%d) .. ',videos(vi).name, vi, numel(videos));
    %pause
    %[V, opts] = process_video_intraface(videopath);
    [V, opts] = process_video_intraface_loose(videopath, struct('alignment','intraface', 'histeq', 0, 'grayscale', 0, 'face_size', alignment.meta.image_size(1), 'face_depth', alignment.meta.image_size(3)));
    %close all, plot_video(V); pause
%    V.label = e-2;
 %   V.label_string = x(e).name;
    %alignment.data{vi,1} = V;
    %alignment.label(vi,1) = e-2;
    alignment.filename{vi,1} = V.filename;
    if vi==1,            alignment.meta.alignment_options = opts; end
    %if vi == returnAt, return; end
    if numel(V.data)==0, disp('WARNING: Something might be wrong..'); end
    save([outpath bss videos(vi).name '.mat'], 'V');
    waitbar(vi/numel(videos));
    
    fprintf(' Took %f seconds.\n', toc(tv));
    
end

end % main function

%% Helpers

%% IntraFace
function [V, opts] = process_video_intraface_loose(vid, opts)
% vid should be read by read_video. or should have the field:
%   % data: N x [H * W * D]
% options.alignment : 
%   % 1: intraface alignment
%   % 2: DPM alignment
%% 1. input validation
if isa(vid, 'char')
    vid = read_video(vid);
end
if nargin<2
    opts = struct('alignment','intraface', 'histeq', 0, 'face_size', 64, 'face_depth', 1);
end
Models = evalin('base', 'Models'); option = evalin('base', 'option');
%% 2. process video
N = size(vid.data,1);
%N = 3
V = vid;
rids = []; % samples to remove at the end
for i=1:N
    waitbar(i/N);
    %% 2.1. preprocess frame
    img = vid.data(i,:);
    img = reshape(img, vid.image_size);
    if opts.face_depth == 1 && size(img,3)>1
        img=mean(img,3); 
    elseif opts.face_depth == 3 && size(img,3) == 1
        img = repmat(img,1,1,3);
    end
    %% 2.2. detect face and landmarks
    [lms, bbox, pose] = furkan_img_to_landmarks(img, Models, option, 0);
    if numel(lms)==98 % detected
        % v1
        % [bbox] = bbox_from_landmarks(lms);
        % nimg = imcrop(img, bbox);
        % v2
        % [nimg, fitted_lms] = cropFaceAndLandmarks(img, lms, opts.face_size);
        % v3: finds the landmarks and rotates according to them.
        [rimg,rlms] = rotate_image_and_landmarks(img, lms, pose.angle(3));
        [nimg, fitted_lms, cropRect] = cropFaceAndLandmarks(rimg, rlms, opts.face_size);
        % loosen the cropRect and re-crop:
        wf = 0.15; % widening factor
        cropRect(1) = max(1, cropRect(1)-cropRect(3)*wf);
        cropRect(2) = max(1, cropRect(2)-cropRect(3)*wf);
        cropRect(3) = max(1, cropRect(3)+cropRect(4)*2*wf);
        cropRect(4) = max(1, cropRect(4)+cropRect(4)*2*wf);
        
        nimg = imcrop(rimg, cropRect);
        nimg = imresize(nimg, repmat(opts.face_size,1,2));
        if opts.histeq
            nimg = histeq(nimg/255)*255;
        end
        
        
        poseangle = pose.angle;
    else % not detected
        rids = [rids; i];
        nimg = imresize(img, repmat(opts.face_size,1,2));        lms = zeros(49,2); bbox = zeros(1,4); poseangle = zeros(1,3);
    end
    
%     %% 2.2.1 visualize:
%     close all; figure;
%     subplot(1,2,1); imshow(img, []); title('Original Image');
%     subplot(1,2,2); imshow(nimg, []); title('Aligned Image');
%     pause
    
    %% 2.3. Save face and other data
    if i==1 % pre-allocate
        %V.data = single(zeros(N,numel(nimg))); 
        V.data = single(zeros(N, opts.face_depth*(opts.face_size)^2));
        %V.data = single(zeros(N, opts.face_size, opts.face_size));
        V.landmarks = zeros(N, numel(lms));
        %V.fitted_landmarks = zeros(N, numel(lms));
        V.bbox = zeros(N, 4);
        V.pose = zeros(N, 3);
    end
    V.data(i,:) = reshape(nimg,1,[]);
    %V.data(i,:,:,:) = img;
    V.landmarks(i,:) = reshape(lms,1,[]);
    V.bbox(i,:) = reshape(bbox,1,[]);
    V.pose(i,:) = reshape(poseangle,1,[]);
    
end

V = clean_dataset(V, rids);

end 

%% mmread
function [vid] = read_video(basepath, verbose)
% basepath can be the path to a video file, or a folder with images inside
% output vid will be a struct with fields:
%  data : N x [H * W * D], where [H,W,D]=size(frame) and N = num. of frames
%  readby (if basepath is a video file) : 1 for MATLAB's VideoReader, 2 for MMread
%  filename: string
bss = '\';
if nargin<2, verbose = 0; end
vid = struct;
vid.data=[];
%% 1. determine the name of file (or folder):
    if strcmp(basepath(end), '\') || strcmp(basepath(end), '/')
        basepath(end)=[];
    end
    [pathstr,name,ext] = fileparts(basepath);
    vid.filename = [name ext];

%% 2. read
if exist(basepath) == 2 % this is a file, hopefully a video
    if verbose, fprintf('reading frames from video file'); end
    %% 2.1. try with VideoReader
    % MATLAB's built-in video reader :
    try
        %disp('reading w VideoReader');
        vidObj = VideoReader(filePath);
        lastFrame = read(vidObj, inf);
        N = vidObj.NumberOfFrames;
        vid.input_numframes = N;
        if verbose, fprintf('VideoReader found %d frames\n',N); end
        for i=1:N
            try
                img = read(vidObj,i);
                %if size(img,3)==3, img=rgb2gray(img); end
                if i==1
                    [H,W,D] = size(img);
                    vid.data = single(zeros(N,(H*W*D)));
                    vid.image_size = [H,W,D];
                end
                %vid.data(i,:,:,:) = img;
                vid.data(i,:) = reshape(img,1,[]);
            catch error_videocannotread
                continue;
            end
        end
    
    catch
        %disp(['will try mmread because VideoReader failed in ' basepath]);    
        %% 2.2. if VideoReader fails, try with mmread
        [video, audio] = mmread(basepath);
        N = numel(video.frames);
        vid.input_numframes = N;
        if verbose, fprintf('VideoReader failed. mmread found %d frames\n',N); end
        for i=1:N
            img = video.frames(i).cdata;
            if i==1
                [H,W,D] = size(img);
                vid.data = single(zeros(N,(H*W*D)));
                vid.image_size = [H,W,D];
            end
            %vid.data(i,:,:,:) = img;
            vid.data(i,:) = reshape(img,1,[]);
        end
    end % try if VideoReader works
    
elseif exist(basepath) == 7 % this is a folder. read the images inside
    imgnames = [dir([basepath bss '*.jpg']); dir([basepath bss '*.png']); dir([basepath bss '*.bmp'])];
    N = numel(imgnames);
    vid.input_numframes = N;
    if verbose, fprintf('This is a folder with %d images\n',N); end
    for i=1:N
        img = imread([basepath bss imgnames(i).name]);
        %img = mean(img,3); if i==1 && rand>0.95, warning('converting to grayscale'); end
        if i==1
            [H,W,D] = size(img);
            vid.data = single(zeros(N,(H*W*D)));
            vid.image_size = [H,W,D];
        end
        %vid.data(i,:,:,:) = img;
        vid.data(i,:) = reshape(img,1,[]);
    end
else
    fprintf('%s does not exist.\n', basepath);
end

end 

%% landmarking
function [face, fitted_lms, cropRect] = cropFaceAndLandmarks(img, lms, FaceSize)
% Assumes rotation is already handled.
%FaceSize = 120;
lms = reshape(lms,[],2);
L = size(lms,1); % number of landmarks

face_width = norm(lms(20,:)-lms(29,:));
horizontal_padding_faceWidth_ratio = 0.2;
cropRect = zeros(1,4);
cropRect(1) = lms(20,1) - face_width*horizontal_padding_faceWidth_ratio; % X-left
cropRight = lms(29,1) + face_width*horizontal_padding_faceWidth_ratio; % X-right
cropRect(3) = cropRight - cropRect(1);
% y
face_height = max(lms(:,2))-min(lms(:,2));
%verticalPadding_faceHeight_ratio = 0.15;
verticalPadding_faceHeight_ratio_UP = 0.20;
verticalPadding_faceHeight_ratio_DOWN = 0.20;
cropRect(2) = min(lms(:,2)) - face_height*verticalPadding_faceHeight_ratio_UP;
cropRect(4) = max(lms(:,2)) + face_height*verticalPadding_faceHeight_ratio_DOWN - cropRect(2);
%aimg=rimg;%NOT!

face = imcrop(img, cropRect);
fitted_lms = reshape(lms,[],2);
fitted_lms(:,1) = fitted_lms(:,1) - repmat(cropRect(1)-1,L,1);
fitted_lms(:,2) = fitted_lms(:,2) - repmat(cropRect(2)-1,L,1);



%[originalSize, ~] = size(face)
%fitted_lms = fitted_lms * FaceSize/originalSize;
originalFaceSize=size(face); % to fit the landmarks after imresize
face = imresize(face, [FaceSize FaceSize]);
fitted_lms(:,1) = fitted_lms(:,1) * FaceSize/originalFaceSize(2);
fitted_lms(:,2) = fitted_lms(:,2) * FaceSize/originalFaceSize(1);




end

%% 
function [rimg,rlms] = rotate_image_and_landmarks(img, lms,angle, rotCenter)
% angle is in degrees
% rotCenter default = centroid of lms. It SHOULD be default.
lms = reshape(lms,[],2); 
lms=double(lms); % required for rotateAround
angle=-angle;
lmCentroid = [mean(lms(:,1)),mean(lms(:,2))];
if nargin<4, rotCenter = lmCentroid; end
%% 1st, center the landmarks
%imgCenter = [imgSize(2)/2+0.5,imgSize(1)/2+0.5];
clms = zeros(size(lms));
for i=1:size(lms,1)
    clms(i,:) = lms(i,:) - lmCentroid;
end
rotMatrix=[cosd(angle), sind(angle); -sind(angle) ,cosd(angle)];
rlms = clms*rotMatrix;
%% 2, add center back to the rotated landmarks
for i=1:size(lms,1)
    rlms(i,:) = rlms(i,:) + lmCentroid;
end

rotCenter = double(rotCenter);
rimg = rotateAround(img,rotCenter(2),rotCenter(1),-angle);

end % function rotate_image_and_landmarks

function output=rotateAround(image, pointY, pointX, angle, varargin)
% ROTATEAROUND rotates an image.
%   ROTATED=ROTATEAROUND(IMAGE, POINTY, POINTX, ANGLE) rotates IMAGE around
%   the point [POINTY, POINTX] by ANGLE degrees. To rotate the image
%   clockwise, specify a negative value for ANGLE.
%
%   ROTATED=ROTATEAROUND(IMAGE, POINTY, POINTX, ANGLE, METHOD) rotates the
%   image with specified method:
%       'nearest'       Nearest-neighbor interpolation
%       'bilinear'      Bilinear interpolation
%       'bicubic'       Bicubic interpolation
%    The default is fast 'nearest'. Switch to 'bicubic' for nicer results.
%
%   Example
%   -------
%       imshow(rotateAround(imread('eight.tif'), 1, 1, 10));
%
%   See also IMROTATE, PADARRAY.

%   Contributed by Jan Motl (jan@motl.us)
%   $Revision: 1.2 $  $Date: 2014/05/01 12:08:01 $

% FG: image = double(image);
% pointY=double(pointY);
% pointX=double(pointX);

% Parameter checking.
numvarargs = length(varargin);
if numvarargs > 1
    error('myfuns:somefun2Alt:TooManyInputs', ...
        'requires at most 1 optional input');
end
optargs = {'nearest'};    % Set defaults for optional inputs
optargs(1:numvarargs) = varargin;
[method] = optargs{:};    % Place optional args in memorable variable names

% Initialization.
[imageHeight imageWidth ~] = size(image);
centerX = floor(imageWidth/2+1);
centerY = floor(imageHeight/2+1);

dy = centerY-pointY;
dx = centerX-pointX;

% How much would the "rotate around" point shift if the 
% image was rotated about the image center. 
[theta, rho] = cart2pol(-dx,dy);
[newX, newY] = pol2cart(theta+angle*(pi/180), rho);
shiftX = round(pointX-(centerX+newX));
shiftY = round(pointY-(centerY-newY));

% Pad the image to preserve the whole image during the rotation.
padX = abs(shiftX);
padY = abs(shiftY);

padded = padarray(image, double([padY padX]));

% Rotate the image around the center.
rot = imrotate(padded, angle, method, 'crop');

% Crop the image.
output = rot(padY+1-shiftY:end-padY-shiftY, padX+1-shiftX:end-padX-shiftX, :);

end
