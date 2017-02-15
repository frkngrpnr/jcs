function [lbptop, block_lbp_tops] = video_to_lbp_top_2014(vid, verbose)
if nargin<2, verbose = 0; end
%% Inputs :
% vid is a 3d matrix [I x I x F]
% I : image size (=w*h)
% F : number of frames
%% Outputs :
% lbptop is a 1 x 174 (=3*58) vector, which is obtained by concatenating 3
% orthogonal lbp histograms.


%W = 64; H = 80;

%[I,~,F] = size(vid);
%W = I; H = I;

[H,W,F] = size(vid);
% convert video to 3 dimensional :
%vid3d = zeros(H,W,F);

% obtain all spatial lbp histograms :

% xy:
all_lbps_xy = [];
for i=1:F
    %current_frame = reshape(vid(:,i) , H, W);
    current_frame = vid(:,:,i);        
    all_lbps_xy = [all_lbps_xy img_to_lbp_histogram(single(current_frame))];
end
%lbp_xy = mean(all_lbps_xy , 2);
lbp_xy_1 = mean(all_lbps_xy(:,1:ceil(F/2)),2);
if F == 1
lbp_xy_2 = lbp_xy_1; % there is only one image to extract LBP
else
lbp_xy_2 = mean(all_lbps_xy(:,(ceil(F/2)+1):F),2);
end
lbp_xy = [lbp_xy_1; lbp_xy_2]; % temporally enhanced LBP

% having constructed the 3d video, obtain 2 temporal lbp histograms xt and yt :
bs = 4; % number of frames in blocks
segments = 1:bs:F;
%if F==3, segments = [1,4]; end % a little trick for 3-frame videos.
%if F==4, segments = [1,5]; end % a little trick for 4-frame videos.
if F<bs, lbptop = [lbp_xy; zeros(H/4 * 58 * 2 , 1); zeros(W/4 * 58 * 2 ,1);]; return; end
if F==bs, segments = [1,F+1]; end

% xt, with constant-sized temporal segments
block_lbps_xt = [];
sc = numel(segments)-1; % segment count
for i = 1:sc
    %if t>=F, break; end
    %t:t+bs-1    
    current_segment = vid(:,:, segments(i):segments(i+1)-1); % now it has a consistent size : H x W x bs    
    cs_lbps = []; % lbps of current segment.
    for j = 1:H
        segment_xt_frame = reshape(current_segment(j,:,:) , W , bs);
        %size(segment_xt_frame)
        cs_lbps = [cs_lbps img_to_lbp_histogram(segment_xt_frame)];
    end
    block_lbps_xt = [block_lbps_xt mean(cs_lbps,2)];
end
%lbp_xt = mean(block_lbps_xt,2);

lbp_xt_1 = mean(  block_lbps_xt(:,1:ceil(sc/2))   ,2);
if sc==1
    lbp_xt = [lbp_xt_1; lbp_xt_1];
else
lbp_xt_2 = mean(block_lbps_xt(:,ceil(sc/2)+1:sc),2);
lbp_xt = [lbp_xt_1; lbp_xt_2];
end
% TODO : segment this to keep more temporal information !!!



% yt, with constant-sized temporal segments
block_lbps_yt = [];
for i = 1:numel(segments)-1
    %if t>=F, break; end
    %t:t+bs-1    
    current_segment = vid(:,:, segments(i):segments(i+1)-1); % now it has a consistent size : H x W x bs    
    cs_lbps = []; % lbps of current segment.
    for j = 1:W        
        segment_yt_frame = reshape(current_segment(:,j,:) , H , bs);
        cs_lbps = [cs_lbps img_to_lbp_histogram(segment_yt_frame)];
    end
    block_lbps_yt = [block_lbps_yt mean(cs_lbps,2)];
end
%lbp_yt = mean(block_lbps_yt,2);

lbp_yt_1 = mean(block_lbps_yt(:,1:ceil(sc/2)),2);
if sc==1
lbp_yt = [lbp_yt_1; lbp_yt_1];    
else
lbp_yt_2 = mean(block_lbps_yt(:,ceil(sc/2)+1:sc),2);
lbp_yt = [lbp_yt_1; lbp_yt_2];
end

% OLD, without temporal segmentation : 
%
% all_lbps_xt = [];
% all_lbps_yt = [];
% for i=1:H
%     img = vid3d(i,:,:);
%     %size(single(img)), W, F
%     img = reshape(single(img) , W,F);    
%     all_lbps_xt = [all_lbps_xt img_to_lbp_histogram(img)];
% end 
% lbp_xt = mean(all_lbps_xt,2);
% % yt : 
% for i=1:W
%     img = vid3d(:,i,:);
%     img = reshape(single(img) , H,F);    
%     all_lbps_yt = [all_lbps_yt img_to_lbp_histogram(img)];
% end
% lbp_yt = mean(all_lbps_yt,2);

lbptop = [lbp_xy; lbp_xt; lbp_yt];
if verbose
fprintf('xy : %d , xt : %d , yt : %d , Total = %d \n',numel(lbp_xy),numel(lbp_xt),numel(lbp_yt),(numel(lbptop)));
end


end

