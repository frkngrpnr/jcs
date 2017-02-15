function [lgbp_top] = video_to_lgbp_top_2015(vid,gfb)
%% Inputs :
% vid is a matrix [I x F]
% I : image size (=w*h)
% F : number of frames
%% Outputs :
% lgbptop is a 174 (3*58) x 18 matrix, where each column is the LBP-TOP of
% a gabor video, and we have 18 different gabor videos per video.
lgbp_top=[];

S = size(gfb,1);
O = size(gfb,2);
gs = cell(1,S*O);

counter = 1;
for s=1:S
    for o=1:O
        gaborVid = get_gabor_video(vid,gfb{s,o});        
        lgbp_top = [lgbp_top;video_to_lbp_top_2014(gaborVid)];            
    end
    %waitbar(s/S);
end


% we will also reshape the element to fit in a single column vector : 

%lgbp_top = reshape(lgbp_top ,prod(size(lgbp_top)) ,1 );
%lgbp_top = reshape(lgbp_top ,[] ,1 );

end

function gv = get_gabor_video(vid,gaborFilter)
gv=[];
[FrameHeight,FrameWidth,NumberOfFrames] = size(vid);
for f=1:NumberOfFrames
    thisFrame = vid(:,:,f);
    g = conv2(thisFrame, real(gaborFilter));
    %% Now, crop the image back to the original size :
    [Height,Width] = size(g);    
    g = imcrop(g,[round((Width-FrameWidth)/2) round((Height-FrameHeight)/2), FrameWidth-1, FrameHeight-1]);    
    gv(:,:,f)=g;
end

end