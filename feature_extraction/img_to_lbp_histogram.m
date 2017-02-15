function [lbp] = img_to_lbp_histogram(img, lbp_parameter)
% returns a 58-dimensional LBP histogram from the image.
% img should be single(rgb2gray(imread))

% Fixed Parameter : assuming all images are 80 x 64.

% Nope : inputs are assumed either 80 x 64, 64 x 5 or 80 x 5
% 5 is the temporal block size and it can change.

img = single(img);

if nargin<2

mindim = min(size(img)); % we get min(width,height), in order to obtain a single patch.
% lbp = vl_lbp(img, min(size(img)) * 0.99); 
% lbp = reshape(lbp,prod(size(lbp)),1); 
% return;
if mindim >= 64
    lbp = vl_lbp(img, mindim/4);
elseif mindim > 10
    lbp = vl_lbp(img, mindim);
elseif mindim >= 3
    lbp = vl_lbp(img, mindim);
end

else
    lbp = vl_lbp(img,lbp_parameter);
end

lbp = reshape(lbp,[],1);







end

