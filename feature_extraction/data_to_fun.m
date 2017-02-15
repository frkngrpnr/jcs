function [fun, funids, featids] = data_to_fun(data)
% Input : data, column-observations extracted from a video. each column is the descriptor for a frame.
% Output : returns functional parameteres of data. features include :
% mean, range, first 3 parameters of fitted polynomial
% statistics are extracted from averaging over frames (columns)

if numel(data)==0; fun=[]; funids=[]; featids=[]; return, end

%F = size(data,2);% number of observations
[D, F] = size(data); % dimension, samples
%fprintf('FUN-encoding %d features w. %d samples\n',D,F);

if F<3
    %disp('F < 3. Something strange will happen !');
    fun = zeros(size(data,1)*5,1); % !! make sure to implement this exception for caller.
    funids = [ones(D,1)*1; ones(D,1)*2; ones(D,1)*3; ones(D,1)*4; ones(D,1)*5];
    featids = repmat([1:D]',5,1);
    return;
end

% Polyfit requires > 3 

fun_mean = mean(data, 2);
fun_std = std(data, 0, 2);


x_set=1:F;
data_poly = zeros(size(data,1),3);
%disp('size data,1');
%size(data,1)
for i = 1:size(data,1) % for each feature
   [p] = polyfit(x_set,data(i,:),2); % check if frames < 2
   %curvature (second degree poly coeff a)
   data_poly(i,1) = p(1);
   [p] = polyfit(x_set,data(i,:),1);
   %slope of first degree poly
   data_poly(i,2)=p(1);
   %offset of first degree poly
   data_poly(i,3)=p(2); 
   %i
end
%size(fun_mean)
%size(fun_std)
%size(data_poly)
%disp('');
%fprintf(['size(fun_mean) = [' int2str(size(fun_mean,1)) ' , ' int2str(size(fun_mean,2)) '] whereas size(data_poly) = [' int2str(size(data_poly,1)) ' , ' int2str(size(data_poly,2)) '] \n'  ]);
%data_poly



%% Zero-Crossing Rate after z-normalization :
%% For all features :
% [xn,mx,stdx] = autosc(data);
% fun_zcr = ZCR(xn')';
%% For PCA-reduced features :
% [rData, PCA_Model] = reduceDim_Train(data, 3);
% [xn,mx,stdx] = autosc(rData);
% fun_zcr = ZCR(xn')';
%% For PCA-reduced features from pre-learned PCA model :
% [rData] = reduceDim_Test(data, PCA_Model);
% [xn,mx,stdx] = autosc(rData);
% fun_zcr = ZCR(xn')';

%% Range
%range = 'TODO';
%fun_range=max(data,[],2)-min(data,[],2);

data_poly_column = reshape(data_poly,[],1);
%fun = [fun_mean; fun_std; data_poly_column; max(data,[],2); min(data,[],2)];
fun = [fun_mean; fun_std; data_poly_column];
funids = [ones(D,1)*1; ones(D,1)*2; ones(D,1)*3; ones(D,1)*4; ones(D,1)*5];
featids = repmat([1:D]',5,1);
%fun = [fun_std; data_poly_column; max(data,[],2); min(data,[],2)];
%fun = reshape(fun, prod(size(fun)), 1);
end