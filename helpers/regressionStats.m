function stats = regressionStats(y,yhat, stdevs)
% INPUT
% y = true labels
% yhat = predicted labels
% stdev (optional) : returns normal score too
% 
% OUTPUT
% stats is a structure array with fields MAE, RMSE, normalScore etc
%
y = reshape(y,[],1);
yhat = reshape(yhat,[],1);

stats = struct;
stats.MAE = sum(abs(y-yhat))/numel(yhat);
stats.RMSE = sqrt(mean((y - yhat).^2));

if nargin==3 % compute normal score with stdevs:
    todo = stdevs;
    stats.normalScore = todo;    
end


end
    