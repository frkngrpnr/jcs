function [ds] = fuse_features(ds1, ds2)
% you can also call like :
% fg_fuseFeatures({ds1, ds2, ds3, ...}) % !!! Don't forget the cell delimiters !
if nargin==1 && isa(ds1,'cell')
    c = ds1;
    if numel(c) == 1
        ds = c{1};
    else
    ds = fuse_features(c{1},fuse_features(c(2:end))); % recurse !
    end
    return
end
%% Else,
%ds1 = fg_fixdatastruct(ds1, size(ds1.data,1));
%ds2 = fg_fixdatastruct(ds2, size(ds2.data,1));
ds = ds1;
%ds.labels = [reshape(ds1.labels,[],1); reshape(ds2.labels,[],1)];
if isfield(ds1, 'labels')
l1 = reshape(ds1.labels,[],1); l2 = reshape(ds2.labels,[],1);
ds.labels = l1;%reshape(ds1.labels,[],1); % since the labels should be the same
elseif isfield(ds1, 'label')
l1 = reshape(ds1.label,[],1); l2 = reshape(ds2.label,[],1);
ds.label = l1;%reshape(ds1.labels,[],1); % since the labels should be the same
end
%if numel(l1) ~= numel(l2) || sum(abs(l1-l2))~=0, error('labels are not the same'); end
% instead of checking equal labels, just check equal NUMBER OF labels
if numel(l1) ~= numel(l2), error('labels are not of the same size'); end

ds.data = [ds1.data, ds2.data]; 

end

