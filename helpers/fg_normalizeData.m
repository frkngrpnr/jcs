function [ntraindata, norm_model, ntestdata] = fg_normalizeData(traindata, normtype, testdata, verbose)
% [ntraindata, norm_model, ntestdata] = fg_normalizeData(traindata, 'MMN', testdata, 1);
% or 
% [data, norm_model] = fg_normalizeData(data, 'MMN');
% or for test data:
% [ntestdata] = fg_normalizeData(testdata, norm_model);
%% normtype:
% can be a string like 'MMN', or 'MMN+L2' for chained normalization
% can be a struct, i.e. the output norm_model
% can be a cell of structs for chained norm.
%% data should be N x D (samples x features)
% normtype should be a string. options:
% 'minmax'
% 'znorm'
% 'power'
if nargin<4, verbose=1; end
ntestdata=[]; if nargin<3, testdata = traindata(1,:); verbose=0; end
if numel(testdata)==0, testdata = traindata(1,:); end

if isa(normtype, 'char') % training set
    C = strsplit(normtype, '+');
    if numel(C) == 1 % just one normalization
        [ntraindata, norm_model, ntestdata] = normalize_once_train(traindata, normtype, testdata, verbose);
        norm_model.name = normtype;
    else % chain normalization
        norm_models = cell(1,numel(C));
        ntraindata = traindata; ntestdata = testdata;
        for i=1:numel(C)
            [ntraindata, norm_models{i}, ntestdata] = normalize_once_train(ntraindata, C{i}, ntestdata, verbose);
            norm_models{i}.name = C{i};
        end
        norm_model = norm_models;
    end
elseif isa(normtype, 'struct') % test set, one norm.model
    norm_model = normtype;
%     ntraindata = fg_normalizeData(traindata, norm_model.name);
%     if nargin>2
%     ntestdata = fg_normalizeData(testdata, norm_model.name);
%     end
    [ntraindata, ~, ntestdata] = normalize_once_test(traindata, norm_model, testdata, verbose);
elseif isa(normtype, 'cell') % test set, chained norm. models
    norm_model = normtype;
    ntraindata = traindata;
    ntestdata = testdata;
    for i=1:numel(norm_model)
        [ntraindata, ~, ntestdata] = fg_normalizeData(ntraindata, norm_model{i}, ntestdata, verbose); % recurse
    end
end % if normtype is a char or struct (or cell of structs)



end % main function

%% Helpers
function [ntraindata, norm_model, ntestdata] = normalize_once_test(traindata, norm_model, testdata, verbose)
normtype = norm_model.name;
if strcmp(normtype, 'nonorm') || strcmp(normtype, 'NN')
    ntraindata=traindata;
    ntestdata=testdata;
    norm_model = struct;
elseif strcmp(normtype, 'minmax') || strcmp(normtype, 'mapminmax') || strcmp(normtype, 'MMN') % mapminmax normalizes each row, so transpose the data : 
    if verbose, fprintf('min-max normalizing %d features\n',size(traindata,2)); end
    ntraindata = mapminmax.apply(traindata', norm_model)';
    if nargin>2
        ntestdata = mapminmax.apply(testdata', norm_model)';
    end    
elseif strcmp(normtype, 'znorm') || strcmp(normtype, 'z') || strcmp(normtype, 'ZN')
    if verbose, fprintf('z-normalizing %d features\n',size(traindata,2)); end
    ntraindata = scal(traindata, norm_model.mx, norm_model.stdx);
    if nargin>2, ntestdata = scal(testdata, norm_model.mx, norm_model.stdx); end
elseif strcmp(normtype, 'power') || strcmp(normtype, 'pow') || strcmp(normtype, 'Pow')
    if verbose, fprintf('power-normalizing %d features\n',size(traindata,2)); end
    p=2;
    ntraindata=sign(traindata).*abs(traindata).^(1/p);
    if nargin>2, ntestdata=sign(testdata).*abs(testdata).^(1/p); end
    norm_model=struct;
elseif strcmp(normtype, 'sigmoid') || strcmp(normtype, 'sig') || strcmp(normtype, 'Sig') 
    if verbose, fprintf('sigmoid-normalizing %d features\n',size(traindata,2)); end
    ntraindata = logsig(traindata);
    if nargin>2
        ntestdata = logsig(testdata);
    end, norm_model = struct;
elseif strcmp(normtype, 'L2') || strcmp(normtype, 'l2')
    if verbose, fprintf('L2-normalizing %d features\n',size(traindata,2)); end
    ntraindata = zeros(size(traindata));
    for n=1:size(traindata,1)
        x = traindata(n,:);
        ntraindata(n,:) = x/norm(x);
    end
        % repeat for test data:
        if nargin>2
        ntestdata = zeros(size(testdata));
        for n=1:size(testdata,1)
            x = testdata(n,:);
            ntestdata(n,:) = x/norm(x);
        end
        end
    norm_model = struct;
else
    error(['Unknown norm.type: ' normtype]);
end
end


function [ntraindata, norm_model, ntestdata] = normalize_once_train(traindata, normtype, testdata, verbose)
norm_model = struct;
if strcmp(normtype, 'nonorm') || strcmp(normtype, 'NN')
    ntraindata=traindata;
    ntestdata=testdata;
elseif strcmp(normtype, 'minmax') || strcmp(normtype, 'mapminmax') || strcmp(normtype, 'MMN') % mapminmax normalizes each row, so transpose the data : 
    if verbose, fprintf('min-max normalizing %d features\n',size(traindata,2)); end
    [ntraindata, norm_model] = mapminmax(traindata', 0, 1);
    ntraindata = ntraindata';
    if nargin>2
        ntestdata = mapminmax.apply(testdata', norm_model)';
    end    
elseif strcmp(normtype, 'znorm') || strcmp(normtype, 'z') || strcmp(normtype, 'ZN')
    if verbose, fprintf('z-normalizing %d features\n',size(traindata,2)); end
    [ntraindata, mx,stdx] = autosc(traindata);
    if nargin>2, ntestdata = scal(testdata,mx,stdx); end
    norm_model = struct;
        norm_model.mx=mx;
        norm_model.stdx=stdx;
elseif strcmp(normtype, 'power') || strcmp(normtype, 'pow') || strcmp(normtype, 'Pow')
    if verbose, fprintf('power-normalizing %d features\n',size(traindata,2)); end
    p=2;
    ntraindata=sign(traindata).*abs(traindata).^(1/p);
    if nargin>2, ntestdata=sign(testdata).*abs(testdata).^(1/p); end
    norm_model=struct;
elseif strcmp(normtype, 'sigmoid') || strcmp(normtype, 'sig') || strcmp(normtype, 'Sig') 
    if verbose, fprintf('sigmoid-normalizing %d features\n',size(traindata,2)); end
    ntraindata = logsig(traindata);
    if nargin>2
        ntestdata = logsig(testdata);
    end, norm_model = struct;
elseif strcmp(normtype, 'L2') || strcmp(normtype, 'l2')
    if verbose, fprintf('L2-normalizing %d features\n',size(traindata,2)); end
    ntraindata = zeros(size(traindata));
    for n=1:size(traindata,1)
        x = traindata(n,:);
        ntraindata(n,:) = x/norm(x);
    end
        % repeat for test data:
        if nargin>2
        ntestdata = zeros(size(testdata));
        for n=1:size(testdata,1)
            x = testdata(n,:);
            ntestdata(n,:) = x/norm(x);
        end
        end
    norm_model = struct;
else
    error(['Unknown norm.type: ' normtype]);
end
end % function normalize_once
