function [newdset, rids] = clean_dataset(dset, rids)
if nargin>1 && numel(rids)==0, newdset=dset; return; end
%% 1, rids not given: calculate
N = size(dset.data,1);
if nargin<2
    rids = [];
    
    min_face_score = 0.2;

    if isfield(dset, 'face_score')    
        newrids = find(dset.face_score < min_face_score);
        warning(['removing ' int2str(numel(newrids)) ' samples w. face_score < ' num2str(min_face_score)]);
        rids = unique([reshape(rids,[],1); reshape(newrids,[],1)]);
    end

    if isfield(dset, 'face_detected')
        newrids = find(dset.face_detected==0);
        warning(['removing ' int2str(numel(newrids)) ' non-detected faces w DPM']);
        rids = unique([reshape(rids,[],1); reshape(newrids,[],1)]);
    end

    if isfield(dset, 'landmarks_detected')
        newrids = find(dset.landmarks_detected==0);
        warning(['removing ' int2str(numel(newrids)) ' non-landmarked samples']);
        rids = unique([reshape(rids,[],1); reshape(newrids,[],1)]);
    end
    
end % if no 2nd argument (rids)

%% Or, rids might be given as a string:
if isa(rids,'char')
    if strcmp(rids,'secondhalf') || strcmp(rids,'sh')
        rids = [(N/2)+1: N];
    elseif strcmp(rids,'firsthalf') || strcmp(rids,'fh')
        rids = [1:(N/2)];
    elseif strcmp(rids,'secondhalf_auto') || strcmp(rids,'sh_auto') || strcmp(rids,'sh_ifmirrored') % in this version, code decides whether the dataset is a mirrored one, and removes the second half if so.
        N = numel(dset.labels);
        if mod(N,2)==0 && ... % there is an even number of labels
                sum(abs( dset.labels(1:N/2) - dset.labels(N/2+1:N) )) == 0 % and the second part is equal to the first part
            %disp('removing the second half because this looks like a mirrored dataset');            
            dset = clean_dataset(dset, 'sh');
        end
    elseif strcmp(rids,'secondhalf_and_auto') || strcmp(rids,'sh_and_auto') || strcmp(rids,'sh_n_a')
        dset = clean_dataset(dset, 'sh');
        dset = clean_dataset(dset);
    else, error(['unknown option: ' rids]);
    end
end

%% A final check, if rids are negative, they are actually indices to keep.
if rids(1)<0
    fprintf('%s.m: rids negative, so converting them to kids (keep ids)\n',mfilename);
    kids = -rids;
    rids = 1:N;
    rids(kids) = [];
end

%% 2, rids given (or calculated by now), remove samples
%% 2.1. iterate through each field, and detect the ones with N rows (or 1xN ones and reshape them) !! nope, dont deal with K x N fields..
newdset = dset;
myfieldnames = fieldnames(dset);
for i = 1:numel(myfieldnames)
    tf = dset.(myfieldnames{i}); % this field
    if size(tf,1) == N
        tf(rids,:) = [];
        newdset.(myfieldnames{i}) = tf;
%     else
%         % do nothing
    end
end

%% 2.1.2 also iterate through cell fields to see if their members are NxD too:
% for fields like labelset in LAPFI

for i = 1:numel(myfieldnames)
    tf = dset.(myfieldnames{i}); % this field
    if isa(tf,'cell')
        for j=1:numel(tf)
            tm = tf{j};
            if size(tm,1) == N
                tm(rids,:)=[];
                newdset.(myfieldnames{i}){j} = tm;
            else % do nothing
            end
        end
    end
end


end

