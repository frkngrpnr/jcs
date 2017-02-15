function [groups] = kFoldSubjectIndependent(subject_names, K)
%% Input :
% subject_ids should be a Nx1 cell or array
%% Output :
% output groups [Nx1] will contain group indices, i.e. each member btw. 1 and K
%% Implementation Details : 
% This code will simply get the unique subject ids and divide it to k
% folds, then assign each subject's samples to the according group. So it
% will assume similar number of samples per subject.

% * comments will accompany with example contents
% * subject_names = {'a','a','a','b','b'}
N = numel(subject_names);
groups = zeros(N,1);
unique_names = unique(subject_names);
% * unique_names = {'a','b'}
indices = crossvalind('Kfold',numel(unique_names),K);

for i=1:numel(unique_names)
    this_ind = indices(i);
    this_name = unique_names{i};
    % assign this index to each member where subject_names has this name
    groups(find(strcmp(this_name, subject_names)),:) = this_ind;
end

if sum(groups==0)
    error('this shouldn t have happened..');
end



% run simple crossvalind code for unique subjects

end

