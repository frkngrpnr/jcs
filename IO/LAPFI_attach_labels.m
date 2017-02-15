function [ldset] = LAPFI_attach_labels(dset, gt_train)
if nargin<2, gt_train = evalin('base','gt_train'); end
labelnames = {'ValueExtraversion', 'ValueAgreeableness', 'ValueConscientiousness', 'ValueNeurotisicm', 'ValueOpenness'};
ldset = dset;
% also attach subject_ids
N = size(dset.data,1);
ldset.subject_id = cell(N,1); 
for i=1:N
    str=dset.filename{i}; 
    % here, check if str has something after '.mp4', and if so, remove it:
    if ~strcmp(str(end-3:end), '.mp4')
        fprintf('string (%s) does not end with .mp4\n',str);
        str(strfind(str, '.mp4')+4:end)=[]
        pause
    end
    %str = str
    %str = fix_end(str, '.mp4'); str(1)=[];
    dset.filename{i} = str;
    ldset.filename{i} = str;
    
    ldset.subject_id{i} = str(1:end-8); 
end

% also attach folds:
if ~isfield(ldset,'fold')
    ldset.fold = 2*ones(size(ldset.data,1),1);
end
for li=1:numel(labelnames)
    %lfset.label = gt_train.(labelnames{li}); % nope
    %ldset.label = zeros(size(ldset.data,1),1);
    ldset.label = dset.label;
    origfield = gt_train.(labelnames{li});
    for samplei=1:size(ldset.data,1)
        waitbar(samplei/size(ldset.data,1));
        m = find(strcmp(ldset.filename{samplei}, gt_train.VideoName));
        if numel(m)==0 % not a training sample, remove it
            ldset.filter(samplei,:) = 0;
        else
            ldset.label(samplei) = origfield(m);
            ldset.fold(samplei) = 1;
            ldset.filter(samplei,:) = 1;
        end
        
    end
    
    ldset.labelset{li} = ldset.label;
    ldset.labelnames{li} = labelnames{li};
    
    %fprintf('|%s|\n',labelnames{li});
    %[output{li}] = LAPFI_CVonTrain(ldset, gt_train);
    %output{li}.labelname = labelnames{li};
end

ldset.set=ldset.fold;

end

%% Helpers
%function new_str = fix_end(str, wanted_ending) % 'x.jpg' = fix_end('x.jpg.mat', '.jpg')
%    new_str = [strtok(str,wanted_ending), wanted_ending];
%end
% function new = fix_end(str, ending)
%     new = regexprep(str, ['(?<=', regexptranslate('escape', ending), ').*$'], '');
% end
function new = fix_end(str, ending)
    new = regexprep(str, ['(?<=', regexptranslate('escape', ending), ').*'], '');
end
