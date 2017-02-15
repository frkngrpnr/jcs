function [output] = LAPFI_train_KFSI(dset)
% k-fold subject independent cross validation on training set.
%% 0. Input validation
verbose = 1;
%% 1. Initialization
if ~isfield(dset, 'labelset')
    dset = LAPFI_attach_labels(dset, evalin('base', 'gt_train'));
end
dset = clean_dataset(dset, find(dset.fold~=1));
%[output, opts] = initialize_output(dset)   
output = rmfield(dset, 'data');
opts = struct('classifier','kELM',  'kernel_type', 'lin_kernel',  'normtype', 'MMN');
opts.rPCA = 0;
opts.search_paramnames = {'kp', 'C'};
%opts.search_grid = {10.^[-2:4:8], 10.^[-4:1:2]}; if strcmp(opts.kernel_type, 'lin_kernel'), opts.search_grid{1} = 42; end
opts.search_grid = {10.^[-2:4:12], 10.^[-4:1:2]}; if strcmp(opts.kernel_type, 'lin_kernel'), opts.search_grid{1} = 42; end
output.opts = opts;
output.reg_model = opts; 
for li=1:5
    output.models{li} = opts;
    output.models{li}.labelname = dset.labelnames{li};
    output.models{li}.MAE = 5^6;
    output.models{li}.predset = dset.labelset;
end
%trains = find(dset.fold==1); vals = find(dset.fold==2); tests = find(dset.fold==3);
%% 2. For each label, do K-fold subj. ind. cross validation
K = 5;
N = size(dset.data,1);
%[output.fold, unique_subject_ids] = subject_ids_to_kfold(s.subject_id, K);
[output.fold] = kFoldSubjectIndependent(dset.subject_id, K);

%% 3.1. grid search for hyper-parameters
trainkernels = cell(1,K);
testkernels = cell(1,K);
for pi1=1:numel(opts.search_grid{1})
    param1 = opts.search_grid{1}(pi1);
    for pi2=1:numel(opts.search_grid{2})
        param2 = opts.search_grid{2}(pi2);
        predset = zeros(N, 5);
        for k=1:K
            trains = find(output.fold~=k);
            tests = find(output.fold==k);
            if pi2==1 % compute kernels once
                traindata = dset.data(trains,:);
                testdata = dset.data(tests,:);
                if isfield(opts, 'rPCA') && opts.rPCA > 0
                    [traindata, output.norm_model, testdata] = fg_normalizeData(traindata, opts.normtype, testdata, k*pi2*pi1==1);
                    % a small cheat, use the first model for the rest K-1 folds
                    if k>1, output.pca_models{k} = output.pca_models{k-1}; [traindata] = fg_pca(traindata, output.pca_models{k}); else
                    [traindata, output.pca_models{k}] = fg_pca(traindata, opts.rPCA, 1, 1);
                    end
                    [testdata] = fg_pca(testdata, output.pca_models{k});
                end
                %traindata(isnan(traindata))=0; traindata(isinf(traindata))=0; testdata(isnan(testdata))=0; testdata(isinf(testdata))=0; % clean nan and inf values
                [traindata, output.norm_model, testdata] = fg_normalizeData(traindata, opts.normtype, testdata, k*pi2*pi1==1);
                trainkernels{k} = ELM_kernel_matrix(traindata, opts.kernel_type, param1);
                testkernels{k} = ELM_kernel_matrix(traindata, opts.kernel_type, param1, testdata);  
            end % if pi2==1, compute kernels
            for li=1:numel(output.labelset)
                [TrainingTime, TestingTime, ~, TestingAccuracy,Y,TY,~] = elm_kern(trainkernels{k}', dset.labelset{li}(trains), testkernels{k}', dset.labelset{li}(tests), 0, param2, li*k*pi1*pi2==1);
                %TrainingTime
                %TestingTime
                predset(tests,li) = TY';
            end
        end
        % all k-folds have been processed, compute accuracy
        updated=0;
        maes = zeros(1,5);
        for li=1:numel(output.labelset)
            stats = regressionStats(output.labelset{li}, predset(:,li));
            maes(li) = stats.MAE;
            % update each task's model separately
            if stats.MAE < output.models{li}.MAE
                updated=1;
                output.models{li}.MAE = stats.MAE;
                output.models{li}.(opts.search_paramnames{1}) = param1;
                output.models{li}.(opts.search_paramnames{2}) = param2;
                output.models{li}.pred_label = predset(:,li);
                output.models{li}.pred_truth = output.labelset{li};
                output.models{li}.pred_subject_id = output.subject_id;
                output.models{li}.pred_filename = output.filename;
                output.models{li}.pred_fold = output.fold;
            end
        end
        fprintf('score = %s (mean (%s)) for [kp,C]=%s',num2str(1-mean(maes),3),mat2str(1-maes,3),mat2str([param1 param2]));
        if updated, fprintf('<--- new best'); end
%         if mean(maes) < output.reg_model.MAE
%             fprintf('<-- new best');
%             output.reg_model.MAE = mean(maes);
%             output.reg_model.(opts.search_paramnames{1}) = param1;
%             output.reg_model.(opts.search_paramnames{2}) = param2;
%             output.reg_model.pred_labelset = predset;
%             output.reg_model.pred_truthset = output.labelset;
%             output.reg_model.pred_subject_id = output.subject_id;
%             output.reg_model.pred_filename = output.filename;
%         end
        fprintf('\n');
        
    end
    fprintf('----------------\n');
end

output.challenge_scores=zeros(1,6);
for li=1:5
    output.challenge_scores(li+1) = 1-output.models{li}.MAE;
end
output.challenge_scores(1) = mean(output.challenge_scores(2:6));

end % main function

