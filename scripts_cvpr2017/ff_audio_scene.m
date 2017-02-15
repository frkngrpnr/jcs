 if ~(exist('vd19','var'))
        disp('Loading vd19..');
        load('vd19'); % face feature
 end
 if ~(exist('OS_IS13', 'var'))
        disp('Loading OS_IS13..');
        load('OS_IS13'); % face feature
        
 end
    
% popt: pre-processing and kernel options
popt.norm_type=2;%1:min-max, 2: z-norm
popt.do_power_norm=1;
popt.do_logsig=0;
popt.do_imp_pp=1;
popt.kernel_type=1; % 1= linear, 2= rbf
popt.gamma=5e-4; % needed for rbf kernel

% prepare folds
[mycel_trainfolds,all_labels_train] =prepare_train_CV(Nfold);

trdata= [OS_IS13.data(OS_IS13.set==1,:)  vd19.data(vd19.set==1,:);...
         OS_IS13.data(OS_IS13.set==2,:)  vd19.data(vd19.set==2,:)];
% form kernels once for each pre-processing option
[mycel_trainfolds]=compute_kernels_CV(mycel_trainfolds,trdata,popt);

% learner (regressor) parameter options : nComp_set for Partial Least Squares Regression, 
% C_set is for Kernel Extreme Learning Machines
% if you provide a single pair for  nComp_set and C_set scores_elm_trval
% becomes menaningful (providing the predictions for the last pair of hyper-parameres
% if you provide a set (of equal lenght for both params)
% best_perf_6dims,best_pars_6dims will give you the best MAE and
% corresponding hyper-parameters, respectively

lopts.nComp_set=22; %16 optimized over 2:2:20, if at the boundary then 22:2:40
lopts.C_set=0.5;% optimized over  10.^[-5:4]; 
[best_perf_6dims,best_pars_6dims,scores_elm_trval_ff_audio_vd19,gt_trval] =evaluate_kernels_CV(mycel_trainfolds,lopts,dimensions);
best_Accuracy_ff_audio_vd19=1-best_perf_6dims
clear mycel_trainfolds
if numel(lopts.nComp_set)==1 % if it is optimized and a single param is specified
    savepath=[opts.output_path opts.bss 'gt_trval.mat'];
    save(savepath,'gt_trval')
    
    savepath=[opts.output_path opts.bss 'scores_elm_trval_ff_audio_vd19.mat'];
    save(savepath,'scores_elm_trval_ff_audio_vd19')
    
    if ~(exist('vd19_test','var'))
        disp('Loading vd19_test..');
        load('vd19_test'); % face feature
    end
    if ~(exist('OS_IS13_test', 'var'))
        disp('Loading OS_IS13_test..');
        load('OS_IS13_test'); % face feature
    end

    % apply the optimized system on val/test data
    testdata=[OS_IS13_test.data  vd19_test.data];

    [mycel_testfold]=prepare_train_test(all_labels_train);

    [mycel_testfold]=compute_kernels_test(mycel_testfold,trdata,testdata,popt);
    clear trdata testdata vd19_test OS_IS13_test
    % the val/test scores heres are set randomly therefore the results aren't meaningful
    [scores_elm_test_ff_audio_vd19]=evaluate_kernels_test(mycel_testfold,lopts,dimensions);

    savepath=[opts.output_path opts.bss 'scores_elm_test_ff_audio_vd19.mat'];
    save(savepath,'scores_elm_test_ff_audio_vd19')
    clear mycel_testfold
end 
clear OS_IS13 vd19 trdata 
