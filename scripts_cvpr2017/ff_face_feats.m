 if ~(exist('vggfer33fun','var'))
        disp('Loading vggfer33fun..');
        load('vggfer33fun'); % face feature
 end
 if ~(exist('lgbptop', 'var'))
        disp('Loading lgbptop..');
        load('lgbptop'); % face feature
        
 end
 
    
% popt: pre-processing and kernel options
popt.norm_type=1;%1:min-max, 2: z-norm
popt.do_power_norm=0;
popt.do_logsig=0;
popt.do_imp_pp=0;
popt.kernel_type=1; % 1= linear, 2= rbf
popt.gamma=5e-4; % needed for rbf kernel

% prepare folds
[mycel_trainfolds,all_labels_train] =prepare_train_CV(Nfold);

trdata= [lgbptop.data(lgbptop.fold==1,:)  vggfer33fun.data(vggfer33fun.fold==1,:);...
        lgbptop.data(lgbptop.fold==2,:)  vggfer33fun.data(vggfer33fun.fold==2,:)];

clear lgbptop vggfer33fun 
% form kernels once for each pre-processing option
[mycel_trainfolds]=compute_kernels_CV(mycel_trainfolds,trdata,popt);

% learner (regressor) parameter options : nComp_set for Partial Least Squares Regression, 
% C_set is for Kernel Extreme Learning Machines
% if you provide a single pair for  nComp_set and C_set scores_elm_trval
% becomes menaningful (providing the predictions for the last pair of hyper-parameres
% if you provide a set (of equal lenght for both params)
% best_perf_6dims,best_pars_6dims will give you the best MAE and
% corresponding hyper-parameters, respectively

lopts.nComp_set=24; % optimized over 2:2:20 if at the boundary then 22:2:40
lopts.C_set=5*10.^-4;% optimized over  5*10.^[-5:4]; 
[best_perf_6dims,best_pars_6dims,scores_elm_trval_ff_lgbptop_vggfer33] =evaluate_kernels_CV(mycel_trainfolds,lopts,dimensions);
best_Accuracy_ff_lgbptop_vggfer33=1-best_perf_6dims
clear mycel_trainfolds
if numel(lopts.nComp_set)==1 
    
    savepath=[opts.output_path opts.bss 'scores_elm_trval_ff_lgbptop_vggfer33.mat'];
    save(savepath,'scores_elm_trval_ff_lgbptop_vggfer33')

     if ~(exist('vggfer33fun_test','var'))
        disp('Loading vggfer33fun_test..');
        load('vggfer33fun_test'); % face feature
     end
     if ~(exist('lgbptop_test', 'var'))
            disp('Loading lgbptop_test..');
            load('lgbptop_test'); % face feature

     end
 
    % apply the optimized system on val/test data
    testdata=[lgbptop_test.data  vggfer33fun_test.data];
    clear lgbptop_test vggfer33fun_test 
    [mycel_testfold]=prepare_train_test(all_labels_train);

    [mycel_testfold]=compute_kernels_test(mycel_testfold,trdata,testdata,popt);
    clear trdata testdata 
    % test set labels are set randomly therefore the results aren't meaningful
    [scores_elm_test_ff_lgbptop_vggfer33]=evaluate_kernels_test(mycel_testfold,lopts,dimensions);
    clear mycel_testfold
    savepath=[opts.output_path opts.bss 'scores_elm_test_ff_lgbptop_vggfer33.mat'];
    save(savepath,'scores_elm_test_ff_lgbptop_vggfer33')
end
clear trdata 