function [scores_elm]=evaluate_kernels_test(mycel_testfold,lopts,dimensions)
do_pls=0;
do_elm=1;
do_cca=0;
scores_elm=zeros(2000,6);

for cl=1:6
disp(dimensions{cl});

[mycel_testfold{1,1}.best_linperf,mycel_testfold{1,1}.best_linmodels]=...
    eval_methods_rmse(mycel_testfold{1,1}.train_kernel,mycel_testfold{1,1}.trainlabels(:,cl),...
    mycel_testfold{1,1}.val_kernel,mycel_testfold{1,1}.testlabels(:,cl),lopts.C_set(1),lopts.nComp_set(1),do_pls,do_elm,do_cca);    

% compute overall corr

if (do_elm)
    preds_elm_test=[mycel_testfold{1,1}.best_linmodels.best_pred_elm'];
end
if (do_pls)
    preds_pls_test=[mycel_testfold{1,1}.best_linmodels.best_pred_pls'];
end
if do_cca
    preds_cca_test=[mycel_testfold{1,1}.best_linmodels.best_pred_cca'];
end

scores_elm(:,cl)=preds_elm_test;

end
end
