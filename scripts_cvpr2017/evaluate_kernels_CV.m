function [best_perf_6dims,best_pars_6dims,scores_elm_trval,gt_trval] =evaluate_kernels_CV(mycel_trainfolds,lopts,dimensions)
% given the nComp_set for PLS and C_set
Nfolds=numel(mycel_trainfolds);
%mycel_trainfolds
do_pls=1;
do_elm=1;
do_cca=0;
best_perf_6dims=zeros(6,2);
best_pars_6dims=zeros(6,2);
scores_elm_trval=zeros(numel(mycel_trainfolds{1, 1}.trainind),6);
gt_trval=zeros(numel(mycel_trainfolds{1, 1}.trainind),6);

for cl=1:6
disp(dimensions{cl});
perf_methods=zeros(numel(lopts.C_set),2); 
for l=1:numel(lopts.C_set)
    for i=1:Nfolds
        [mycel_trainfolds{i,1}.best_linperf,mycel_trainfolds{i,1}.best_linmodels]=...
            eval_methods_rmse(mycel_trainfolds{i,1}.train_kernel,mycel_trainfolds{i,1}.trainlabels(:,cl),...
            mycel_trainfolds{i,1}.val_kernel,mycel_trainfolds{i,1}.testlabels(:,cl),lopts.C_set(l),lopts.nComp_set(l),do_pls,do_elm,do_cca);    

    end

    % compute overall corr

    gt_test=[];
    preds_elm_test=[];
    preds_pls_test=[];
    preds_cca_test=[];
   
    for i=1:Nfolds
        gt_test=[gt_test;mycel_trainfolds{i, 1}.testlabels(:,cl)];
        if (do_elm)
            preds_elm_test=[preds_elm_test;mycel_trainfolds{i, 1}.best_linmodels.best_pred_elm'];
        end
        if (do_pls)
            preds_pls_test=[preds_pls_test;mycel_trainfolds{i, 1}.best_linmodels.best_pred_pls'];
        end
        if do_cca
            preds_cca_test=[preds_cca_test;mycel_trainfolds{i, 1}.best_linmodels.best_pred_cca'];
        end
    end
     %compute corr
    gt_trval(:,cl)=gt_test;
    if (do_pls)
        [RMSE_pls,MAE_pls]=rmse(gt_test,preds_pls_test);
        perf_methods(l,1)=MAE_pls;
    end
    if (do_elm)
        [RMSE_elm,MAE_elm]=rmse(gt_test,preds_elm_test);
        perf_methods(l,2)=MAE_elm;
        scores_elm_trval(:,cl)=preds_elm_test;
    end
    if do_cca
        [RMSE_cca,MAE_cca]=rmse(gt_test,preds_cca_test);
        perf_methods(l,3)=MAE_cca;
    end

end
[best_perf_overall,min_ind]=min(perf_methods,[],1);
perf_methods=[lopts.nComp_set' perf_methods lopts.C_set'];
disp(dimensions{cl});
best_perf_overall
best_perf_6dims(cl,:)=best_perf_overall;
best_Params=[lopts.nComp_set(min_ind(1)) lopts.C_set(min_ind(2))]
best_pars_6dims(cl,:)=best_Params;
end

