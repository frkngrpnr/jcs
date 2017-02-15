function [best_perf,best_models]=eval_methods_rmse(train_kernel,trainlabels,val_kernel,vallabels,C_set,nComp,do_pls,do_elm,do_cca)
    if (nargin<7)
       do_pls=1;
       do_elm=1;
       do_cca=1;
    end
    if nargin<6
        nComp=20;
    end
        
    if nargin < 5
        C_set=1;
    end
    best_perf=zeros(1,3);
    if (do_pls)
        best_score_pls=[];
        best_perf_pls=intmax;
        bestComp_pls=0;
        for ncomp =nComp
            [score_pls,rmse_pls] = PLSreg(train_kernel,trainlabels,val_kernel,vallabels,ncomp);
            if (best_perf_pls>rmse_pls)
                best_perf_pls=rmse_pls;
                best_score_pls=score_pls';
                bestComp_pls=ncomp;
            end
        end

        best_models.best_score_pls=best_score_pls;
        best_models.best_pred_pls=best_score_pls;

        best_models.best_correlation_pls=best_perf_pls;
        best_models.bestComp_pls=bestComp_pls;
        best_perf(1,1)=best_perf_pls;
    end
    %C_set=2.^[-15:15]; % 
    if do_elm
        perfomance_elm=zeros(size(C_set));
        best_score_elm=[];
        best_perf_elm=intmax;
        bestC_elm=0;
        j=0;
        for C=C_set
            j=j+1;
            mu_Y=mean(trainlabels);
            [~, ~, TrainingError, TestingAccuracy,Y,TY] = elm_kern(train_kernel, trainlabels-mu_Y, val_kernel, vallabels-mu_Y, 0, C);
            rmse_elm = rmse(vallabels,TY'+mu_Y);
            fprintf('RMSE ELM= %f\n',rmse_elm);
            perfomance_elm(j)=rmse_elm;
            if (best_perf_elm>rmse_elm)
                best_perf_elm=rmse_elm;
                best_score_elm=TY+mu_Y;
                bestC_elm=C;
            end
        end

        best_models.best_score_elm=best_score_elm;
        best_models.best_pred_elm=best_score_elm;
        %best_models.best_UAR_elm=getUAR(vallabels,best_models.best_pred_elm');
        best_models.best_correlation_elm=best_perf_elm;
        best_models.bestC_elm=bestC_elm;

        best_perf(1,2)=best_perf_elm;
    end
    %best_models.best_UAR_elm;
%     
 if do_cca
    best_score_cca=[];
    best_perf_cca=-1;

    [score_cca,rmse_cca] = CCAreg(train_kernel,trainlabels,val_kernel,vallabels);
    if (best_perf_cca>rmse_cca)
        best_perf_cca=rmse_cca;
        best_score_cca=score_cca';
    end
    
    
    best_models.best_score_cca=best_score_cca;
    best_models.best_pred_cca=best_score_cca;
    
    best_models.best_correlation_cca=best_perf_cca;
    
    best_perf(1,3)=best_perf_cca;
 end
end
 
