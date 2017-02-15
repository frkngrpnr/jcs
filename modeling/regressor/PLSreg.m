function [prediction,rmse_perf,model] = PLSreg(traindata,trainlabel,testdata,testlabel,ncomp)
    model.mu_Y=mean(trainlabel);
    trainlabel_n=trainlabel-repmat(model.mu_Y,size(traindata,1),1);
    [~,~,~,~,model.beta] = plsregress(traindata,trainlabel_n,ncomp);
    
    prediction = [ones(size(testdata,1),1) testdata]*model.beta+repmat(model.mu_Y,size(testdata,1),1);
    %correl = corr(testlabel,prediction,'type','Spearman');
    %fprintf('Spearman Rank Correlation PLS= %f\n',correl);
    rmse_perf = rmse(testlabel,prediction);
    fprintf('RMSE PLS= %f\n',rmse_perf);
end
