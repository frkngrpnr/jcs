function [prediction,rmse_perf,model] = CCAreg(traindata,trainlabel,testdata,testlabel)
% ====================================================================================
% 
% 
% Author:Heysem Kaya @ BU/CmpE 
% E-mail: heysem@boun.edu.tr
% 
% Jan.23, 2015
% ====================================================================================

model.mn_X=mean(traindata);

model.mn_Y=mean(trainlabel);
[A,B,r] = canoncorr(traindata,trainlabel);
model.beta=A/B;
prediction = (testdata-repmat(model.mn_X,size(testdata,1),1))*model.beta+repmat(model.mn_Y,size(testdata,1),1);

%correl = corr(testlabel,prediction,'Type','Spearman');
%fprintf('Spearman Rank Correlation CCA = %f\n',correl);
rmse_perf = rmse(testlabel,prediction);
fprintf('RMSE CCA = %f\n',rmse_perf);