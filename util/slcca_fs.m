function [ranking_ind, w_sorted,cca_outputs] = slcca_fs(trainingdata,target_labels_cls)
% Sample versus Labels CCA feature selection
% Author: Heysem Kaya (email: kaya.heysem@gmail.com)
% Relevant Paper: CCA based feature selection with Application to
% continuous depression recognition from acoustic speech features, ICASSP 2014
%
% Inputs 
% trainingdata      : numSamples x numDims
% target_labels_cls : target labels matrix
% -- classification > numSamples x (numClasses - 1) arranged in binary coding format
% -- regression     > numSamples x 1
% 
% Outputs
% ranking_ind       : Feature ranking based relative to given feature set 
% w_sorted          : Weights of features in the ranking
% cca_outputs       : struct for cca outputs and training set mean
cca_outputs.mu= mean(trainingdata);
trainingdata=trainingdata-repmat(cca_outputs.mu,size(trainingdata,1),1);

[cca_outputs.A,cca_outputs.B,cca_outputs.r,cca_outputs.U,cca_outputs.V]=canoncorr(trainingdata,target_labels_cls);
% Take absolute value of projection weights (eigen vectors of CCA)
cca_outputs.AbsA=abs(cca_outputs.A);
% replicate canonical correlations (eigenvalues) for fast summation
R=repmat(cca_outputs.r,size(cca_outputs.AbsA,1),1);
% Compute the weighted sum of the absolute eigenvectors and obtain a single
% discriminative saliency vector of features
cca_outputs.w=sum(cca_outputs.AbsA.*R,2);
[w_sorted,ranking_ind]=sort(cca_outputs.w,'descend');
end

