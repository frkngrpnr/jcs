function [prediction,oob_perf,ens_models ] = plsregens(traindata,trainlabels, options )
% A Random Forest like ensemble for PLS regression
[N,d]=size(traindata);
if nargin<3
  T=1;
  numComp=1;
  do_feature_sampling=0;
  max_feats=d;

else
if (isfield(options,'numTrees')) 
    T=options.numTrees;
else
    T=1;
end

if (isfield(options,'numComp')) 
    numComp=max(options.numComp,1);
    numComp=min(numComp,d);
else
    numComp=1;
end

if (isfield(options,'do_feature_sampling')) 
    do_feature_sampling=options.do_feature_sampling;
else
    do_feature_sampling=0;
end

if (isfield(options,'numFeats')) 
    max_feats=max(options.numFeats,1);
    max_feats=min(max_feats,d);
else
    do_feature_sampling=0;
    max_feats=d;
end
end

ens_models=cell(T,1);
indic_oob=ones(N,T);
ens_preds=zeros(N,size(trainlabels,2));

for t=1:T
    
    sampling=randi(N,N,1);
    tmp_train_data=traindata(sampling,:);
    tmp_train_labels=trainlabels(sampling,:);
    indic_oob(unique(sampling),t)=0;
    ens_models{t,1}.indic_oob=indic_oob(:,t)==1;
    ens_models{t,1}.feats=1:d;
    
    if (do_feature_sampling)
        feat_perm=randperm(d,max_feats);
        ens_models{t,1}.feats=feat_perm;
    end
    
    % finally use a regressor
    [ens_models{t,1}.prediction,ens_models{t,1}.rmse_perf,ens_models{t,1}.model] = ...
    PLSreg(tmp_train_data(:,ens_models{t,1}.feats),tmp_train_labels,...
    traindata(ens_models{t,1}.indic_oob,ens_models{t,1}.feats),trainlabels(ens_models{t,1}.indic_oob,:),numComp);
    ens_preds(ens_models{t,1}.indic_oob,:)=ens_preds(ens_models{t,1}.indic_oob,:)+...
    ens_models{t,1}.prediction;
end

tot_oob=sum(indic_oob,2);

for i=1:N
    if (tot_oob(i)>0)
        ens_preds(i,:)=ens_preds(i,:)/tot_oob(i);
    end
end

prediction=ens_preds;
tDims=size(prediction,2);
t_indic=tot_oob>0;
oob_perf=zeros(1,tDims);
for i=1:tDims
   oob_perf(i)=rmse(trainlabels(t_indic,i),prediction(t_indic,i)); 
    
end
