function [mycel_trainfolds,all_labels_train] =prepare_train_CV(Nfold)
mycel_trainfolds=cell(Nfold,1);

all_labels_train=[];
load('anotations_train')
load('anotations_val')
for i=1:6
    tmp_lbl=[anotations_train{1, i}.scores;anotations_val{1, i}.scores];
    all_labels_train=[all_labels_train tmp_lbl];
end

N=size(all_labels_train,1);
clear all_labels

for i=1:Nfold
    sti=1+(i-1)*round(N/Nfold);
    eni=i*round(N/Nfold);
    if (i==Nfold)
       eni=N; 
    end
    t=zeros(N,1);
    t(sti:eni)=1;
    mycel_trainfolds{i}.testind=t==1;
    mycel_trainfolds{i}.trainind=t==0;
    mycel_trainfolds{i}.testlabels=all_labels_train(mycel_trainfolds{i}.testind,:);
    mycel_trainfolds{i}.trainlabels=all_labels_train(mycel_trainfolds{i}.trainind,:);
end
end
