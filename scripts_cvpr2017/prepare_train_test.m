function [mycel_testfold]=prepare_train_test(all_labels_train)

mycel_testfold=cell(1,1);


N=size(all_labels_train,1);
clear all_labels

for i=1:1
    mycel_testfold{i}.testlabels=randn(2000,6)*0.125+0.5;
    mycel_testfold{i}.trainlabels=all_labels_train;
end

end