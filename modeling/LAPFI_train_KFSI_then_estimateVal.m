function [output] = LAPFI_train_KFSI_then_estimateVal(dset)
dset = LAPFI_attach_labels(dset);
trainset = clean_dataset(dset, find(dset.fold~=1));
%trainset = clean_dataset(trainset, 1:5500);% temp
output.train_KFSI_output = LAPFI_train_KFSI(trainset);
x=42;

%% Estimate val set:
predstruct = struct;
for li=1:5
    predstruct.labelnames{li} = output.train_KFSI_output.models{li}.labelname;
    param1 = output.train_KFSI_output.models{li}.kp;
    param2 = output.train_KFSI_output.models{li}.C;
    trains = find(dset.fold==1);
    tests = find(dset.fold==2);
    traindata = dset.data(trains,:);
    testdata = dset.data(tests,:);
    [traindata, output.norm_model, testdata] = fg_normalizeData(traindata, output.train_KFSI_output.models{li}.normtype, testdata, 0);
    trainkernel = ELM_kernel_matrix(traindata, output.train_KFSI_output.models{li}.kernel_type, param1);
    testkernel = ELM_kernel_matrix(traindata, output.train_KFSI_output.models{li}.kernel_type, param1, testdata); 
    [~, ~, ~, TestingAccuracy,Y,TY,~] = elm_kern(trainkernel', dset.labelset{li}(trains), testkernel', dset.labelset{li}(tests), 0, param2, 0);
    predstruct.predset{li} = TY';
    predstruct.pred_filename = dset.filename(tests);
end

LAPFI_write_predictions(predstruct);

end % main function
