function [gt] = LAPFI_read_gt_csv(filepath)

gt = struct;
x = importdata(filepath);

gt.VideoName = x.textdata(2:end,1);
gt.ValueExtraversion = x.data(:,1);
gt.ValueAgreeableness = x.data(:,2);
gt.ValueConscientiousness = x.data(:,3);
gt.ValueNeurotisicm = x.data(:,4);
gt.ValueOpenness = x.data(:,5);
end

