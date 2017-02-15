function [noout] = LAPFI_write_predictions(predstruct, outpath, example_file)
% outpath should end with a backslash.
% predstruct.predset is a 5x1 cell
% OR you can supply predstruct.predmatrix which is Nval x 5
% predstruct.labelnames is a 5x1 cell
% predstruct.pred_filename is a Nval x 1 cell
outfile = [outpath 'predictions.csv']; % zip filename doesnt matter but the file inside must be named predictions.csv
%example_file = 'example_predictions_keeptheorder.csv';
ordered_labelnames = {'ValueExtraversion','ValueAgreeableness','ValueConscientiousness','ValueNeurotisicm','ValueOpenness'};

x = importdata(example_file);
ordered_filenames = x.textdata(2:end,1);
N = numel(ordered_filenames);

if isfield(predstruct,'predmatrix')
    predmatrix = predstruct.predmatrix;
else
predmatrix = zeros(N, numel(ordered_labelnames));
for li=1:numel(ordered_labelnames) % for each label
    whichlabelset = find(strcmp(ordered_labelnames{li}, predstruct.labelnames));
    %thislabels = predstruct.labelset{whichlabelset};
    thispreds = predstruct.predset{whichlabelset};
for i=1:N % for each test sample
    % try to find each filename in preds 
    thispred = thispreds(find(strcmp(ordered_filenames{i}, predstruct.pred_filename)));
    predmatrix(i,li) = thispred;
end
end
end

%% Preds collected in a matrix, write them to new csv file:
%% write to csv :
%fileID = fopen('C:\Users\pc\Dropbox\ChaLearn-Age-2016-FG\Predictions.csv','w');
fileID = fopen(outfile,'w');
% write first line:
fprintf(fileID,'VideoName,ValueExtraversion,ValueAgreeableness,ValueConscientiousness,ValueNeurotisicm,ValueOpenness\n');
for i=1:numel(ordered_filenames)
    if i<numel(ordered_filenames)
        fprintf(fileID,'%s,%f,%f,%f,%f,%f\n',ordered_filenames{i},predmatrix(i,1),predmatrix(i,2),predmatrix(i,3),predmatrix(i,4),predmatrix(i,5) );
    else
        fprintf(fileID,'%s,%f,%f,%f,%f,%f',ordered_filenames{i},predmatrix(i,1),predmatrix(i,2),predmatrix(i,3),predmatrix(i,4),predmatrix(i,5) );
    end
end
fclose(fileID);

%delete('C:\Users\pc\Dropbox\ChaLearn-Age-2016-FG\Predictions.zip');
%zip('C:\Users\pc\Dropbox\ChaLearn-Age-2016-FG\Predictions.zip', 'C:\Users\pc\Dropbox\ChaLearn-Age-2016-FG\Predictions.csv');
delete([outpath 'lapfi_preds.zip']);
zip([outpath 'lapfi_preds.zip'], [outpath 'predictions.csv']);

x2=42;

end

