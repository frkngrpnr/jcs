function [ r ] = writepreds(csvfilename, preds,filenames,dimensions)
% write predictions to csv file
fileID=fopen(csvfilename,'w');
fprintf(fileID,'%s;%s;%s;%s;%s;%s;%s\r\n','filename',dimensions{1},dimensions{2},dimensions{3},dimensions{4},dimensions{5},dimensions{6});
for i=1:numel(filenames)
   fprintf(fileID,'%s;%f;%f;%f;%f;%f;%f\r\n',filenames{i},preds(i,1),preds(i,2),preds(i,3),preds(i,4),preds(i,5),preds(i,6)); 
    
end

fclose(fileID);
r=1;
disp(['Predictions are written to ' csvfilename]);
end

