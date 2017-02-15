function [r1]=prepare_submission(scores_elm,dimensions,test_filenames)
if nargin<3
   load('test_filenames')
end
if nargin<2
    load('anotations_train')
    dimensions=cell(1,1);
    for i=1:6
        dimensions{i}=anotations_train{1, i}.dimension;
    end
end
writepreds('predictions.csv',scores_elm,test_filenames,dimensions);
command='python readPickles.py > preds_all.txt'; % works fine in Linux (Ubuntu)
r1=system(command);
if (r1==0)
    disp('Predictions are written to predictions.pkl file');
end

end