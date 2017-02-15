function [fset] = LAPFI_encode_frame_features(inpath, encoding_name)
% 3rd argument is optional, but recommended to get all the labels at once
% inpath folder will have .mat files with processed (frame feat. ext.) videos in them
%% Input validation
extra_params={}; if nargin<3, feature_params = []; end
bss = '\';
%feature_name = upper(feature_name);
encoding_name = upper(encoding_name);

files = dir([inpath bss '*.mat']);
N = numel(files); %N = 100
fset = struct;
%fset.meta.feature_options = options; % will be done in the loop
fset.label = zeros(N,1);
fset.filter = ones(N,1);
fset.filename = cell(N, 1);
for i=1:N
    waitbar(i/N);
    %% Load video
    load([inpath bss files(i).name]);
        %PV.data(:,1:4000)=[]; % !!! temporary
    filename_withoutmat = strrep(files(i).name, '.mat', '');
    fset.filename{i,:} = filename_withoutmat;
    %close all, plot_video(V); pause
    tv=tic();
    %% Construct 3D video:
    %PV.data = PV.data(1,:); % TEMPORARY!
    F = size(PV.data, 1); % number of frames in this video
    %if F<3
     %   desc = zeros(1, size(fset.data,2));
      %  fset.filter(i)=0;
    %else
        if strcmp(encoding_name(1:3),'FUN')
            [desc, funids, featids] = data_to_fun(PV.data');
        elseif strcmp(encoding_name(1:4),'MEAN')
            desc = mean(PV.data,1);
            funids = ones(numel(desc));
            featids = 1:numel(desc);
        end
    %end
    % save frame level features, i.e. the struct PV:
%    save([outpath bss files(i).name], 'PV');
%     end % if enough frames
    %% Extract and save features
    if i==1
        fset.meta.funids = funids;
        fest.meta.featids = featids;
        fset.meta.feature_options = PV.meta.feature_options;
        fset.meta.feature_options.encoding_name = encoding_name;
        fset.data = zeros(N, numel(desc)); 
        fprintf('feature size = %d for %s[%s]-%s , will process %d videos\n',numel(desc),fset.meta.feature_options.feature_name, ...
            mat2str(fset.meta.feature_options.feature_params),...
            encoding_name, N);
    end
       %fprintf('i=%d, desc size = %s \n',i,mat2str(size(desc)));
       % bad solution:
       if numel(desc)==0
           desc = mean(fset.data(1:i-1,:));
       end
    fset.data(i,:) = desc;
    if mod(i,500)==1, fprintf('video %d/%d processed in %f seconds.\n',i,N,toc(tv)); end
end

end % main function

