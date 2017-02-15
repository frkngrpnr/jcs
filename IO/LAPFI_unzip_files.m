function [no_output] = LAPFI_unzip_files(inpath, outpath, bss)

zipfiles = dir([inpath bss '*.zip']);

if numel(zipfiles)==0
    fprintf('No *.zip files found in %s\n',inpath);
    return
end

fprintf('%s.m: unzipping files..\n',mfilename);
for i=1:numel(zipfiles)
    waitbar(i/numel(zipfiles));
    unzip([inpath bss zipfiles(i).name], outpath);
end

end

