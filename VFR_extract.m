%this script is to extract VFR features directly and save into each txt
%file

%% read all files and save file names in cell
fid=fopen('/media/kingformatty/king/csrc_spapl/data/SetC_train/wav.scp');
count=1;
HEADER_MAT={};
FEATURE_MAT=[];
filenames={};
while ~feof(fid)
    tline = fgets(fid);
    filenames{count} = tline;
    count=count+1;
    
end

fclose(fid);
framestart=1;
for i = 1:length(filenames)
filename_split=split(filenames{i});
path=filename_split{2};
[seg,Fs]=audioread(path);
len=length(seg);
vad=ones(1,len);
feat=VFR(seg,Fs,2.5,vad);%winshift is determined in the function using entropy
frameend=framestart+size(feat,1)-1;
HEADER_MAT{i,1}=filename_split{1};
HEADER_MAT{i,2}=size(feat,1);
HEADER_MAT{i,3}=size(feat,2);
HEADER_MAT{i,4}=framestart;
HEADER_MAT{i,5}=frameend;


`FEATURE_MAT(framestart:frameend,:)=feat;
framestart=frameend+1;
fprintf('Processing %s\n',[filename_split{1} '0001']);
end

%disp('Writing ark files')
%arkwrite('/media/kingformatty/king/VFR_feat/SetC_train/feats.ark',HEADER_MAT,FEATURE_MAT);
%ark2scp('/media/kingformatty/king/VFR_feat/SetC_train/feats.ark');
