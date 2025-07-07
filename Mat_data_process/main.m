clc
clear
% Texture image generation
D_Paths = {
    'F:\data\sec_data\E1\ch1';
    'F:\data\sec_data\E2\ch1';
    'F:\data\sec_data\E3\ch1';
    'F:\data\sec_data\E4\ch1';
    'F:\data\sec_data\E5\ch1';
    };
% you data path 
addpath '..\Texture\' 
%Data loading
class = 5;
name = 33;
count = 5;
line = 10;        % mm
scan_h = 0.08;  %mm
scan_v = {800;800;800;150;100;};%mm/s
nums = 21;
windows1 = floor([150000,150000,150000,800000,1200000]/nums);
sca = 31;
windows2 = {223 * sca + 224;223 * sca + 224;223 * sca + 224; floor(223 * sca * 80/15) + 224;223 * sca * 8 + 224};

cut = {sca;sca;sca;floor(sca*80/15);sca*8};
fs = 100000;
% Loading data
Time_path = '..\Time\';
isdircre(Time_path);
for p = 1:size(D_Paths,1)
    Path = D_Paths{p};
    cd(Path);
    split_str = strsplit(Path, '\');
    % path +  E*
    Time_root_path = strcat(Time_path,split_str{4},'\');
    isdircre(Time_root_path);

    File = dir(fullfile(Path,'*.mat'));            % Extract the file
    FileNames = {File.name}';                  % Extract the file names and convert them into a cell array with n rows and 1 column.
    num = size(FileNames,1);                   % The quantity of documents
    %% Generate image
    for f = 1:num
        name = FileNames{f};                 % Read the name of the i-th variable
        path_name = strcat(Path,'\',name);
        data = importdata(path_name);
        data = (data - min(data)) / (max(data)-min(data)) ;
        data = data * 255;
        utpr_root_1 = strcat(Time_root_path,'time','\'); isdircre(utpr_root_1);
        fs = 100000;
        
        win1=hanning(windows1(p));
        X=enframe(data,win1,windows1(p))';

        zhen = size(X,2);
        for z = 1:nums
        win1=rectwin(windows2{p});
        time=enframe(X(:,z),win1,windows2{p})';
        norm_time = (time - min(time)) / (max(time)-min(time));
        win2=rectwin(224);
        time=enframe(time,win2,cut{p});
        formattedStr = sprintf('%03d', z);
        save(strcat(utpr_root_1,name(1:6),"_",num2str(formattedStr),'.mat'),"time")
        
        end

    end

end
