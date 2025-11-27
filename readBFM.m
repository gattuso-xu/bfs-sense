clear all;close all;
%% read BFM data
% folderPath = 'C:\Windows\System32\BFM_data\'; 
%folderPath = 'D:\BFI activity\push1\'; 
folderPath = 'D:\BFI activity\activity0111\wave1\'; 
files = dir(folderPath);
files = files(~[files.isdir]); 
fileNames = {files.name};
fileNumbers = zeros(1, length(fileNames));
for i = 1:length(fileNames)
    fileName = fileNames{i};
    fileNumbers(i) = str2double(fileName(end-9:end-4));
end
[~, sortedIndices] = sort(fileNumbers);
concatenatedData = [];
concatenatedMAC =[];
timestamp = [];
for i = 1:length(sortedIndices)
    sortedFileName = fileNames{sortedIndices(i)};
    filePath = fullfile(folderPath, sortedFileName);
    disp(['Processing file: ', filePath]);
    data = load(filePath);
    
    if isempty(concatenatedData)
        concatenatedData = data.packet_infos; 
        concatenatedMAC = data.ether_srcs;
        timestamp = str2num(data.packet_timestamps);
    else
        time=str2num(data.packet_timestamps);
        concatenatedData = cat(1, concatenatedData, data.packet_infos);
        concatenatedMAC =cat(1,concatenatedMAC,data.ether_srcs);
        timestamp = cat(1,timestamp,time);
    end
end
 
mac_addr_list=unique(concatenatedMAC,'row');
mac_addr = cellstr(concatenatedMAC);
mac_datasize = [];
for k = 1:size(mac_addr_list,1)
    idx=find(strcmp(mac_addr, mac_addr_list(k,:)));
    mac_datasize = [mac_datasize, length(idx)];
end
[~,max_id] = max(mac_datasize);
idx = find(strcmp(mac_addr, mac_addr_list(max_id,:)));
concatenatedData = concatenatedData(idx,:,:,:);
timestamp = timestamp(idx);


% 
% data = abs(squeeze(concatenatedData(1,:,:,:)));
% figure;imagesc([1 size(data,2)],[1 size(data,1)],data)
% 
% data1 = abs(struct2array(load('D:\BFI activity\conV_matrices.mat')));
% figure;imagesc([1 size(data1,2)],[1 size(data1,1)],data1)



%% BFM  ratio
Ant_idx = [2,1]; % Two elements for BFM ratio
BFM_ratio = squeeze(concatenatedData(:,Ant_idx(1),1,:)./concatenatedData(:,Ant_idx(2),1,:));
figure
subplot(211)
% plot(timestamp,abs(BFM_ratio))
plot(abs(BFM_ratio))
% xlabel('time')
ylabel('amplitude of BFM ratio')
subplot(212)
% plot(timestamp,unwrap(angle(BFM_ratio)))
plot(unwrap(angle(BFM_ratio)))
% xlabel('time')
ylabel('phase of BFM ratio')

c1 = abs(squeeze(concatenatedData(:,1,1,:)));
figure;imagesc([1 size(c1,1)],[1 size(c1,2)],c1')

c2 = abs(BFM_ratio);
figure;imagesc([1 size(c2,1)],[1 size(c2,2)],c2')


%%数据切分
% indices=[
%     21, 45; 
%     70,82;
%     86,96;
%     
%     ];
%  wave1

indices=[
    21, 45; 
    70,82;
    86,96;
    
    ];

sliced_arrays = slice_multiple(concatenatedData, indices);


% size(sliced_arrays{1})


concatenatedData1 = squeeze(concatenatedData);



samples = split_signal2(concatenatedData);





function sliced_arrays = slice_multiple(x, indices)
    % x: 输入的向量
    % indices: 一个n×2的矩阵，n表示切片的数量，每行的第一个元素为起始索引，第二个元素为结束索引
    
    n = size(indices, 1); % 获取切片数量
    sliced_arrays = cell(n, 1); % 使用元胞数组保存多个切片
    
    for i = 1:n
        start_index = indices(i, 1);
        end_index = indices(i, 2);
        
        % 检查输入是否有效
        if start_index < 1 || end_index > length(x) || start_index > end_index
            error('Invalid indices for slice %d. Ensure 1 <= start_index <= end_index <= length(x).', i);
        end
        
        % 切片操作
        sliced_arrays{i} = x(start_index:end_index,:,:,:);
    end
end

function samples = split_signal(input_signal, sample_length)
    % 输入信号为 input_signal
    % 每个样本的长度为 sample_length（这里是15）
    % 计算可以切分的样本数量,sit

    num_samples = floor(size(input_signal,1) / sample_length);
    
    
    % 切分信号并填充到样本矩阵中,输出格式长度*4*234
    for i = 1:num_samples
        start_index = (i-1) * sample_length + 1;
        end_index = i * sample_length;
        samples(i,:,:,:) = input_signal(start_index:end_index,:,:);
    end
end



function sub_samples = split_signal2(signal)
    % 输入参数 signal: N*4*1*234 的信号
    % 输出参数 sub_samples: 一个 cell 数组，每个 cell 中保存一个 M*4*1*234 的子样本
    
    % 获取输入信号的维度
    [N, dim1, dim2, dim3] = size(signal);
    
    % 检查输入信号的维度是否符合预期
    if dim1 ~= 4 || dim2 ~= 1 || dim3 ~= 234
        error('输入信号的维度不符合预期。应为 N*4*1*234');
    end
    
    % 初始化存储子样本的 cell 数组
    sub_samples = {};
    
    % 设置随机切分的子样本数量，假设我们切分 10 次
    num_splits = 60;  % 可以根据需要调整
    
    for i = 1:num_splits
        % 随机生成一个子样本的长度，范围在 18 到 26 之间
        sample_length = randi([18, 26]);
        
        % 随机选择切分的起始位置
        start_idx = randi([1, N - sample_length + 1]);
        
        % 获取该切分位置的子样本
        sub_sample = signal(start_idx:start_idx + sample_length - 1, :, :, :);
        
        % 将子样本添加到 cell 数组中
        sub_samples{end+1} = sub_sample;
    end
end






