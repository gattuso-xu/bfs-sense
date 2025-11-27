import numpy as np
import scipy.io as scio
from scipy.interpolate import interp1d

# 随机选择第一维度中的一个索引
#random_index = np.random.choice(matrix.shape[0])
#sub_matrix = matrix[random_index]


def generate_mask(length, num_ones):
    #生成一个长度为 length 的随机向量，其中包含 num_ones 个 1，其余为 0
    vector = np.zeros(length, dtype=int)
    ones_indices = np.random.choice(length, num_ones, replace=False)
    vector[ones_indices] = 1
    return vector

#insert函数，将 gene 中的某些样本插入到 A 的空隙中，根据 mask 来选择插入位置。
#返回:result -- 修改后的样本，形状为 (M+K, 4, 234)
def insert(gene, mask, A):
    # gene -- 形状为 (M, 4, 234) 的样本
    # A -- 形状为 (K, 4, 234) 的样本
    # mask -- 长度为 M 的 01 向量，指示哪些 gene 的样本需要插入

    # 1. 获取被选中的 gene 样本
    selected_gene = gene[mask == 1]  # 形状 (num_selected, 4, 234)
    num_selected = len(selected_gene)

    # 2. 构造 A 的索引，并在随机位置插入 gene 样本
    result = np.zeros((len(A) + num_selected, 4, 234))  # 初始化结果

    # 3. 生成插入位置（确保不重复）
    insert_positions = np.random.choice(
        len(A) + 1,  # 可插入的位置：0 ~ len(A)（包括开头和结尾）
        size=num_selected,
        replace=False  # 不重复插入同一位置
    )
    insert_positions = np.sort(insert_positions)  # 排序以便按顺序插入

    # 4. 合并 A 和 selected_gene
    a_idx = 0  # 当前 A 的索引
    g_idx = 0  # 当前 gene 的索引
    for i in range(len(result)):
        if g_idx < num_selected and i == insert_positions[g_idx]:
            # 插入 gene 样本
            result[i] = selected_gene[g_idx]
            g_idx += 1
        else:
            # 插入 A 样本
            result[i] = A[a_idx]
            a_idx += 1

    return result


# resample函数，均匀插值
def resample_bfi(BFI_data, timestamp, target_freq=40):
    '''
    将信号进行时间均匀的插值
    BFI_data: 维度是时间维度N*4*234的numpy数组
    timestamp: 一维N长度数组，表示每个数据点的时间戳
    target_freq: 目标插值频率，默认为40Hz
    return: 线性插值和样条插值后的BFI数据，维度为(M*4*234)，其中M是新的时间戳数量
    '''
    # 计算总的时间间隔
    t_total = timestamp[-1] - timestamp[0]

    BFI_data = np.array(BFI_data)
    # 计算新的时间戳数量
    num_new_points = max(int(t_total * target_freq) + 1, 112)
    #num_new_points = int(t_total * target_freq) + 1  # 加1是为了包含结束时间点

    # 创建新的时间戳数组
    Xq = np.linspace(timestamp[0], timestamp[-1], num_new_points)

    # 初始化存储插值结果的数组
    BFI_inserted = np.zeros((num_new_points, BFI_data.shape[1], BFI_data.shape[2]))
    # BFI_inserted_spline = np.zeros((num_new_points, BFI_data.shape[1], BFI_data.shape[2]))

    # 插值处理
    for i in range(BFI_data.shape[2]):  # 遍历234个通道
        for k in range(BFI_data.shape[1]):  # 遍历4个维度
            V = BFI_data[:, k, i]

            # 使用线性插值
            Vq = interp1d(timestamp, V, kind='linear')(Xq)

            # 使用样条插值（三次插值）
            # Vq_spline = interp1d(timestamp, V, kind='cubic')(Xq)

            BFI_inserted[:, k, i] = Vq
            # BFI_inserted_spline[:, k, i] = Vq_spline

    # BFI_inserted = BFI_inserted[:int(2.8*target_freq), :, :]
    # print(len(BFI_inserted))
    BFI_inserted = BFI_inserted[:112, :, :]
    # print(len(BFI_inserted[0][0]))
    return BFI_inserted


# 可视化结果
#visualize_samples_separately(A, B, modified_B, sample_index=0, time_index=0)