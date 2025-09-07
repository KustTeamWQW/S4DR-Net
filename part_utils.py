import numpy as np
from tqdm import tqdm

def write_plyfile(file_name, point_cloud):
    f = open(file_name + '.ply', 'w')
    init_str = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(point_cloud)) + \
               "\nproperty float x\nproperty float y\nproperty float z\n" \
               "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(init_str)
    for i in range(len(point_cloud)):
        f.write(str(round(float(point_cloud[i][0]), 6)) + ' ' + str(round(float(point_cloud[i][1]), 6)) + ' ' +
                str(round(float(point_cloud[i][2]), 6)) + '\n')
    f.close()


def get_bbox(point_cloud):#"""获取点云的边界框信息
    """Check if two bbox are intersected
    Args:
        point_cloud (_type_): (N, C)
    """
    point_cloud = np.array(point_cloud).T  # C,N
    x_diff = max(point_cloud[0]) - min(point_cloud[0])
    y_diff = max(point_cloud[1]) - min(point_cloud[1])
    z_diff = max(point_cloud[2]) - min(point_cloud[2])
    return min(point_cloud[0]), min(point_cloud[1]), min(point_cloud[2]), x_diff, y_diff, z_diff


def is_connected(point_cloud_1, point_cloud_2, theta=0.2):
    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return False
    x_start_1, y_start_1, z_start_1, x_diff_1, y_diff_1, z_diff_1 = get_bbox(
        point_cloud_1)
    x_start_2, y_start_2, z_start_2, x_diff_2, y_diff_2, z_diff_2 = get_bbox(
        point_cloud_2)

    # scale bbox
    x_end_1 = x_start_1 + x_diff_1 + x_diff_1*theta
    y_end_1 = y_start_1 + y_diff_1 + y_diff_1*theta
    z_end_1 = z_start_1 + z_diff_1 + z_diff_1*theta
    x_end_2 = x_start_2 + x_diff_2 + x_diff_2*theta
    y_end_2 = y_start_2 + y_diff_2 + y_diff_2*theta
    z_end_2 = z_start_2 + z_diff_2 + z_diff_2*theta

    x_start_1 -= x_diff_1*theta
    y_start_1 -= y_diff_1*theta
    z_start_1 -= z_diff_1*theta
    x_start_2 -= x_diff_2*theta
    y_start_2 -= y_diff_2*theta
    z_start_2 -= y_diff_2*theta

    # check if two bbox are intersected
    if (x_start_1 <= x_start_2 <= x_end_1 or x_start_2 <= x_start_1 <= x_end_2) and (y_start_1 <= y_start_2 <= y_end_1 or y_start_2 <= y_start_1 <= y_end_2) and (z_start_1 <= z_start_2 <= z_end_1 or z_start_2 <= z_start_1 <= z_end_2):
        return True
    else:
        return False


def histogram_intersection(hist1, hist2):
    """计算两个直方图的交集
    Args:
        hist1 (np.ndarray): 直方图1
        hist2 (np.ndarray): 直方图2
    Returns:
        float: 直方图交集值
    """
    return np.sum(np.minimum(hist1, hist2))


def is_connected_histogram(hist1, hist2, threshold=0.3):
    """通过颜色直方图交集判断两个体素是否相连
    Args:
        hist1 (np.ndarray): 体素1的颜色直方图
        hist2 (np.ndarray): 体素2的颜色直方图
        threshold (float): 交集阈值
    Returns:
        bool: 是否连通
    """
    intersection = histogram_intersection(hist1, hist2)
    return intersection >= threshold


def compute_color_histograms(data, p2v_indices, split_num=3, bins=8):
    """计算每个体素的颜色直方图
    Args:
        data (np.ndarray): 点云数据，形状为 (B, N, C)
        p2v_indices (np.ndarray): 体素索引，形状为 (B, N)
        split_num (int): 每个轴的分割数
        bins (int): 颜色直方图的桶数
    Returns:
        np.ndarray: 颜色直方图，形状为 (B, split_num**3, bins*3)
    """
    B, N, C = data.shape
    color_histograms = np.zeros((B, split_num ** 3, bins * 3), dtype=np.float32)

    # 假设颜色信息在 data 的后3个通道（RGB）
    for b in tqdm(range(B), desc="计算颜色直方图"):
        for v in range(split_num ** 3):
            voxel_points = data[b][p2v_indices[b] == v]
            if voxel_points.shape[0] == 0:
                continue
            # 提取RGB颜色
            colors = voxel_points[:, 3:6]
            # 计算每个颜色通道的直方图
            hist_r, _ = np.histogram(colors[:, 0], bins=bins, range=(0, 1), density=True)
            hist_g, _ = np.histogram(colors[:, 1], bins=bins, range=(0, 1), density=True)
            hist_b, _ = np.histogram(colors[:, 2], bins=bins, range=(0, 1), density=True)
            # 合并直方图
            color_histograms[b, v] = np.concatenate([hist_r, hist_g, hist_b])

    return color_histograms


def split_part(data, split_num=3, max_distance=3):
    # data: (B,N,C)
    B, N, C = data.shape
    print('Split point cloud to part...')
    max_p = np.max(data, axis=1)[:, np.newaxis, :]  # (B,N,C)->(B,1,C)
    min_p = np.min(data, axis=1)[:, np.newaxis, :]
    diff = max_p - min_p  # (B,C)

    diff /= split_num
    diff[diff == 0] = 1e-5
    p2v_indices = ((data - min_p) / diff).astype(int)  # (B,N,C)
    p2v_indices[p2v_indices == split_num] = split_num - 1
    p2v_indices[p2v_indices > split_num] = 1  #边缘情况的处理
    # voxel index : (B,N,C) -> (B,N)
    p2v_indices = p2v_indices[:, :, 2] + split_num * p2v_indices[:, :,
                                                         1] + split_num**2 * p2v_indices[:, :, 0] #一个 (B, N) 的数组，表示每个点所属的分块ID（0到26，对于3x3x3的分块）。

    print('Get part hop distance...')
    print('--1 Get adjacency matrix...') #初始化 (B, 27, 27) 的邻接矩阵 adjacency，0表示不可连通，1表示两个分块之间连通。体现每一块的连通性
    adjacency = np.zeros((B, split_num**3, split_num ** 3)).astype(int)  #一个邻接矩阵 adjacency，形状为 (B, 27, 27) 表示第 b 个样本的第 i 个分块与第 j 个分块是否相邻（可连通）

    for b, pc in enumerate(tqdm(data)):
        for i in range(split_num**3):
            for j in range(i): 
                pc_i = pc[p2v_indices[b] == i]
                pc_j = pc[p2v_indices[b] == j]
                if is_connected(pc_i, pc_j):
                    adjacency[b, i, j] = 1
                    adjacency[b, j, i] = 1

    
    print('--2 Get hop distance...')
    part_distance = np.empty((B, split_num**3, split_num ** 3)) #分块间最短路径距离 (Hop Distance)
    part_distance.fill(1e3)
    part_distance[adjacency == 1] = 1
    batch_index = np.arange(B).reshape((B, 1)).repeat(split_num ** 3, axis=1)
    part_distance[batch_index, np.arange(
        split_num ** 3), np.arange(split_num ** 3)] = 0  # set self distance=0
    part_distance = part_distance.astype(int)

    for b, pc in enumerate(tqdm(data)):
        # Floyd–Warshall algorithm: O(V^3)
        for k in range(split_num ** 3):
            for i in range(split_num ** 3):
                for j in range(split_num ** 3):
                    if part_distance[b, i, j] > part_distance[b, i, k] + part_distance[b, k, j]:
                        part_distance[b, i, j] = part_distance[b,
                                                               i, k] + part_distance[b, k, j]
    part_distance[part_distance > max_distance] = max_distance
#------------------------------------光谱特征距离矩阵计算-----------------------------------------------------
    color_histograms = compute_color_histograms(data, p2v_indices, split_num=split_num, bins=8) #计算光谱最短路径

    print('Get adjacency matrix based on color histogram intersection...')
    adjacency_sp = np.zeros((B, split_num ** 3, split_num ** 3), dtype=int)
    for b in tqdm(range(B), desc="构建邻接矩阵"):
        for i in range(split_num ** 3):
            for j in range(i):
                hist_i = color_histograms[b, i]
                hist_j = color_histograms[b, j]
                if is_connected_histogram(hist_i, hist_j, threshold=0.5): #闸值，超参数
                    adjacency_sp[b, i, j] = 1
                    adjacency_sp[b, j, i] = 1

    print('--2 Get hop_sp distance...')
    part_distance_sp = np.empty((B, split_num ** 3, split_num ** 3))
    part_distance_sp.fill(1e3)
    part_distance_sp[adjacency == 1] = 1
    for b in range(B):
        part_distance_sp[b, np.arange(split_num ** 3), np.arange(split_num ** 3)] = 0  # set self distance=0
    part_distance_sp = part_distance_sp.astype(int)

    print('Compute shortest path distances ...')
    for b in tqdm(range(B), desc="计算最短路径距离"):
        for k in range(split_num ** 3):
            for i in range(split_num ** 3):
                for j in range(split_num ** 3):
                    if part_distance_sp[b, i, j] > part_distance_sp[b, i, k] + part_distance_sp[b, k, j]:
                        part_distance_sp[b, i, j] = part_distance_sp[b, i, k] + part_distance_sp[b, k, j]
    part_distance_sp[part_distance_sp > max_distance] = max_distance

    return p2v_indices, part_distance,part_distance_sp #(B, N) 每个点对应的分块ID 、 (B, 27, 27) 每个样本的分块间最短距离矩阵



