import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from util import *
from part_utils import *
import json
import cv2
import requests
import zipfile
import shutil

#检查并下载S3DIS数据集的HDF5格式文件（indoor3d_sem_seg_hdf5_data），以及校验Stanford3dDataset_v1.2_Aligned_Version数据集是否已存在
def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)  # 确保数据目录存在

    # 下载并解压 indoor3d_sem_seg_hdf5_data
    indoor_data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    indoor_zip = 'indoor3d_sem_seg_hdf5_data.zip'
    indoor_zip_path = os.path.join(DATA_DIR, indoor_zip)

    if not os.path.exists(indoor_data_dir):
        url = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        try:
            print(f'Downloading {url}...')
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 检查请求是否成功
            with open(indoor_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print('Download complete.')

            print('Extracting zip file...')
            with zipfile.ZipFile(indoor_zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print('Extraction complete.')

            os.remove(indoor_zip_path)
            print('Removed zip file.')
        except requests.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
            print('Please download indoor3d_sem_seg_hdf5_data.zip manually and place it under the data/ directory.')
            return  # 或者 raise e
        except zipfile.BadZipFile as e:
            print(f'Error extracting {indoor_zip}: {e}')
            raise

    else:
        print(f'{indoor_data_dir} already exists. Skipping download.')

    # 检查并处理 Stanford3dDataset_v1.2_Aligned_Version
    stanford_data_dir = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')
    stanford_zip = 'Stanford3dDataset_v1.2_Aligned_Version.zip'
    stanford_zip_path = os.path.join(DATA_DIR, stanford_zip)

    if not os.path.exists(stanford_data_dir):
        print(f'{stanford_data_dir} does not exist.')
        if not os.path.exists(stanford_zip_path):
            print(
                'Please download Stanford3dDataset_v1.2_Aligned_Version.zip from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            exit(0)
        else:
            try:
                print(f'Extracting {stanford_zip}...')
                with zipfile.ZipFile(stanford_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print('Extraction complete.')

                # 确保目录移动正确
                extracted_dir = 'Stanford3dDataset_v1.2_Aligned_Version'
                extracted_path = os.path.join(DATA_DIR, extracted_dir)
                if os.path.exists(extracted_path):
                    print(f'{extracted_dir} already extracted.')
                else:
                    shutil.move(os.path.join(DATA_DIR, extracted_dir), stanford_data_dir)
                    print(f'Moved {extracted_dir} to {stanford_data_dir}.')

                os.remove(stanford_zip_path)
                print('Removed zip file.')
            except zipfile.BadZipFile as e:
                print(f'Error extracting {stanford_zip}: {e}')
                raise
    else:
        print(f'{stanford_data_dir} already exists. Skipping download and extraction.')###


def prepare_test_data_semseg(): # 在数据目录下检查stanford_indoor3d与indoor3d_sem_seg_hdf5_data_test数据是否存在，
    # 如不存在则通过调用外部脚本（collect_indoor3d_data.py和gen_indoor3d_h5.py）对原始数据进行预处理与生成测试数据集的HDF5文件
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, '_indoor3d_HG')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data_test_')): #//
        os.system('python prepare_data/gen_indoor3d_h5.py')


def prepare_test_data_semseg_test(): # 在数据目录下检查stanford_indoor3d与indoor3d_sem_seg_hdf5_data_test数据是否存在，
    # 如不存在则通过调用外部脚本（collect_indoor3d_data.py和gen_indoor3d_h5.py）对原始数据进行预处理与生成测试数据集的HDF5文件
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, '_indoor3d_HG')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data_test_')): #//
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    prepare_test_data_semseg() #确保数据已准备就绪
    if partition == 'train': #据partition选择对应的数据目录
        data_dir = os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data') #//
    else:
        data_dir = os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data_test') #//
    with open(os.path.join(data_dir, "all_files.txt")) as f:#从all_files.txt和room_filelist.txt中读取所有HDF5文件列表以及房间名称列表
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files: #加载并合并 HDF5 文件中的数据
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)

    data_batches = np.concatenate(data_batchlist, 0) ##?
    seg_batches = np.concatenate(label_batchlist, 0) #依次读取所有HDF5文件中的点云数据(data)与标签(label)并将其合并(concatenate)成大数组。
    print(test_area)
    test_area_name = "Area_" + str(test_area)   #???
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:#根据test_area（如test_area=1时对应Area_1）筛选数据集
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train': #根据partition决定返回训练集或测试集对应的点云数据(all_data)与标签(all_seg)
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg

def load_data_semseg_DX(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    prepare_test_data_semseg() #确保数据已准备就绪
    if partition == 'train': #据partition选择对应的数据目录
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data') #//
    else:
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data_test') #//
    with open(os.path.join(data_dir, "all_files.txt")) as f:#从all_files.txt和room_filelist.txt中读取所有HDF5文件列表以及房间名称列表
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files: #加载并合并 HDF5 文件中的数据
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)

    data_batches = np.concatenate(data_batchlist, 0) ##?
    seg_batches = np.concatenate(label_batchlist, 0) #依次读取所有HDF5文件中的点云数据(data)与标签(label)并将其合并(concatenate)成大数组。
    print(test_area)
    test_area_name = "Area_" + str(test_area)   #???
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:#根据test_area（如test_area=1时对应Area_1）筛选数据集
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train': #根据partition决定返回训练集或测试集对应的点云数据(all_data)与标签(all_seg)
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg

def load_data_semseg_Umamba(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    prepare_test_data_semseg() #确保数据已准备就绪
    if partition == 'train': #据partition选择对应的数据目录
        data_dir = os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data') #//
    else:
        data_dir = os.path.join(DATA_DIR, 'outdoorHG_sem_seg_hdf5_data_test') #//
    with open(os.path.join(data_dir, "all_files.txt")) as f:#从all_files.txt和room_filelist.txt中读取所有HDF5文件列表以及房间名称列表
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files: #加载并合并 HDF5 文件中的数据
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)

    data_batches = np.concatenate(data_batchlist, 0) ##?
    seg_batches = np.concatenate(label_batchlist, 0) #依次读取所有HDF5文件中的点云数据(data)与标签(label)并将其合并(concatenate)成大数组。
    print(test_area)
    test_area_name = "Area_" + str(test_area)   #???
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:#根据test_area（如test_area=1时对应Area_1）筛选数据集
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train': #根据partition决定返回训练集或测试集对应的点云数据(all_data)与标签(all_seg)
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_data_semseg_Umamba_test(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data')  # 训练集路径
    else:
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data_test_')  # 测试集路径

    # 文件列表
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.strip() for line in f]

    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.strip() for line in f]

    # 分离 raw 和 norm 文件
    raw_files = [f for f in all_files if 'raw' in f]
    norm_files = [f for f in all_files if 'norm' in f]

    assert len(raw_files) == len(norm_files), "Raw/Norm 文件数量不一致，请检查 all_files.txt 内容"

    data_raw_list, data_norm_list, label_list = [], [], []

    for raw_f, norm_f in zip(raw_files, norm_files):
        raw_path = os.path.join(DATA_DIR, raw_f)
        norm_path = os.path.join(DATA_DIR, norm_f)

        with h5py.File(raw_path, 'r') as f_raw, h5py.File(norm_path, 'r') as f_norm:
            data_raw = f_raw['data'][:]
            label = f_raw['label'][:]  # label 是相同的
            data_norm = f_norm['data'][:]

        data_raw_list.append(data_raw)
        data_norm_list.append(data_norm)
        label_list.append(label)

    # 合并所有批次
    all_raw = np.concatenate(data_raw_list, axis=0)
    all_norm = np.concatenate(data_norm_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)

    test_area_name = "Area_" + str(test_area)
    train_idxs, test_idxs = [], []

    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    if partition == 'train':
        return all_raw[train_idxs], all_norm[train_idxs], all_label[train_idxs]
    else:
        return all_raw[test_idxs], all_norm[test_idxs], all_label[test_idxs]


def load_data_semseg_Umamba_DX(partition, test_area):  ### 重点修改 ， 获取原始点云的坐标
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    prepare_test_data_semseg() #确保数据已准备就绪
    if partition == 'train': #据partition选择对应的数据目录
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data') #//
    else:
        data_dir = os.path.join(DATA_DIR, 'outdoorDX_sem_seg_hdf5_data_test') #//
    with open(os.path.join(data_dir, "all_files.txt")) as f:#从all_files.txt和room_filelist.txt中读取所有HDF5文件列表以及房间名称列表
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files: #加载并合并 HDF5 文件中的数据
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)

    data_batches = np.concatenate(data_batchlist, 0) ##?
    seg_batches = np.concatenate(label_batchlist, 0) #依次读取所有HDF5文件中的点云数据(data)与标签(label)并将其合并(concatenate)成大数组。
    print(test_area)
    test_area_name = "Area_" + str(test_area)   #???
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:#根据test_area（如test_area=1时对应Area_1）筛选数据集
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train': #根据partition决定返回训练集或测试集对应的点云数据(all_data)与标签(all_seg)
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg




def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


import numpy as np
import torch
from torch.utils.data import Dataset

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def voxelization(points, voxel_size):
    """
    Perform voxelization on a given point cloud.

    Parameters:
    points (numpy.ndarray): Nx3 array of points (x, y, z).
    voxel_size (float): Size of the voxel grid.

    Returns:
    numpy.ndarray: Nx3 array of voxelized coordinates.
    """
    # Calculate the voxel indices
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)  # [1,4096,3]

    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

    bounding_box = coord_max - coord_min  # 计算包围盒的尺寸

    voxel_total = np.ceil(bounding_box[0] * bounding_box[1] * bounding_box[2] / voxel_size ** 3).astype(
        np.int32)  # 25*25*25  基于体积计算体素数量 [1]
    voxel_valid = np.unique(voxel_indices, axis=0)  # 去除重复的体素索引，得到所有被点占用的体素（有效体素）[N_vaild , 3]

    return points, voxel_indices, voxel_total, voxel_valid


def fps_series_func(points, voxel_indices, samplepoints_list):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().cuda().unsqueeze(0)
    voxel_indices = torch.Tensor(voxel_indices).float().cuda().unsqueeze(0)
    fps_index_list = []
    series_idx_lists = []

    x1y1z1 = [1, 1, 1]
    x0y1z1 = [-1, 1, 1]
    x1y0z1 = [1, -1, 1]
    x0y0z1 = [-1, -1, 1]
    x1y1z0 = [1, 1, -1]
    x0y1z0 = [-1, 1, -1]
    x1y0z0 = [1, -1, -1]
    x0y0z0 = [-1, -1, -1]

    series_list = []
    # series_list.append(x1y1z1)
    # series_list.append(x0y1z1)
    # series_list.append(x1y0z1)
    series_list.append(x0y0z1)
    series_list.append(x1y1z0)
    # series_list.append(x0y1z0)
    # series_list.append(x1y0z0)
    # series_list.append(x0y0z0)

    for i in range(len(samplepoints_list)):  # 遍历 samplepoints_list 中每个采样数 S（例如 512、128、32），即对点云进行多层次采样。
        S = samplepoints_list[i]
        xyz = points[:, :, :3]  # 提取前3个通道

        fps_index = farthest_point_sample(xyz, S)  # 最远点采样S个点，并且返回索引

        points = index_points(points, fps_index)  # [1,4096,f]->[1,s,f]
        new_voxel_indices = index_points(voxel_indices, fps_index).squeeze(0).cpu().data.numpy()  # [1,4096,1]->[1,s,1]
        voxel_indices = index_points(voxel_indices, fps_index)

        fps_index = fps_index.cpu().data.numpy()
        padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(padded_fps_index)

        series_idx_list = []
        for j in range(len(series_list)):  # 遍历包含不同体素方向的列表

            series = series_list[j]
            new_voxel_indices_ForSeries = new_voxel_indices * series
            sorting_indices = np.expand_dims(np.lexsort((new_voxel_indices_ForSeries[:, 0],
                                                         new_voxel_indices_ForSeries[:, 1],
                                                         new_voxel_indices_ForSeries[:, 2])), axis=0)
            padded_sorting_indices = np.expand_dims(
                np.pad(sorting_indices, ((0, 0), (0, pad_width - sorting_indices.shape[1])), mode='constant'), axis=0)
            series_idx_list.append(padded_sorting_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1)  # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)

    series_idx_arrays = np.concatenate(series_idx_lists, axis=0)  # 3 8 N
    fps_index_array = np.vstack(fps_index_list)  # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1', split_num=3, small_classes=[1, 2, 5]):
        self.data, self.seg = load_data_semseg(partition, test_area)  # 返回numpy数组[M,N,9]---data
        self.num_points = num_points
        self.partition = partition
    
        self.split_num = split_num
        self.max_distance = self.split_num + 1  # 0,1,2,3

        self.small_classes = small_classes  # 小类别标签

        self.debug = False

        load_partdata = True


        abs_path = os.path.abspath(
            'scene_seg/data/outdoor_HG_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(
                partition, self.split_num, self.max_distance, test_area))
        print(os.path.exists(abs_path))
        print(abs_path)

        if load_partdata and os.path.exists(
                'scene_seg/data/outdoor_HG_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                               self.max_distance, test_area)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'scene_seg/data/outdoor_HG_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                               self.max_distance, test_area))
            self.part_distance = np.load(
                'scene_seg/data/outdoor_HG_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                 self.max_distance, test_area))
            self.part_distance_sp = np.load(
                'scene_seg/data/outdoor_HG_part_distance_sp_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                    self.max_distance, test_area))
        else:
            self.p2v_indices, self.part_distance, self.part_distance_sp = split_part(  # 点到块的索引   以及 块间距离矩阵的构建
                self.data, self.split_num, self.max_distance)  # p2v_indices: (B,N). part_distance: (B, 27, 27)
            if not self.debug:
                np.save('scene_seg/data/outdoor_HG_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                       self.max_distance, test_area),
                        self.p2v_indices)
                np.save('scene_seg/data/outdoor_HG_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                         self.max_distance, test_area),
                        self.part_distance)
                np.save('scene_seg/data/outdoor_HG_part_distance_sp_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                            self.max_distance,
                                                                                            test_area),
                        self.part_distance_sp)
            print('Split part done!')

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        part_distance = self.part_distance[item]  
        part_distance_sp = self.part_distance_sp[item]
        p2v_indices = self.p2v_indices[item] 

        if self.partition == 'train':
            pc_rand_idx = np.arange(self.num_points)
            np.random.shuffle(pc_rand_idx)
            pointcloud = pointcloud[pc_rand_idx, :]
            p2v_indices = p2v_indices[pc_rand_idx]
            seg = seg[pc_rand_idx]

            # shuffle parts along N dim
            part_rand_idx = np.arange(self.split_num ** 3)
            np.random.shuffle(part_rand_idx)
            part_distance = part_distance[part_rand_idx][:, part_rand_idx]
            part_distance_sp = part_distance_sp[part_rand_idx][:, part_rand_idx]
        else:
            part_rand_idx = np.arange(self.split_num ** 3)

        seg = torch.LongTensor(seg)

        return (pointcloud, seg, p2v_indices, part_distance, part_distance_sp, part_rand_idx)

    def __len__(self):
        return self.data.shape[0]


class S3DIS_DX(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1', split_num=3, small_classes=[1, 2, 5]):
        self.row, self.data, self.seg = load_data_semseg_Umamba_test(partition, test_area)  # 返回numpy数组[M,N,9]---data
        self.num_points = num_points
        self.partition = partition
        self.semseg_colors = load_color_semseg()

        self.split_num = split_num
        self.max_distance = self.split_num + 1  # 0,1,2,3

        self.small_classes = small_classes  # 小类别标签

        self.debug = False

        load_partdata = True
        if load_partdata and os.path.exists(
                'data/outdoor_DXT_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                               self.max_distance, test_area)):
            print('load pre part data')
            self.p2v_indices = np.load(
                'data/outdoor_DXT_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                               self.max_distance, test_area))
            self.part_distance = np.load(
                'data/outdoor_DXT_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                 self.max_distance, test_area))
            self.part_distance_sp = np.load(
                'data/outdoor_DXT_part_distance_sp_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                    self.max_distance, test_area))
        else:
            self.p2v_indices, self.part_distance, self.part_distance_sp = split_part(  # 点到块的索引   以及 块间距离矩阵的构建
                self.data, self.split_num, self.max_distance)  # p2v_indices: (B,N). part_distance: (B, 27, 27)
            if not self.debug:
                np.save('data/outdoor_DXT_p2v_indices_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                       self.max_distance, test_area),
                        self.p2v_indices)
                np.save('data/outdoor_DXT_part_distance_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                         self.max_distance, test_area),
                        self.part_distance)
                np.save('data/outdoor_DXT_part_distance_sp_{}_splitnum_{}_md{}_ta{}.npy'.format(partition, self.split_num,
                                                                                            self.max_distance,
                                                                                            test_area),
                        self.part_distance_sp)
            print('Split part done!')

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        row = self.row[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        part_distance = self.part_distance[item]  # (1, 27, 27)
        part_distance_sp = self.part_distance_sp[item]
        p2v_indices = self.p2v_indices[item]  # (1, N)

        if self.partition == 'train':
            pc_rand_idx = np.arange(self.num_points)
            np.random.shuffle(pc_rand_idx)
            pointcloud = pointcloud[pc_rand_idx, :]
            p2v_indices = p2v_indices[pc_rand_idx]
            seg = seg[pc_rand_idx]

            # shuffle parts along N dim
            part_rand_idx = np.arange(self.split_num ** 3)
            np.random.shuffle(part_rand_idx)
            part_distance = part_distance[part_rand_idx][:, part_rand_idx]
            part_distance_sp = part_distance_sp[part_rand_idx][:, part_rand_idx]
        else:
            part_rand_idx = np.arange(self.split_num ** 3)

        seg = torch.LongTensor(seg)

        return (row,pointcloud, seg, p2v_indices, part_distance, part_distance_sp, part_rand_idx)

    def __len__(self):
        return self.data.shape[0]

class S3DIS_Umamba(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1',  fps_n_list = [512, 128, 32]):
        self.row, self.data, self.seg = load_data_semseg_Umamba_test(partition, test_area)  # 返回numpy数组[M,N,9]---data
        self.num_points = num_points
        self.partition = partition
        self.fps_n_list = fps_n_list

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        row = self.row[item][:self.num_points]
        seg = self.seg[item][:self.num_points]

        pc_rand_idx = np.arange(self.num_points)
        np.random.shuffle(pc_rand_idx)
        pointcloud = pointcloud[pc_rand_idx, :]
        row = row[pc_rand_idx, :]
        seg = seg[pc_rand_idx]

        points, voxel_indices, voxel_total, voxel_valid = voxelization(pointcloud, 0.4)
        fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, self.fps_n_list)

        seg = torch.LongTensor(seg)

        return row, pointcloud, seg, fps_index_array, series_idx_arrays

    def __len__(self):
        return self.data.shape[0]

class S3DIS_Umamba_DX(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1',  fps_n_list = [512, 128, 32]):
        self.row, self.data, self.seg = load_data_semseg_Umamba_test(partition, test_area)  # 返回numpy数组[M,N,9]---data
        self.num_points = num_points
        self.partition = partition
        self.fps_n_list = fps_n_list

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        row = self.row[item][:self.num_points]
        seg = self.seg[item][:self.num_points]

        pc_rand_idx = np.arange(self.num_points)
        np.random.shuffle(pc_rand_idx)
        pointcloud = pointcloud[pc_rand_idx, :]
        row = row[pc_rand_idx, :]
        seg = seg[pc_rand_idx]

        points, voxel_indices, voxel_total, voxel_valid = voxelization(pointcloud, 0.4)
        fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, self.fps_n_list)

        seg = torch.LongTensor(seg)

        return row, pointcloud, seg,fps_index_array,series_idx_arrays

    def __len__(self):
        return self.data.shape[0]