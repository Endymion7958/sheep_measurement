# -*-coding:utf-8 -*-
import os  # 用于操作文件和目录
import json  # 用于处理JSON文件
import warnings  # 用于处理警告信息
import numpy as np  # 用于数值计算
from torch.utils.data import Dataset  # 导入PyTorch的Dataset类

# 忽略警告信息
warnings.filterwarnings('ignore')

# 定义一个函数，用于对点云进行归一化处理
def pc_normalize(pc):
    """
    对点云数据进行标准化处理。

    参数:
    pc: numpy数组，代表点云数据，其中每一行是一个点的坐标。

    返回值:
    标准化后的点云数据，使其具有单位范数。
    """

    # 计算点云数据的质心（中心点）
    centroid = np.mean(pc, axis=0)

    # 将点云数据中的每个点减去质心，以质心为原点重新坐标化
    pc = pc - centroid

    # 计算点云数据在重新坐标化后的最大长度（投影到任意轴上的最大长度）
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

    # 将点云数据中的每个点按最大长度进行缩放，使其具有单位范数
    pc = pc / m

    return pc


class PartNormalDataset(Dataset):
    class ShapeNetCorePartAnnotation:
        """
        构建ShapeNetCore部分注解数据集的加载器。

        参数:
        - root: 数据集根目录的路径，默认为'./data/shapenetcore_partanno_segmentation_benchmark_v0_normal'。
        - npoints: 采样点的数量，默认为2500。
        - split: 数据集的划分，可以选择'train'、'val'、'test'或'trainval'，默认为'train'。
        - class_choice: 选择加载特定类别数据的选项，默认为None，即加载所有类别。
        - normal_channel: 是否包含法向量通道，默认为False。
        """

        def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
            # 初始化参数和基本类别信息
            self.npoints = npoints
            self.root = root
            self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
            self.cat = {}
            self.normal_channel = normal_channel

            # 读取类别文件，映射类别ID到类别名称
            with open(self.catfile, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]
            self.cat = {k: v for k, v in self.cat.items()}
            self.classes_original = dict(zip(self.cat, range(len(self.cat))))

            # 根据class_choice过滤类别
            if not class_choice is None:
                self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

            # 加载数据集划分信息，并根据split选择对应的数据
            self.meta = {}
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
                test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            for item in self.cat:
                self.meta[item] = []
                dir_point = os.path.join(self.root, self.cat[item])
                fns = sorted(os.listdir(dir_point))
                if split == 'trainval':
                    fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
                elif split == 'train':
                    fns = [fn for fn in fns if fn[0:-4] in train_ids]
                elif split == 'val':
                    fns = [fn for fn in fns if fn[0:-4] in val_ids]
                elif split == 'test':
                    fns = [fn for fn in fns if fn[0:-4] in test_ids]
                else:
                    print('Unknown split: %s. Exiting..' % (split))
                    exit(-1)
                for fn in fns:
                    token = (os.path.splitext(os.path.basename(fn))[0])
                    self.meta[item].append(os.path.join(dir_point, token + '.txt'))

            # 构建数据路径列表和类别映射
            self.datapath = []
            for item in self.cat:
                for fn in self.meta[item]:
                    self.datapath.append((item, fn))

            self.classes = {}
            for i in self.cat.keys():
                self.classes[i] = self.classes_original[i]

            # 构建类别到分割标签的映射
            self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                                'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                                'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                                'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

            # 初始化缓存
            self.cache = {}
            self.cache_size = 20000

    # 定义__getitem__方法，用于按索引获取数据集中的样本
    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    # 定义__len__方法，返回数据集中样本的数量
    def __len__(self):
        return len(self.datapath)



