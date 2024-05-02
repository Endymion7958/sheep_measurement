import os
import numpy as np
import json
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class SheepDataset(Dataset):
    def __init__(self, root='./data/sheep', npoints=2500, split='train', normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.datapath = []
        self.split = split
        self.catfile = os.path.join(self.root, 'category.txt')  # 假设有一个category.txt文件包含类别信息
        self.cat = {}
        self.normal_channel = normal_channel

        # 读取类别文件，映射类别ID到类别名称
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        # 加载数据集划分信息，并根据split选择对应的数据
        self.meta = {}
        train_ids = set()  # 这里需要根据实际的train, val, test划分文件来加载
        val_ids = set()
        test_ids = set()
        if split == 'trainval':
            train_ids = self._load_split('train_ids.txt')  # 需要提供训练集ID的文件路径
            val_ids = self._load_split('val_ids.txt')  # 需要提供验证集ID的文件路径
        elif split == 'train':
            train_ids = self._load_split('train_ids.txt')
        elif split == 'val':
            val_ids = self._load_split('val_ids.txt')
        elif split == 'test':
            test_ids = self._load_split('test_ids.txt')
        else:
            print('Unknown split: %s. Exiting..' % split)
            exit(-1)

        # 根据split集合构建数据路径列表
        self.datapath = []
        for cat in self.cat:
            dir_point = os.path.join(self.root, self.cat[cat])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if (os.path.splitext(os.path.basename(fn))[0] in train_ids) or (
                            os.path.splitext(os.path.basename(fn))[0] in val_ids)]
            elif split == 'train':
                fns = [fn for fn in fns if os.path.splitext(os.path.basename(fn))[0] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if os.path.splitext(os.path.basename(fn))[0] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if os.path.splitext(os.path.basename(fn))[0] in test_ids]
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.datapath.append(os.path.join(dir_point, token + '.txt'))

        # 初始化缓存
        self.cache = {}
        self.cache_size = 20000

    def _load_split(self, split_file):
        split_path = os.path.join(self.root, split_file)
        with open(split_path, 'r') as f:
            split_ids = [line.strip() for line in f.readlines()]
        return set(split_ids)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, _, _ = self.cache[index]
        else:
            fn = self.datapath[index]
            data = np.loadtxt(fn).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)  # 假设最后一列是分割标签

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, np.array([0]), seg)  # 类别标签固定为0

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # 返回点云数据，类别标签（虽然这里是固定的0），和分割标签
        return point_set, np.array([0]), seg

    def __len__(self):
        return len(self.datapath)