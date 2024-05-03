import os
import warnings
import numpy as np
import json
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/sheep', npoints=2500,
                 split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.cat = {}
        self.normal_channel = normal_channel

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
        self.meta = {}  # 读取分好类的文件夹jason文件并将其名字放入列表中
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
                print('Unknown split: %s. Exiting..' % split)
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
        self.seg_classes = {'Sheep': [0, 1, 2, 3]}

        # 初始化缓存
        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            # 读取并处理数据
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            seg = data[:, -1].astype(np.int32)  # 假设最后一列是分割标签

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # 返回点云数据和分割标签
        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
