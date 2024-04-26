import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    class get_model(nn.Module):
        """
        初始化模型结构。

        参数:
        - num_classes: 整数，表示分类的数量。
        - normal_channel: 布尔值，决定是否使用普通通道。
        """
        def __init__(self, num_classes, normal_channel=False):
            super(get_model, self).__init__()
            # 根据是否使用普通通道，决定额外通道的数量
            if normal_channel:
                additional_channel = 3
            else:
                additional_channel = 0

            self.normal_channel = normal_channel

            # 定义第一个点云集采样模块
            self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])

            # 定义第二个点云集采样模块
            self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])

            # 定义第三个点云集采样模块
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

            # 定义三个点云特征传播模块
            self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
            self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
            self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])

            # 定义两个卷积层和一个批量归一化层、一个dropout层用于特征提取的末尾
            self.conv1 = nn.Conv1d(128, 128, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(0.5)

            # 定义最后一层卷积层，用于输出分类结果
            self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    class get_loss:
        """
        初始化get_loss类。
        该类主要用于初始化损失函数。继承自父类，以确保类的正确初始化。
        """
        def __init__(self):
            # 调用父类的初始化方法
            super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        """
        计算并返回前向传播的总损失。

        参数:
        - pred: 网络的预测输出，通常是概率分布形式。
        - target: 目标标签，即网络预测的正确标签。
        - trans_feat: 转换特征，可能用于辅助计算损失或进行其他操作，本函数中未使用。

        返回值:
        - total_loss: 基于预测和目标计算的总损失。
        """
        # 计算负对数似然损失
        total_loss = F.nll_loss(pred, target)

        return total_loss