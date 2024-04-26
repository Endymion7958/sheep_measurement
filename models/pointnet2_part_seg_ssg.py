import torch.nn as nn   # 导入PyTorch的神经网络模块
import torch    # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的功能性模块
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
# 从自定义模块导入PointNetSetAbstraction和PointNetFeaturePropagation类

# 定义一个名为get_model的类，它继承自nn.Module，用于构建神经网络模型
class get_model(nn.Module):
    """
    构造PointNet2模型用于分类任务。

    参数:
    - num_classes(int): 类别的数量。
    - normal_channel(bool): 是否使用法向量通道，默认为False。

    属性:
    - sa1, sa2, sa3: PointNetSetAbstraction模块，用于进行点云的采样和特征提取。
    - fp3, fp2, fp1: PointNetFeaturePropagation模块，用于特征传播。
    - conv1, bn1, drop1, conv2: 用于特征处理的卷积层、批量归一化层和dropout层。
    """
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()   # 调用基类的构造函数
        # 根据是否使用法向量通道决定额外的通道数
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel    # 标记是否包含法线信息
        # 定义三个Set Abstraction (SA) 层，用于点云特征的提取
        # 初始化PointNetSetAbstraction模块
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6+additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # 定义三个Feature Propagation (FP) 层，用于将高层的特征信息传递回低层
        # 初始化PointNetFeaturePropagation模块
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        # 定义两个全连接层和批量归一化层，以及一个Dropout层用于正则化
        # 初始化卷积层、批量归一化层和dropout层
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    # 定义模型的前向传播过程
    def forward(self, xyz, cls_label):
        """
        前向传播函数。

        参数:
        - xyz(Tensor): 输入的点云坐标，形状为(B, C, N)，B为批次大小，C为通道数，N为点的数量。
        - cls_label(Tensor): 类别标签，形状为(B, )。

        返回:
        - x(Tensor): 经过模型处理得到的分类概率分布，形状为(B, num_classes, N)。
        - l3_points(Tensor): 最后一层的特征点，用于其他任务或进一步的处理。
        """
        # 进行Set Abstraction操作
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
        # 进行Feature Propagation操作
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # 将类别标签扩展为和点云同样大小的张量，用于条件特征传播
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # 进行全连接层操作
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # 调整形状以便输出
        x = x.permute(0, 2, 1)
        return x, l3_points

# 定义一个用于计算损失的类，它继承自nn.Module
class get_loss(nn.Module):
    """
    定义损失函数计算模块。
    """

    # 定义一个用于计算损失的类，它继承自nn.Module
    def __init__(self):
        super(get_loss, self).__init__()    # 调用基类的构造函数

    # 定义损失函数的计算过程
    def forward(self, pred, target, trans_feat):
        """
        计算模型预测和真实标签之间的损失。

        参数:
        - pred(Tensor): 模型预测的概率分布，形状为(B, num_classes, N)。
        - target(Tensor): 真实标签，形状为(B, )。
        - trans_feat(Tensor): 传递给损失函数的特征，形状依赖于具体任务。

        返回:
        - total_loss(Tensor): 总损失，单一标量张量。
        """
        total_loss = F.nll_loss(pred, target)   # 使用负对数似然损失函数计算分类损失

        return total_loss
