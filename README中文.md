```markdown
# Pytorch 实现 PointNet 和 PointNet++

此仓库是 PointNet 和 PointNet++ 在 Pytorch 中的实现。

## 更新

### 2021/03/27
1. 发布了用于语义分割的预训练模型，其中 PointNet++ 可以达到 53.5% mIoU。
2. 在 log/ 中发布了用于分类和部分分割的预训练模型。

### 2021/03/20
更新了分类的代码，包括：
1. 为训练 ModelNet10 数据集添加了代码。使用设置 `--num_category 10`。
2. 添加了仅在 CPU 上运行的代码。使用设置 `--use_cpu`。
3. 添加了离线数据预处理的代码以加速训练。使用设置 `--process_data`。
4. 添加了使用均匀采样进行训练的代码。使用设置 `--use_uniform_sample`。

### 2019/11/26
1. 修正了之前代码中的一些错误，并添加了数据增强技巧。现在分类由仅 1024 个点可以达到 92.8%！
2. 添加了测试代码，包括分类和分割，以及带有可视化的语义分割。
3. 将所有模型组织到 `./models` 文件夹中，便于使用。

## 安装

最新的代码已在 Ubuntu 16.04、CUDA10.1、PyTorch 1.6 和 Python 3.7 上测试：
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## 分类 (ModelNet10/40)

### 数据准备
在这里下载对齐的 ModelNet 并保存在 `data/modelnet40_normal_resampled/`。

### 运行
你可以使用以下代码以不同的模式运行。
- 如果你想要使用离线数据处理，你可以在首次运行时使用 `--process_data`。
- 如果你想要训练 ModelNet10，你可以使用 `--num_category 10`。

#### ModelNet40
在 `./models` 中选择不同的模型。
- 例如，没有法线特征的 `pointnet2_ssg`:
  ```shell
  python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
  python test_classification.py --log_dir pointnet2_cls_ssg
  ```
- 例如，带有法线特征的 `pointnet2_ssg`:
  ```shell
  python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
  python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal
  ```
- 例如，使用均匀采样的 `pointnet2_ssg`:
  ```shell
  python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
  python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_fps
  ```

#### ModelNet10
与 ModelNet40 类似的设置，只是使用 `--num_category 10`。
- 例如，没有法线特征的 `pointnet2_ssg`:
  ```shell
  python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --num_category 10
  python test_classification.py --log_dir pointnet2_cls_ssg --num_category 10
  ```

### 性能

| 模型                          | 准确率  |
|-----------------------------|------|
| PointNet (官方)               | 89.2 |
| PointNet2 (官方)              | 91.9 |
| PointNet (Pytorch 无法线)      | 90.6 |
| PointNet (Pytorch 有法线)      | 91.4 |
| PointNet2_SSG (Pytorch 无法线) | 92.2 |
| PointNet2_SSG (Pytorch 有法线) | 92.4 |
| PointNet2_MSG (Pytorch 有法线) | 92.8 |

## 部分分割 (ShapeNet)

### 数据准备
在这里下载对齐的 ShapeNet 并保存在 `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`。

### 运行
```
检查 `./models` 中的模型
例如，`pointnet2_msg`
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
```

### 性能

| 模型                      | 实例平均 IoU | 类别平均 IoU |
|-------------------------|----------|----------|
| PointNet (官方)           | 83.7     | 80.4     |
| PointNet2 (官方)          | 85.1     | 81.9     |
| PointNet (Pytorch)      | 84.3     | 81.1     |
| PointNet2_SSG (Pytorch) | 84.9     | 81.8     |
| PointNet2_MSG (Pytorch) | 85.4     | 82.5     |

## 语义分割 (S3DIS)

### 数据准备
在这里下载 3D 室内解析数据集 (S3DIS)，并保存在 `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`。
```shell
cd data_utils
python collect_indoor3d_data.py
```
处理后的数据将保存在 `data/stanford_indoor3d/`。

### 运行
```
检查 `./models` 中的模型
例如，`pointnet2_ssg`
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```
可视化结果将保存在 `log/sem_seg/pointnet2_sem_seg/visual/`，你可以通过 MeshLab 来可视化这些 `.obj` 文件。

### 性能

| 模型                      | 总体准确率 | 类别平均 IoU | 大小     |
|-------------------------|-------|----------|--------|
| PointNet (Pytorch)      | 78.9  | 43.7     | 40.7MB |
| PointNet2_ssg (Pytorch) | 83.0  | 53.5     | 11.2MB |

## 可视化

### 使用 show3d_balls.py

```
构建 C++ 可视化代码
cd visualizer
bash build.sh
运行一个示例
python show3d_balls.py
```

### 使用 MeshLab

## 引用

如果你发现这个代码库对你的研究有帮助，请考虑引用它和我们的其他工作：

```markdown
@article{Pytorch_Pointnet_Pointnet2,
    Author = {Xu Yan},
    Title = {Pointnet/Pointnet++ Pytorch},
    Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
    Year = {2019}
}
@InProceedings{yan2020pointasnl,
    title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
    author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
    journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2020}
}
@InProceedings{yan2021sparse,
    title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
    author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
    journal={AAAI Conference on Artificial Intelligence ({AAAI})},
    year={2021}
}
@InProceedings{yan20222dpass,
    title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
    author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
    year={2022},
    journal={ECCV}
}
```

## 使用此代码库的选定项目

- PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19
- On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20
- Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20
- PCT: Point Cloud Transformer
- PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud
- Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22
```