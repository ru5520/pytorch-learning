# Stage 03: MNIST 手写数字识别

## 文件说明

| 文件 | 内容 |
|------|------|
| 01_data_loading.py | 数据加载与可视化 |
| 02_training.py | 完整训练流程 |

## MNIST 数据集

- 训练集: 60000 张图片
- 测试集: 10000 张图片
- 图片尺寸: 28 × 28 像素（灰度）
- 标签: 0-9（手写数字）

## 学习目标

1. 学会使用 `torchvision.datasets` 加载数据
2. 学会使用 DataLoader 实现 mini-batch 训练
3. 构建完整的神经网络进行分类
4. 评估模型准确率

## 运行

```bash
python 01_data_loading.py
python 02_training.py