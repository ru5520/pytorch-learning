"""
ResNet MNIST - 残差网络
========================

目标：构建 ResNet 风格的神经网络，理解残差连接
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 50)
print("ResNet MNIST - 残差网络")
print("=" * 50)

# ===== 1. 加载数据 =====
# 自动找上级的 digit_recognizer 文件夹里的 train.csv
data_path = os.path.join(script_dir, "..", "05_kaggle", "digit_recognizer", "train.csv")
print(f"\n加载数据: {data_path}")

train_df = pd.read_csv(data_path)

X = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
y = train_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_tr_t = torch.FloatTensor(X_train).view(-1, 1, 28, 28)
y_tr_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val).view(-1, 1, 28, 28)
y_val_t = torch.LongTensor(y_val)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)}")

# ===== 2. 残差块 =====
class ResidualBlock(nn.Module):
    """残差块：输出 = 输入 + F(x)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# ===== 3. ResNet 网络 =====
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print(f"\n模型结构:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n参数量: {total_params:,}")

# ===== 4. 训练 =====
print("\n开始训练:")
best_acc = 0

for epoch in range(1, 11):
    model.train()
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (y_pred.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_acc = correct / total * 100
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            val_correct += (y_pred.argmax(1) == batch_y).sum().item()
            val_total += batch_y.size(0)
    
    val_acc = val_correct / val_total * 100
    print(f"  Epoch {epoch:2d} | 训练: {train_acc:.2f}% | 验证: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc

print(f"\n最佳验证准确率: {best_acc:.2f}%")
print(f"\n对比:")
print(f"  简单 CNN (之前): 99.19%")
print(f"  ResNet (现在):   {best_acc:.2f}%")