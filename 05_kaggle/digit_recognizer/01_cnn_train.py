"""
Digit Recognizer - CNN 神经网络
=================================

用 CNN 在 Kaggle Digit Recognizer 比赛上正式提交
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 50)
print("Digit Recognizer - Kaggle 正式比赛")
print("=" * 50)

# ===== 1. 加载数据 =====
print("\n加载数据...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"训练集: {train_df.shape}")
print(f"测试集: {test_df.shape}")

# 分离特征和标签
X_train = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
y_train = train_df["label"].values

X_test = test_df.values.astype(np.float32) / 255.0

print(f"特征数: {X_train.shape[1]}")
print(f"标签分布: {np.bincount(y_train)}")

# ===== 2. 构建 CNN =====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一个卷积层: 1通道 -> 32通道
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 28x28 -> 14x14
            nn.Dropout(0.25),

            # 第二个卷积层: 32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 14x14 -> 7x7
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print(f"\n模型结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")

# ===== 3. 准备 DataLoader =====
# 划分训练集和验证集
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# 转换为张量
X_tr_t = torch.FloatTensor(X_tr)
y_tr_t = torch.LongTensor(y_tr)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False
)

print(f"\n训练集: {len(X_tr)} | 验证集: {len(X_val)}")

# ===== 4. 训练 =====
print("\n开始训练:")
best_acc = 0

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (y_pred.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total * 100

    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            val_correct += (y_pred.argmax(1) == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_acc = val_correct / val_total * 100
    print(f"  Epoch {epoch:2d} | 训练准确率: {train_acc:.2f}% | 验证准确率: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc

print(f"\n最佳验证准确率: {best_acc:.2f}%")

# ===== 5. 生成提交文件 =====
print("\n生成提交文件...")
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t).argmax(1).numpy()

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(test_pred) + 1),
    "Label": test_pred
})

submission.to_csv("submission.csv", index=False)
print(f"提交文件已保存: {len(submission)} 行")
print(f"预测分布:\n{pd.Series(test_pred).value_counts().sort_index()}")