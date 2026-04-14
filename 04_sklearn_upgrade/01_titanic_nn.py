"""
Titanic 神经网络版
==================

用 PyTorch 重做之前的 Titanic 生存预测
对比 sklearn 版本和神经网络版本
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 50)
print("Titanic 神经网络版")
print("=" * 50)

# ===== 1. 加载数据 =====
train_df = pd.read_csv("train.csv")
print(f"\n训练集大小: {train_df.shape}")

# ===== 2. 数据预处理 =====
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# 复制一份，避免警告
df = train_df.copy()

# 处理缺失值 - 用中位数填充数值列
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 性别转数值
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# 港口转数值
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 检查是否还有 NaN
print(f"\n各列缺失值数量:")
print(df[features].isnull().sum())

# 提取特征和标签
X = df[features].values.astype(np.float32)
y = df["Survived"].values

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\n特征数量: {X.shape[1]}")
print(f"训练样本数: {X.shape[0]}")
print(f"生还人数: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)}")

# ===== 3. 构建网络 =====
class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

model = TitanicNet(input_size=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print(f"\n模型结构:\n{model}")

# ===== 4. 训练 =====
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)

best_acc = 0
print("\n开始训练:")

for epoch in range(1, 201):
    model.train()
    y_pred = model(X_train_t)
    loss = loss_fn(y_pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_acc = (val_pred.argmax(1) == y_val_t).float().mean().item()
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | 验证准确率: {val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc

print(f"\n最佳验证准确率: {best_acc*100:.2f}%")

# ===== 5. 对比 sklearn 版本 =====
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 50)
print("与 sklearn 版本对比")
print("=" * 50)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print(f"sklearn 逻辑回归: {lr.score(X_val, y_val)*100:.2f}%")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"sklearn 随机森林: {rf.score(X_val, y_val)*100:.2f}%")

print(f"PyTorch 神经网络: {best_acc*100:.2f}%")

print("\n结论: sklearn 和神经网络各有优劣，神经网络在小数据集上不一定优于传统方法")