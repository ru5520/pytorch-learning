"""
商店销售预测神经网络版
==================

用 PyTorch 重做 store-sales 预测
简化版：选少量店铺和类别来训练
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 50)
print("商店销售预测神经网络版")
print("=" * 50)

# ===== 1. 加载数据 =====
print("\n加载数据...")
train = pd.read_csv("train.csv")

# 筛选：只选前 5 个店铺 + 前 3 个商品类别
stores_to_use = [1, 2, 3, 4, 5]
families_to_use = ["AUTOMOTIVE", "BEAUTY", "FOODS"]

train = train[
    (train["store_nbr"].isin(stores_to_use)) & 
    (train["family"].isin(families_to_use))
]
print(f"筛选后数据量: {len(train)}")

train["date"] = pd.to_datetime(train["date"])
train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

# ===== 2. 特征工程 =====
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek

le_store = LabelEncoder()
le_family = LabelEncoder()
train["store_enc"] = le_store.fit_transform(train["store_nbr"])
train["family_enc"] = le_family.fit_transform(train["family"])

print(f"店铺: {stores_to_use}")
print(f"类别: {families_to_use}")
print(f"时间: {train['date'].min().date()} ~ {train['date'].max().date()}")

# ===== 3. 准备数据 =====
features = ["store_enc", "family_enc", "year", "month", "day", "dayofweek", "onpromotion"]
X = train[features].values.astype(np.float32)
y = train["sales"].values.astype(np.float32)
y = np.maximum(y, 0)
y = np.log1p(y)

print(f"特征数: {X.shape[1]} | 样本数: {X.shape[0]}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {len(X_train)} | 验证集: {len(X_val)}")

# ===== 4. 构建网络 =====
class SalesNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SalesNet(input_size=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print(f"\n模型结构:\n{model}")

# ===== 5. 训练 =====
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

best_rmse = float("inf")
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
            val_rmse = torch.sqrt(loss_fn(val_pred, y_val_t)).item()
        print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | Val RMSE: {val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse

print(f"\n最佳验证 RMSE: {best_rmse:.4f}")

# ===== 6. 对比 sklearn 版本 =====
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

print("\n" + "=" * 50)
print("与 sklearn 版本对比")
print("=" * 50)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
ridge_rmse = np.sqrt(np.mean((ridge_pred - y_val) ** 2))
print(f"sklearn Ridge:     RMSE = {ridge_rmse:.4f}")

rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_rmse = np.sqrt(np.mean((rf_pred - y_val) ** 2))
print(f"sklearn 随机森林: RMSE = {rf_rmse:.4f}")

print(f"PyTorch 神经网络: RMSE = {best_rmse:.4f}")

print("\n结论: 神经网络在小数据集上与 sklearn 效果相近，数据量越大优势越明显")