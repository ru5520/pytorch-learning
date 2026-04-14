"""
MNIST 02: 完整训练流程
=========================

目标：
1. 构建神经网络（输入层→隐藏层→输出层）
2. 使用 DataLoader 实现 mini-batch 训练
3. 在测试集上评估准确率
"""

import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===== 1. 数据预处理 =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# DataLoader：批量加载，每次取 64 张图
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"训练集 batch 数量: {len(train_loader)}")
print(f"测试集 batch 数量: {len(test_loader)}")

# ===== 2. 构建网络 =====
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入 28×28=784 → 隐藏层 128 → 输出层 10
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)   # 把 28×28 拉平成 784
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print(f"\n模型结构:\n{model}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# ===== 3. 训练函数 =====
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        y_pred = model(data)
        loss = loss_fn(y_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = y_pred.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    acc = correct / total * 100
    print(f"Epoch {epoch} 完成 | 平均误差: {avg_loss:.4f} | 训练准确率: {acc:.2f}%")

# ===== 4. 测试函数 =====
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            y_pred = model(data)
            pred = y_pred.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    acc = correct / total * 100
    print(f"\n测试集准确率: {acc:.2f}% ({correct}/{total} 张图片正确)")
    return acc

# ===== 5. 开始训练 =====
print("\n开始训练:")
for epoch in range(1, 6):  # 训练 5 轮
    train(epoch)

print("\n最终测试结果:")
test()