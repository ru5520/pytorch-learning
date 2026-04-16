"""保存 CNN 模型权重"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
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

train_df = pd.read_csv("D:/prictice/pytorch-learning/05_kaggle/digit_recognizer/train.csv")
X = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
y = train_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=128, shuffle=True
)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("训练模型...")
for epoch in range(1, 6):
    model.train()
    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  Epoch {epoch} 完成")

save_path = "D:/prictice/pytorch-learning/07_deployment/mnist_cnn.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ 模型权重已保存: {save_path}")