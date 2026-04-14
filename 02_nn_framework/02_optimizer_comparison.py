"""PyTorch Learning: SGD vs Adam Comparison"""

import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

x = torch.tensor([[3.0, 4.0]])
y_true = torch.tensor([[10.0]])

# SGD
model_sgd = SimpleNet()
opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
print("=== SGD ===")
for i in range(200):
    y_pred = model_sgd(x)
    loss = (y_pred - y_true) ** 2
    opt_sgd.zero_grad()
    loss.backward()
    opt_sgd.step()
    if (i + 1) % 40 == 0:
        print(f"  Step {i+1:3d} | Pred: {y_pred.item():.4f} | Loss: {loss.item():.4f}")

# Adam
model_adam = SimpleNet()
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)
print("\n=== Adam ===")
for i in range(200):
    y_pred = model_adam(x)
    loss = (y_pred - y_true) ** 2
    opt_adam.zero_grad()
    loss.backward()
    opt_adam.step()
    if (i + 1) % 40 == 0:
        print(f"  Step {i+1:3d} | Pred: {y_pred.item():.4f} | Loss: {loss.item():.4f}")

print("\nSGD: Fast start, oscillates around minimum.")
print("Adam: Slow start, converges more steadily.")
