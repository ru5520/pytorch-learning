"""PyTorch Learning: nn.Module Framework"""

import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNet()
print("Model Structure:")
print(model)

print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}")

x = torch.tensor([[3.0, 4.0]])
y_true = torch.tensor([[10.0]])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining with nn.Module + Adam:")
for i in range(200):
    y_pred = model(x)
    loss = (y_pred - y_true) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 40 == 0:
        print(f"  Step {i+1:3d} | Pred: {y_pred.item():.4f} | Loss: {loss.item():.4f}")

print("\nOld way vs New way: manually define params -> nn.Linear")
print("Optimizer: manual gradient descent -> torch.optim.Adam")
