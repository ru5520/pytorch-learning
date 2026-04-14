"""PyTorch Learning 04: Training Loop"""

import torch

x = torch.tensor([[3.0, 4.0]])
y_true = torch.tensor([[10.0]])

w1 = torch.randn(2, 8, requires_grad=True)
b1 = torch.zeros(1, 8, requires_grad=True)
w2 = torch.randn(8, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)

print("Training Loop (200 steps):")
for i in range(200):
    h = torch.relu(x @ w1 + b1)
    y_pred = h @ w2 + b2
    loss = (y_pred - y_true) ** 2
    loss.backward()
    with torch.no_grad():
        for p in [w1, b1, w2, b2]:
            p -= 0.01 * p.grad
            p.grad.zero_()
    if (i + 1) % 40 == 0:
        print(f"  Step {i+1:3d} | Pred: {y_pred.item():.4f} | Loss: {loss.item():.4f}")

print("\nTraining Loop: Forward -> Loss -> Backward -> Update -> Repeat")
