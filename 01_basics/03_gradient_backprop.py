"""PyTorch Learning 03: Gradient & Backpropagation"""

import torch

x = torch.tensor([[1.0, 2.0]])
y_true = torch.tensor([[3.0]])

w1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(1, 4, requires_grad=True)
w2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)

h = torch.relu(x @ w1 + b1)
y_pred = h @ w2 + b2
loss = (y_pred - y_true) ** 2

print("Before training:")
print(f"  Prediction: {y_pred.item():.4f}, Target: {y_true.item():.4f}, Loss: {loss.item():.4f}")

loss.backward()
print(f"\nw2 gradients:\n{w2.grad}")
print(f"\nFirst two are 0 -> these weights have NO influence on loss.")

with torch.no_grad():
    for p in [w1, b1, w2, b2]:
        p -= 0.01 * p.grad
        p.grad.zero_()

h2 = torch.relu(x @ w1 + b1)
y_pred2 = h2 @ w2 + b2
loss2 = (y_pred2 - y_true) ** 2

print(f"\nAfter one update:")
print(f"  Prediction: {y_pred2.item():.4f}, Loss: {loss2.item():.4f}")
print(f"  Loss decreased: {loss.item():.4f} -> {loss2.item():.4f}")

print("\nConclusion: Gradients tell each parameter how much to adjust.")
