"""PyTorch Learning 02: Neural Network from Scratch"""

import torch

print("=" * 50)
print("02 - Neural Network from Scratch")
print("=" * 50)

x = torch.tensor([[3.0, 4.0]])
y_true = torch.tensor([[10.0]])

w1 = torch.randn(2, 8, requires_grad=True)
b1 = torch.zeros(1, 8, requires_grad=True)
w2 = torch.randn(8, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)

print(f"\nInput: {x}")
print(f"Target: {y_true.item()}")
print(f"\nNetwork: 2 inputs -> 8 hidden -> 1 output")

for i in range(50):
    h = torch.relu(x @ w1 + b1)
    y_pred = h @ w2 + b2
    loss = (y_pred - y_true) ** 2
    loss.backward()
    with torch.no_grad():
        for p in [w1, b1, w2, b2]:
            p -= 0.01 * p.grad
            p.grad.zero_()
    if (i + 1) % 10 == 0:
        print(f"  Step {i+1:2d} | Pred: {y_pred.item():.4f} | Loss: {loss.item():.4f}")

print("\nConclusion: Multi-layer networks can learn any mapping.")
