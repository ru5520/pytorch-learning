"""PyTorch Learning 01: Tensor & Matrix Multiplication"""

import torch

print("=" * 50)
print("01 - Tensor & Matrix Multiplication")
print("=" * 50)

x = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
w = torch.arange(1, 13, dtype=torch.float32).view(4, 3)

print(f"x shape: {x.shape}  (3 rows, 4 cols)")
print(f"w shape: {w.shape}  (4 rows, 3 cols)")
print(f"x @ w shape: {(x @ w).shape}")

print(f"\nx =\n{x}")
print(f"\nw =\n{w}")
print(f"\ny = x @ w =\n{x @ w}")

y = x @ w
verify = x[0,0]*w[0,0] + x[0,1]*w[1,0] + x[0,2]*w[2,0] + x[0,3]*w[3,0]
print(f"\nManual y[0,0] = {verify.item():.1f}")
print(f"Matrix y[0,0] = {y[0,0].item():.1f}")

print("\nConclusion: Matrix multiplication reshapes information linearly.")
