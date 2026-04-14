# Stage 02: PyTorch Standard Framework

## What We Learned

| # | File | Concept |
|---|------|---------|
| 01 | 01_nn_module.py | nn.Module, nn.Linear, torch.optim |
| 02 | 02_optimizer_comparison.py | SGD vs Adam comparison |

## Key Transformations

| Operation | Old (Manual) | New (nn.Module) |
|-----------|-------------|----------------|
| Define params | `w1 = torch.randn(2,8)` | `self.layer1 = nn.Linear(2,8)` |
| Forward pass | `h = torch.relu(x @ w1 + b1)` | `x = torch.relu(self.layer1(x))` |
| Zero grad | `param.grad.zero_()` | `optimizer.zero_grad()` |
| Update params | `param -= lr * param.grad` | `optimizer.step()` |

## Optimizers

- **SGD**: Simple, uses the same learning rate for all parameters. Fast start but can oscillate.
- **Adam**: Adaptive learning rate per parameter. Slower start but more stable convergence.

## When to Use Which

- Simple tasks → SGD is fine
- Complex deep learning → **Adam is preferred**
- Fine-tuning with learning rate scheduling → SGD with scheduler
