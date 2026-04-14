# Stage 01: Neural Network Basics

## What We Learned

| # | File | Concept |
|---|------|---------|
| 01 | 01_tensor_matmul.py | Tensor, matrix multiplication, shapes |
| 02 | 02_neural_network_from_scratch.py | Weights, biases, activation, multi-layer |
| 03 | 03_gradient_backprop.py | Gradient, backward(), parameter responsibility |
| 04 | 04_training_loop.py | Training loop, learning rate, convergence |

## Key Concepts

### Forward Propagation
Data flows from input to output, each layer does "weighted sum + activation".

### Backpropagation
Starting from the loss, automatically computes how much each parameter contributed to the error.

### Gradient Descent
Update parameters opposite to gradient direction: param = param - lr × gradient

### ReLU
`max(0, x)`, introduces non-linearity so the network can learn complex patterns.

## Review Questions

1. **Why do we need multiple layers?**
   Single layer (linear regression) has limited capacity, can only represent straight lines.

2. **Why are gradients not "averaged"?**
   Whoever influences the error gets a gradient. Parameters with zero influence get zero gradient.

3. **What happens if learning rate is too large or too small?**
   Too large → oscillation (overshoots minimum). Too small → very slow convergence.
