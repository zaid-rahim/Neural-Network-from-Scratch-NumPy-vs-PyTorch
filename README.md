# Neural Network from Scratch — NumPy vs PyTorch

A 2-layer neural network (4 → 16 → 3) trained on the Iris dataset, implemented twice: once with raw NumPy (manual forward pass, backpropagation, and gradient descent) and once with PyTorch (autograd, `nn.Module`, built-in optimizers). Both achieve ~95%+ test accuracy on identical data and architecture.

This project exists to build a from-scratch understanding of how neural networks actually learn, before relying on framework abstractions.

## Why this project

Most people learn PyTorch directly and never see what's happening underneath `.backward()`. This project goes the other way: implement every matrix multiplication, every gradient, and every weight update by hand in NumPy first, then rewrite the identical model in PyTorch to see exactly what the framework automates.

## What's inside

- `numpy_nn.py` — full neural network with manual forward pass, backpropagation (chain rule derived by hand), cross-entropy loss, and gradient descent
- `pytorch_nn.py` — the same architecture using `nn.Module`, `nn.CrossEntropyLoss`, and `optim.SGD`
- Loss curve plots for both implementations
- Side-by-side comparison of what each line of NumPy code maps to in PyTorch

## Architecture

```
Input (4 features) → Linear → ReLU → Linear → Softmax → Output (3 classes)
```

Trained on the Iris dataset (150 samples, 4 features, 3 classes), with an 80/20 train/test split and standardized input features.

## Results

| Implementation | Train Accuracy | Test Accuracy |
|---|---|---|
| NumPy (from scratch) | ~98/3% | ~99% |
| PyTorch | ~95% | ~99% |

Both converge to a similar loss curve starting near `1.0986` (the expected loss for random 3-class guessing, `-log(1/3)`) and decreasing steadily over 1000 epochs.

## Key implementation details

**Forward pass (NumPy):**
```python
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
```

**Backward pass (NumPy):** Gradients derived manually using the chain rule, including the simplified softmax + cross-entropy gradient `(a2 - y_true)`.

**PyTorch equivalent:** Same architecture defined as an `nn.Module`, with `loss.backward()` replacing the entire manual backpropagation function, and `optimizer.step()` replacing the manual weight update loop.

## What I learned

- How backpropagation works at the matrix-dimension level — not just conceptually, but tracing exact shapes through every layer
- Why the combined softmax + cross-entropy gradient simplifies to `(a2 - y_true)`, and why PyTorch's `CrossEntropyLoss` applies softmax internally rather than expecting it pre-applied
- Why numerical stability tricks matter in practice (`np.clip` before `log`, max-subtraction before `exp` in softmax)
- Exactly what PyTorch abstracts away: weight initialization, gradient computation via autograd, and the optimizer step — all of which I had to write by hand first

## Tech stack

NumPy, PyTorch, scikit-learn (Iris dataset, train/test split), Matplotlib

## How to run

```bash
pip install numpy torch matplotlib scikit-learn

python numpy_nn.py
python pytorch_nn.py
```

## Next steps

This is Week 1 of a 12-week AI/LLM engineering roadmap. Next: implementing a transformer from scratch and understanding the attention mechanism at the same level of depth.
