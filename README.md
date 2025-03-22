# Causal Attentive Statistics Pooling

This is a minimal PyTorch implementation of **Causal Attentive Statistic Pooling**, useful for scenarios where online or streaming inference requires strict causality.

## ✨ Features

- Computes weighted mean and std over time using attention
- Enforces **causality** with lower-triangular attention masking
- Optionally includes cumulative mean/std for richer context

## 📦 Usage

```python
from causal_attentive_pooling import CausalAttentiveStatisticsPooling
import torch

x = torch.randn(1, 4, 10)  # [batch, channels, time]
lengths = torch.tensor([10])
model = CausalAttentiveStatisticsPooling(4)
output = model(x, lengths)
print(output.shape)  # torch.Size([1, 8])
```

## ✅ Causality Guarantee

This layer ensures that each timestep only attends to itself and previous frames using a strict lower-triangular mask.


