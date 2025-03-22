import torch
from causal_attentive_pooling import CausalAttentiveStatisticsPooling

# Simulated input (batch=1, channels=4, time=10)
x = torch.randn(1, 4, 10)
lengths = torch.tensor([10])

model = CausalAttentiveStatisticsPooling(channels=4)
model.eval()

with torch.no_grad():
    output = model(x, lengths)
    print("Output shape:", output.shape)  # Should be [1, 8]
