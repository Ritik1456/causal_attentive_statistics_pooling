import torch
import torch.nn as nn
import torch.nn.functional as F


def length_to_mask(length, max_len=None, device=None):
    if max_len is None:
        max_len = length.max().item()
    return torch.arange(max_len, device=device).expand(len(length), max_len) < length.unsqueeze(1)


class CausalAttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.channels = channels
        self.global_context = global_context
        
        # Input size depends on whether we use global context
        input_size = channels * 3 if global_context else channels
        self.attention_layer1 = nn.Linear(input_size, attention_channels)
        self.attention_layer2 = nn.Linear(attention_channels, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x, lengths=None):
        batch_size, channels, time_steps = x.shape

        # Generate mask for valid time steps
        if lengths is None:
            lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=x.device)
        
        mask = length_to_mask(lengths, max_len=time_steps, device=x.device).float()
        
        # Compute cumulative mean and variance for global context
        if self.global_context:
            cumsum_x = torch.cumsum(x * mask.unsqueeze(1), dim=2)
            count = torch.cumsum(mask, dim=1).clamp(min=1).unsqueeze(1)
            causal_mean = cumsum_x / count
            cumsum_x2 = torch.cumsum((x ** 2) * mask.unsqueeze(1), dim=2)
            causal_var = (cumsum_x2 / count) - (causal_mean ** 2)
            causal_std = torch.sqrt(causal_var.clamp(min=self.eps))
            attention_input = torch.cat([x, causal_mean, causal_std], dim=1)
        
        else:
            attention_input = x
        
        
        # Compute attention weights
        attention_input = attention_input.transpose(1, 2)  # Shape: [batch, time_steps, features]
        attn_hidden = self.tanh(self.attention_layer1(attention_input))
        attn_logits = self.attention_layer2(attn_hidden).squeeze(-1)  # [batch, time_steps]
        
        
        # Causal masking: Each position can attend only to previous positions
        causal_mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device))
        attn_logits = attn_logits.unsqueeze(1).expand(-1, time_steps, -1)  # Expand for broadcasting
        attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0) == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=2)  # [batch, time_steps, time_steps]
        
        
        # Compute weighted mean directly
        weighted_mean = torch.bmm(attn_weights, x.transpose(1, 2)).transpose(1, 2)  # [batch, channels, time_steps]
        
        
        # Compute weighted variance directly
        squared_diff = (x - weighted_mean) ** 2
        weighted_var = torch.bmm(attn_weights, squared_diff.transpose(1, 2)).transpose(1, 2)
        weighted_std = torch.sqrt(weighted_var.clamp(min=self.eps))


        # Final weighted pooling
        final_weights = attn_weights.sum(dim=2, keepdim=True) / (attn_weights.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True) + self.eps)
        final_mean = torch.sum(weighted_mean * final_weights.transpose(1, 2), dim=2)
        final_std = torch.sum(weighted_std * final_weights.transpose(1, 2), dim=2)
        return torch.cat([final_mean, final_std], dim=1)  # Shape: [batch, 2 * channels]

