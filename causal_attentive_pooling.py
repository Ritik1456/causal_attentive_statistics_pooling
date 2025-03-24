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
            # Compute cumulative statistics efficiently
            cumsum_x = torch.cumsum(x * mask.unsqueeze(1), dim=2)
            cumsum_count = torch.cumsum(mask, dim=1).unsqueeze(1).clamp(min=1)
            causal_mean = cumsum_x / cumsum_count
            
            cumsum_x2 = torch.cumsum((x ** 2) * mask.unsqueeze(1), dim=2)
            causal_var = (cumsum_x2 / cumsum_count) - (causal_mean ** 2)
            causal_std = torch.sqrt(causal_var.clamp(min=self.eps))
            
            # Concatenate for attention input
            attention_input = torch.cat([x, causal_mean, causal_std], dim=1)
        else:
            attention_input = x
            
        # Transpose for attention calculation
        attention_input = attention_input.transpose(1, 2)  # [batch, time_steps, features]
        
        # Calculate attention scores
        attn_hidden = self.tanh(self.attention_layer1(attention_input))
        attn_scores = self.attention_layer2(attn_hidden).squeeze(-1)  # [batch, time_steps]
        
        # Create causal attention mask (we'll use this for fully batched processing)
        causal_mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device))
        
        # For proper causal attention, create a matrix where each row t has scores for positions 0 to t
        # First, we need to expand attn_scores to broadcast properly
        attn_scores = attn_scores.unsqueeze(1)  # [batch, 1, time_steps]
        
        # Create a mask that keeps only valid positions for each timestep
        valid_positions_mask = causal_mask.unsqueeze(0)  # [1, time_steps, time_steps]
        
        # Apply the mask to the scores
        attn_scores_masked = attn_scores.repeat(1, time_steps, 1)  # [batch, time_steps, time_steps]
        attn_scores_masked = attn_scores_masked.masked_fill(valid_positions_mask == 0, float('-inf'))
        
        # Also apply sequence length mask
        seq_mask = mask.unsqueeze(1)  # [batch, 1, time_steps]
        attn_scores_masked = attn_scores_masked.masked_fill(~seq_mask.repeat(1, time_steps, 1).bool(), float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores_masked, dim=2)  # [batch, time_steps, time_steps]
        
        # Compute weighted mean (each position t attends to positions 0 to t)
        x_t = x.transpose(1, 2)  # [batch, time_steps, channels]
        weighted_mean = torch.bmm(attn_weights, x_t)  # [batch, time_steps, channels]
        
        # Compute weighted variance
        # We need to calculate (x - mean)² weighted by attention
        x_expanded = x_t.unsqueeze(1)  # [batch, 1, time_steps, channels]
        mean_expanded = weighted_mean.unsqueeze(2)  # [batch, time_steps, 1, channels]
        
        # Calculate squared differences
        squared_diff = (x_expanded - mean_expanded) ** 2  # [batch, time_steps, time_steps, channels]
        
        # Reshape attention weights for broadcasting
        attn_weights_for_var = attn_weights.unsqueeze(-1)  # [batch, time_steps, time_steps, 1]
        
        # Apply attention weights to squared differences
        weighted_var = torch.sum(attn_weights_for_var * squared_diff, dim=2)  # [batch, time_steps, channels]
        weighted_std = torch.sqrt(weighted_var.clamp(min=self.eps))
        
        # Apply mask to get valid statistics only
        masked_mean = weighted_mean * mask.unsqueeze(-1)
        masked_std = weighted_std * mask.unsqueeze(-1)
        
        # Compute final statistics by averaging over valid timesteps
        norm_factor = mask.sum(dim=1).unsqueeze(-1).clamp(min=self.eps)
        final_mean = masked_mean.sum(dim=1) / norm_factor
        final_std = masked_std.sum(dim=1) / norm_factor
        
        return torch.cat([final_mean, final_std], dim=1)  # [batch, 2*channels]