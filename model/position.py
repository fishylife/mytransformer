import torch.nn as nn
import torch
import math

class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        position = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0)/d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

#test
pos_encoding = positional_encoding(d_model=512, max_len=100)
x = torch.randn(1, 10, 512)  # Example input tensor with shape (batch_size, sequence_length, d_model)
output = pos_encoding(x)
print(output)
