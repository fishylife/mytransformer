import torch.nn as nn
import torch
from position import positional_encoding
from FFN import FFN
from embedding import embedding
from attention import multi_head_attention

class encode_layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.attention = multi_head_attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask = None):
        x = self.norm1(x + self.dropout1(self.attention(x, mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

class encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1, layers = 6):
        super().__init__()
        self.layers = nn.ModuleList([encode_layer(d_model, num_heads, d_ff, dropout) for _ in range(layers)])

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

#test
d_model = 512
num_heads = 8
d_ff = 2048
encoder_layer = encode_layer(d_model, num_heads, d_ff)
x = torch.randn(1, 10, d_model)  # Example input tensor with shape (batch_size, sequence_length, d_model)
output = encoder_layer(x)
print(output.shape)  # Should print torch.Size([1, 10, 512])

