import torch.nn as nn
import torch
from model.attention import multi_head_attention, cross_attention
from model.FFN import FFN

class decoder_layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = multi_head_attention(d_model, num_heads, dropout)
        self.cross_attention = cross_attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attention(encoder_output, x, encoder_output, src_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x

class decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, layers=6):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer(d_model, num_heads, d_ff, dropout) for _ in range(layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

if __name__ == "__main__":
    #test
    d_model = 512
    num_heads = 8
    d_ff = 2048
    decoder_layer_test = decoder_layer(d_model, num_heads, d_ff)
    x = torch.randn(1, 10, d_model)
    encoder_output = torch.randn(1, 10, d_model)
    output = decoder_layer_test(x, encoder_output)
    print(output.shape)
