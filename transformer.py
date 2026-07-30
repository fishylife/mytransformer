import torch.nn as nn
import torch
from model.encoder import encoder
from model.decoder import decoder
from model.position import positional_encoding
from model.embedding import embedding
import math

class transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, src_vocab_size, tgt_vocab_size, dropout=0.1, layers=6):
        super().__init__()
        self.encoder_embedding = embedding(d_model, src_vocab_size)
        self.decoder_embedding = embedding(d_model, tgt_vocab_size)
        self.position = positional_encoding(d_model, max_len=5000)
        self.encoder = encoder(d_model, num_heads, d_ff, dropout, layers)
        self.decoder = decoder(d_model, num_heads, d_ff, dropout, layers)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder_embedding(src)
        src = self.position(src)
        src = self.dropout(src)
        src = self.encoder(src, src_mask)

        tgt = self.decoder_embedding(tgt)
        tgt = self.position(tgt)
        tgt = self.dropout(tgt)
        tgt = self.decoder(tgt, src, src_mask, tgt_mask)

        output = self.output_layer(tgt)
        return output

#test
transformer_model = transformer(d_model=512, num_heads=8, d_ff=2048, src_vocab_size=10000, tgt_vocab_size=10000, dropout=0.1, layers=6)
src = torch.randint(0, 10000, (1, 10))  # Example source input tensor with shape (batch_size, sequence_length)
tgt = torch.randint(0, 10000, (1, 10))  # Example target input tensor with shape (batch_size, sequence_length)
print(src.shape)  # Should print torch.Size([1, 10])
print(tgt.shape)  # Should print torch.Size([1, 10])
output = transformer_model(src, tgt)
print(output.shape)  # Should print torch.Size([1, 10, 10000])

