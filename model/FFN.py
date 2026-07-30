import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    #test
    ffn = FFN(d_model=512, d_ff=2048, dropout=0.1)
    x = torch.randn(1, 10, 512)  # Example input tensor with shape (batch_size, sequence_length, d_model)
    output = ffn(x)
    print(output)

