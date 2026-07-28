import torch
import torch.nn as nn
import torch.optim as optim
import math


class self_attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)

        score = torch.softmax(torch.matmul(K, Q.T) / math.sqrt(x.size(-1)), dim=-1)
        output = torch.matmul(score, V)
        return output


#test
x = torch.randn(1, 2)
attention = self_attention(d_model=2)
output = attention(x)
print(output)