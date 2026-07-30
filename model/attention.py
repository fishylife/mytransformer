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

class multi_head_attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_d = d_model // num_heads
        self.Wk = nn.Linear(d_model, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        K = self.Wk(x).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        V = self.Wv(x).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        score = torch.matmul(K, Q.transpose(-2, -1))/ math.sqrt(self.head_d)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=-1)
        score = self.dropout(score)
        output = torch.matmul(score, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.Wo(output)
        return output

class cross_attention(multi_head_attention):
    def forward(self, k, q, v, mask = None):
        batch_size = q.size(0)
        K = self.Wk(k).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        Q = self.Wq(q).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        V = self.Wv(v).view(batch_size, -1, self.num_heads, self.head_d).transpose(1, 2)
        score = torch.matmul(K, Q.transpose(-2, -1))/ math.sqrt(self.head_d)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=-1)
        score = self.dropout(score)
        output = torch.matmul(score, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.Wo(output)
        return output



if __name__ == "__main__":
    #test
    x = torch.randn(1, 2, 512)
    attention_test = cross_attention(d_model=512, num_heads=8)
    output = attention_test(x, x, x)
    print(output)