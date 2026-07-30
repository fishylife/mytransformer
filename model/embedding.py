import torch.nn as nn
import math
import torch

class embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

if __name__ == "__main__":
    #test
    embedding_layer = embedding(d_model=512, vocab_size=10000)
    x = torch.randint(0, 10000, (1, 10))  # Example input tensor with shape (batch_size, sequence_length)
    print(x.shape)  # Should print torch.Size([1, 10])
    output = embedding_layer(x)
    print(output.shape)  # Should print torch.Size([1, 10, 512])