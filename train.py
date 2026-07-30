import torch
import torch.nn as nn
from transformer import transformer

D_MODEL = 128
NUM_HEADS = 4
D_FF = 256
DROPOUT = 0.1
LAYERS = 2
SRC_VOCAB_SIZE = 1000
TGT_VOCAB_SIZE = 1000

BATCH_SIZE = 16
EPOCHS = 20
LR = 0.01
GRAD_CLIP = 1.0
PAD_ID = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_masks(src, tgt):
    src_pad = (src != PAD_ID).unsqueeze(1).unsqueeze(2)
    tgt_pad = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
    look_ahead = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=DEVICE)).bool()
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_pad & look_ahead
    return src_pad, tgt_mask


def generate_dummy_data(num_samples, max_len):
    src = torch.randint(1, SRC_VOCAB_SIZE, (num_samples, max_len))
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (num_samples, max_len))
    return src, tgt


def train():
    model = transformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
        dropout=DROPOUT, layers=LAYERS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    src_data, tgt_data = generate_dummy_data(num_samples=500, max_len=20)
    total_batches = len(src_data) // BATCH_SIZE

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(src_data))
        src_data, tgt_data = src_data[perm], tgt_data[perm]
        total_loss = 0

        for i in range(total_batches):
            src = src_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(DEVICE)
            tgt = tgt_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input)

            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(output.reshape(-1, TGT_VOCAB_SIZE), tgt_label.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:3d} | Loss: {total_loss / total_batches:.4f}")

    return model


if __name__ == "__main__":
    train()
