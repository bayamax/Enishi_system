import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

ACCOUNT_VECTORS_FILE = "account_vectors.npy"
WALKS_FILE = "walks.txt"
OUTPUT_MODEL = "transformer_for_like.pth"

EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 3
MASK_PROB = 0.15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

account_dict = np.load(ACCOUNT_VECTORS_FILE, allow_pickle=True).item()
nodes = list(account_dict.keys())
node_id_mapping = {n: i for i,n in enumerate(nodes)}
vocab_size = len(nodes)

walks = []
with open(WALKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        seq = [node_id_mapping[p] for p in parts if p in node_id_mapping]
        if len(seq) > 0:
            walks.append(seq)

random.shuffle(walks)
train_size = int(len(walks)*0.9)
train_seqs = walks[:train_size]
valid_seqs = walks[train_size:]

class MaskedNodeDataset(Dataset):
    def __init__(self, sequences, vocab_size, mask_prob=0.15):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_token_id = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = []
        label_seq = []
        for token_id in seq:
            if random.random() < self.mask_prob:
                input_seq.append(self.mask_token_id)
                label_seq.append(token_id)
            else:
                input_seq.append(token_id)
                label_seq.append(-100)
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(label_seq, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max([len(x) for x in xs])
    pad_token = vocab_size+1
    padded_x, padded_y = [], []
    for x,y in zip(xs,ys):
        pad_len = max_len - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.tensor([pad_token]*pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.tensor([-100]*pad_len, dtype=torch.long)])
        padded_x.append(x)
        padded_y.append(y)
    return torch.stack(padded_x, dim=0), torch.stack(padded_y, dim=0)

train_dataset = MaskedNodeDataset(train_seqs, vocab_size=vocab_size, mask_prob=MASK_PROB)
valid_dataset = MaskedNodeDataset(valid_seqs, vocab_size=vocab_size, mask_prob=MASK_PROB)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class NodeTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        emb = emb.transpose(0,1)
        out = self.transformer(emb)
        out = out.transpose(0,1)
        logits = self.fc(out)
        return logits

model = NodeTransformerModel(vocab_size=vocab_size, embed_dim=EMBED_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x,y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss/len(loader)

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, valid_loader, loss_fn)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {train_loss:.4f}, Valid Loss = {val_loss:.4f}")

torch.save(model.state_dict(), OUTPUT_MODEL)
print("Model saved to", OUTPUT_MODEL)