import os
import numpy as np
import torch
import torch.nn as nn
import random

ACCOUNT_VECTORS_FILE = "account_vectors.npy"
TRANSFORMER_MODEL_FILE = "transformer_for_like.pth"
EMBED_DIM = 128
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

account_vectors = np.load(ACCOUNT_VECTORS_FILE, allow_pickle=True).item()
nodes = list(account_vectors.keys())
node_id_mapping = {n:i for i,n in enumerate(nodes)}
vocab_size = len(nodes)

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
        return out, logits

model = NodeTransformerModel(vocab_size=vocab_size, embed_dim=EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load(TRANSFORMER_MODEL_FILE, map_location=DEVICE))
model.eval()

# ユーザーベクトル取得（フォロワー数最大アカウントをハブに簡易近似）
def get_user_following(user_id):
    # 実際はTwitter APIで取得するがここではモック
    follow_count = random.randint(3,10)
    return random.sample(nodes, min(follow_count, len(nodes)))

def get_user_vector(user_id):
    if user_id in account_vectors:
        return account_vectors[user_id]
    followings = get_user_following(user_id)
    if len(followings) == 0:
        return np.random.randn(EMBED_DIM).astype(np.float32)
    # 複数あれば平均などに拡張可能。ここでは例示的に1つ選択+ノイズ
    hub = random.choice(followings)
    if hub in account_vectors:
        hub_vec = account_vectors[hub]
        user_vec = hub_vec + 0.01*np.random.randn(EMBED_DIM).astype(np.float32)
        return user_vec
    else:
        return np.random.randn(EMBED_DIM).astype(np.float32)

# LikeVector抽出用
class LikeVectorExtractor(nn.Module):
    def __init__(self, transformer_model, embed_dim=128):
        super().__init__()
        self.transformer = transformer_model.transformer
        self.input_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_vec):
        x = torch.tensor(user_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        x = self.input_linear(x)
        x = x.transpose(0,1)
        out = self.transformer(x)
        out = out.transpose(0,1)
        like_vector = out.squeeze(0).squeeze(0).detach().cpu().numpy()
        return like_vector

like_extractor = LikeVectorExtractor(model).to(DEVICE)
like_extractor.eval()

def find_similar_accounts(like_vector, account_vectors, top_k=5):
    scores = []
    for uid, vec in account_vectors.items():
        numerator = np.dot(like_vector, vec)
        denom = (np.linalg.norm(like_vector)*np.linalg.norm(vec)+1e-9)
        sim = numerator/denom
        scores.append((uid, sim))
    scores.sort(key=lambda x:x[1], reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    input_user_id = "input_user_1234"
    user_vec = get_user_vector(input_user_id)
    like_vector = like_extractor(user_vec)
    similar = find_similar_accounts(like_vector, account_vectors, top_k=TOP_K)
    print("User:", input_user_id)
    print("Recommended:", similar)