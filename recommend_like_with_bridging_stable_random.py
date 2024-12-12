import os
import numpy as np
import torch
import torch.nn as nn
import hashlib
from collections import deque
import random

###################################
# 設定パラメータ
###################################
ACCOUNT_VECTORS_FILE = "account_vectors.npy"
TRANSFORMER_MODEL_FILE = "transformer_for_like.pth"
EMBED_DIM = 128
TOP_K = 5
MAX_DEPTH = 2  # ブリッジ探索最大深度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################
# データロード
###################################
account_vectors = np.load(ACCOUNT_VECTORS_FILE, allow_pickle=True).item()
nodes = list(account_vectors.keys())
vocab_size = len(nodes)

###################################
# モデル定義
###################################
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

###################################
# 安定した乱数ベクトル生成関数
###################################
def stable_random_vector(seed_str, dim=128):
    # seed_strに基づいて安定的な乱数ベクトル生成
    h = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    seed_val = int(h[:8], 16)
    rstate = np.random.RandomState(seed_val)
    vec = rstate.randn(dim).astype(np.float32)
    return vec

###################################
# フォローネットワーク取得（モック）
# 実際はTwitter APIでユーザーIDからフォロー中アカウント取得が必要
###################################
def get_user_following(user_id):
    # モック：データベースのノードからサンプル
    follow_count = random.randint(2,6)
    return random.sample(nodes, min(follow_count, len(nodes)))

###################################
# ブリッジ探索
# BFSでMAX_DEPTHまでユーザーのフォロー→フォロー→...で既知アカウントを探す
###################################
def find_bridge_account(user_id, max_depth=2):
    visited = set([user_id])
    queue = deque([(user_id, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth > max_depth:
            break
        if current in account_vectors and current != user_id:
            # 自身がDBに無いユーザーを想定しているのでcurrent != user_idは冗長だが保険的に
            return current
        # 次探索
        followings = get_user_following(current)
        for f in followings:
            if f not in visited:
                visited.add(f)
                queue.append((f, depth+1))
    return None

###################################
# ユーザーベクトル取得ロジック
# データベース内：そのまま
# なければブリッジ探索で既知アカウント発見
# 発見すればユーザーID + 見つかったアカウントIDでseedを生成
# 見つからなければユーザーIDのみでseed生成
###################################
def get_user_vector(user_id):
    if user_id in account_vectors:
        return account_vectors[user_id]
    # ブリッジ探索
    bridge_account = find_bridge_account(user_id, max_depth=MAX_DEPTH)
    if bridge_account is not None:
        # ブリッジ成功：ユーザーIDとbridge_account IDを組み合わせたseed
        seed_str = f"{user_id}-{bridge_account}"
    else:
        # ブリッジ失敗：ユーザーIDのみ
        seed_str = str(user_id)
    return stable_random_vector(seed_str, EMBED_DIM)

###################################
# 類似アカウント探索 (コサイン類似度)
###################################
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
    print("User Vector (first 5 elems):", user_vec[:5])
    print("Recommended:", similar)