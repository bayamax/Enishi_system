import os
import csv
import numpy as np
import torch
import torch.nn as nn
import hashlib
from collections import deque
import random
import networkx as nx
from node2vec import Node2Vec

###################################
# 設定パラメータ
###################################
ACCOUNT_VECTORS_FILE = "account_vectors.npy"
TRANSFORMER_MODEL_FILE = "transformer_for_like.pth"
EDGES_FILE = "edges.csv"
EMBED_DIM = 128
TOP_K = 5
MAX_DEPTH = 2  # ブリッジ探索最大深度
SUBGRAPH_RADIUS = 1 # ブリッジ先からどの程度の範囲でサブグラフ抽出するか
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################
# データロード
###################################
account_vectors = np.load(ACCOUNT_VECTORS_FILE, allow_pickle=True).item()
nodes = set(account_vectors.keys())

# 全エッジ読み込み（フォローネットワーク）
G_full = nx.DiGraph()
with open(EDGES_FILE, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        src, dst = row
        G_full.add_edge(src, dst)

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

model = NodeTransformerModel(vocab_size=len(nodes), embed_dim=EMBED_DIM).to(DEVICE)
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
    h = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    seed_val = int(h[:8], 16)
    rstate = np.random.RandomState(seed_val)
    vec = rstate.randn(dim).astype(np.float32)
    return vec

###################################
# フォローネットワーク取得（モック）
###################################
def get_user_following(user_id):
    # 実際はTwitter APIでuser_idのフォロー中アカウントを取得
    # モック：ランダムに既存G_fullから候補を選択
    all_nodes = list(G_full.nodes())
    sample_count = random.randint(2,6)
    return random.sample(all_nodes, min(sample_count, len(all_nodes)))

###################################
# ブリッジ探索
###################################
def find_bridge_account(user_id, max_depth=2):
    visited = set([user_id])
    queue = deque([(user_id, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth > max_depth:
            break
        if current in nodes and current != user_id:
            return current
        followings = get_user_following(current)
        for f in followings:
            if f not in visited:
                visited.add(f)
                queue.append((f, depth+1))
    return None

###################################
# サブグラフ抽出
# ブリッジ先ノードを中心に半径SUBGRAPH_RADIUS以内のノードを取得
###################################
def extract_subgraph(bridge_account, user_id, radius=1):
    # BFSで半径radius以内のノード集合を取得
    # さらにuser_idを新規ノードとして追加し、bridge_accountからuser_idへのエッジを加えるなどして連結
    visited = set([bridge_account])
    queue = deque([(bridge_account, 0)])
    sub_nodes = set([bridge_account])
    while queue:
        cur, d = queue.popleft()
        if d >= radius:
            continue
        # curがフォローしているノード取得
        # 全ノード中、すでにvisitedでないノードを追加
        for nxt in G_full.successors(cur):
            if nxt not in visited:
                visited.add(nxt)
                sub_nodes.add(nxt)
                queue.append((nxt, d+1))

    # 新規ユーザーをこのサブグラフに追加
    # 仮に user_id -> bridge_account のエッジを追加（方向はお好みで）
    # こうすることでユーザーがブリッジ先に関連づけられる
    sub_nodes.add(user_id)

    subG = nx.DiGraph()
    for n in sub_nodes:
        # エッジ追加
        for nxt in G_full.successors(n):
            if nxt in sub_nodes:
                subG.add_edge(n, nxt)

    # user_id と bridge_accountを接続（例：user_id -> bridge_account）
    subG.add_edge(user_id, bridge_account)
    return subG

###################################
# サブグラフ上でNode2Vec実行
###################################
def node2vec_on_subgraph(subG, seed_str, dim=128):
    # seed固定
    h = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    seed_val = int(h[:8],16)
    np.random.seed(seed_val)
    random.seed(seed_val)

    node2vec = Node2Vec(subG, dimensions=dim, walk_length=80, num_walks=10, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4, iter=1)
    # 埋め込み抽出
    sub_vectors = {}
    for n in subG.nodes():
        vec = model.wv[n]
        sub_vectors[n] = vec.astype(np.float32)
    return sub_vectors

###################################
# ユーザーベクトル取得ロジック（Scenario A）
###################################
def get_user_vector(user_id):
    if user_id in account_vectors:
        return account_vectors[user_id]

    bridge_account = find_bridge_account(user_id, max_depth=MAX_DEPTH)
    if bridge_account is not None:
        # サブグラフ抽出
        subG = extract_subgraph(bridge_account, user_id, radius=SUBGRAPH_RADIUS)
        # seedは (user_id + bridge_account)で安定乱数
        seed_str = f"{user_id}-{bridge_account}"
        sub_vectors = node2vec_on_subgraph(subG, seed_str, EMBED_DIM)
        # sub_vectorsにuser_idが含まれているはず
        return sub_vectors[user_id]
    else:
        # ブリッジ失敗時はuser_idのみでseed化
        return stable_random_vector(str(user_id), EMBED_DIM)

###################################
# 類似アカウント探索
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

###################################
# メイン処理例
###################################
if __name__ == "__main__":
    input_user_id = "new_input_user_5678"
    user_vec = get_user_vector(input_user_id)
    like_vector = like_extractor(user_vec)
    similar = find_similar_accounts(like_vector, account_vectors, top_k=TOP_K)
    print("User:", input_user_id)
    print("User Vector (first 5 elems):", user_vec[:5])
    print("Recommended:", similar)