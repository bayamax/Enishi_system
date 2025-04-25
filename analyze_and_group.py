#!/usr/bin/env python3
# file: analyze_and_group.py
import csv, pickle, json
import networkx as nx
import numpy as np
from networkx.algorithms import community
from collections import Counter

##################### CFG #####################
EDGE_FILE      = "edges_full.csv"
VEC_FILE       = "account_vectors.npy"   # dict{screen_name: np.array}
OUT_COMM_FILE  = "communities.json"      # コミュニティ→ノード一覧
OUT_HUB_FILE   = "community_hubs.json"   # representative/hub ノード
TOP_HUBS_PER_C = 10                      # 何位まで “主要” とみなすか
##############################################

print("1) load edges …")
G = nx.read_edgelist(EDGE_FILE, delimiter=",", create_using=nx.Graph())

print("2) Louvain (networkx-community) …")
# greedy_modularity uses Clauset-Newman (高速)。小規模なら問題なし
communities = list(community.greedy_modularity_communities(G))
print(f"   found {len(communities)} communities")

print("3) centrality & hub pick …")
deg_centrality = nx.degree_centrality(G)
comm_dict   = {}
hub_dict    = {}

for i, comm in enumerate(communities, start=1):
    comm_name = f"comm_{i:03d}"
    comm_nodes = list(comm)
    comm_dict[comm_name] = comm_nodes
    
    # コミュニティ内で中心性が高い順
    ranked = sorted(
        comm_nodes,
        key=lambda n: deg_centrality[n],
        reverse=True
    )[:TOP_HUBS_PER_C]
    hub_dict[comm_name] = ranked

# ---------- save ----------
with open(OUT_COMM_FILE, "w", encoding="utf-8") as f:
    json.dump(comm_dict, f, ensure_ascii=False, indent=2)

with open(OUT_HUB_FILE,  "w", encoding="utf-8") as f:
    json.dump(hub_dict,  f, ensure_ascii=False, indent=2)

print(f"» saved {OUT_COMM_FILE} and {OUT_HUB_FILE}")

# ---------- optional: “キャラ付け” 用にベクトルも紐付け ----------
vecs = np.load(VEC_FILE, allow_pickle=True).item()
hub_vecs = {
    comm: {node: vecs[node].tolist() for node in hubs if node in vecs}
    for comm, hubs in hub_dict.items()
}
with open("hub_vectors.json", "w", encoding="utf-8") as f:
    json.dump(hub_vecs, f, ensure_ascii=False, indent=2)
print("» saved hub_vectors.json (for your downstream NLP/分類)")