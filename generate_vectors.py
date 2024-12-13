import networkx as nx
import numpy as np
from node2vec import Node2Vec
import csv

EDGE_FILE = 'edges_full.csv'  # 修正: 'edges.csv'→'edges_full.csv'
OUTPUT_VEC_FILE = 'account_vectors.npy'
WALKS_FILE = 'walks.txt'

DIMENSIONS = 128
NUM_WALKS = 10
WALK_LENGTH = 80
WINDOW_SIZE = 10
WORKERS = 4
EPOCHS = 1
ALLOW_PICKLE = True

print("Loading edges from:", EDGE_FILE)
G = nx.DiGraph()
with open(EDGE_FILE, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        src, dst = row
        G.add_edge(src, dst)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

node2vec = Node2Vec(
    G, 
    dimensions=DIMENSIONS, 
    walk_length=WALK_LENGTH, 
    num_walks=NUM_WALKS, 
    workers=WORKERS
)

# ランダムウォーク出力
with open(WALKS_FILE, 'w', encoding='utf-8') as wf:
    for walk in node2vec.walks:
        wf.write(" ".join(walk) + "\n")

print("Fitting Node2Vec model...")
model = node2vec.fit(
    window=WINDOW_SIZE, 
    min_count=1, 
    batch_words=4, 
    iter=EPOCHS
)
print("Model training done.")

account_vectors = {}
for node in G.nodes():
    vec = model.wv[node]
    account_vectors[node] = vec.astype(np.float32)

np.save(OUTPUT_VEC_FILE, account_vectors, allow_pickle=ALLOW_PICKLE)
print("Saved account vectors to", OUTPUT_VEC_FILE)
print("Walks saved to", WALKS_FILE)
print("All done!")
