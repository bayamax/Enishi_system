import os
import requests
import time
import csv
import sys

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YOUR_BEARER_TOKEN_HERE")
HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"}
INTERVAL = 3600  # 1時間おきに実行
MAX_RESULTS = 1000
RATE_LIMIT_SLEEP = 900  # レート超過時のスリープ
SEEDS = ["123456789", "987654321"]  # シードユーザーIDを適宜指定
API_BASE = "https://api.twitter.com/2"
USER_FIELDS = "id,username"

# 出力ファイル
EDGES_FILE = "edges_full.csv"
VISITED_FILE = "visited_users.csv"
QUEUE_FILE = "queue_users.csv"

def ensure_file(fname, header=None):
    if not os.path.isfile(fname):
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)

def load_set_from_csv(fname):
    s = set()
    if os.path.isfile(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                s.add(row[0])
    return s

def append_to_csv(fname, rows, header=None):
    file_exists = os.path.isfile(fname)
    mode = 'a' if file_exists else 'w'
    with open(fname, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

def load_queue():
    q = []
    if os.path.isfile(QUEUE_FILE):
        with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                q.append(row[0])
    return q

def save_queue(q):
    append_to_csv(QUEUE_FILE, [[u] for u in q], header=["user_id"])

def fetch_followers(user_id, max_results=1000, pagination_token=None):
    url = f"{API_BASE}/users/{user_id}/followers"
    params = {
        "max_results": max_results,
        "user.fields": USER_FIELDS
    }
    if pagination_token:
        params["pagination_token"] = pagination_token

    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 429:
        print("Rate limit exceeded. Sleeping...")
        time.sleep(RATE_LIMIT_SLEEP)
        resp = requests.get(url, headers=HEADERS, params=params)

    if resp.status_code != 200:
        print("Error:", resp.text)
        return [], None

    data = resp.json()
    if "data" not in data:
        return [], None

    followers = data["data"]
    next_token = data["meta"].get("next_token", None)
    return followers, next_token

def main():
    # ファイルがなければ初期化
    ensure_file(EDGES_FILE, header=["source_user","target_user"])
    ensure_file(VISITED_FILE, header=["user_id"])
    ensure_file(QUEUE_FILE, header=["user_id"])

    visited = load_set_from_csv(VISITED_FILE)
    queue = load_queue()

    # 初回起動時にqueueが空ならシードを投入
    if not queue and SEEDS:
        queue = SEEDS[:]
        save_queue(queue)

    while True:
        if not queue:
            print("Queue is empty. Nothing to do.")
            print(f"Sleeping {INTERVAL} seconds...")
            time.sleep(INTERVAL)
            continue

        user_id = queue.pop(0)

        if user_id in visited:
            print(f"User {user_id} already visited, skipping.")
        else:
            print(f"Fetching followers of {user_id}...")
            all_edges = []
            next_token = None
            count = 0
            while True:
                folls, token = fetch_followers(user_id, max_results=MAX_RESULTS, pagination_token=next_token)
                if not folls:
                    break
                # followersのフォロワー関係: follower -> user_id
                # edges: (follower_id, user_id)
                for f in folls:
                    fid = f["id"]
                    all_edges.append((fid, user_id))
                    # 新規ユーザーをqueueに入れる（既出でなければ）
                    if fid not in visited:
                        # すぐには取得しないが後で巡回
                        queue.append(fid)

                if token is None:
                    break
                next_token = token
                count += len(folls)
                # 過剰な拡大を避ける場合、ある程度で中断可能
                if count > 10000:
                    break

            # エッジ追記
            if all_edges:
                append_to_csv(EDGES_FILE, all_edges)
                print(f"Appended {len(all_edges)} edges from {user_id}.")

            # visitedに追加
            append_to_csv(VISITED_FILE, [[user_id]])
            visited.add(user_id)

        # キュー保存
        # ユーザーを追加したので更新
        # 重複ユーザーはvisitedでガード
        # queue保存
        unique_queue = [u for u in queue if u not in visited]
        queue = unique_queue
        # 別途保存
        if queue:
            # 上書き保存
            os.remove(QUEUE_FILE)
            append_to_csv(QUEUE_FILE, [[u] for u in queue], header=["user_id"])
        else:
            ensure_file(QUEUE_FILE, header=["user_id"])

        print(f"Queue size: {len(queue)}")
        print(f"Visited size: {len(visited)}")
        print(f"Sleeping {INTERVAL} seconds before next iteration...")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()