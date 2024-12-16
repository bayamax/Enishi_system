import requests
import time

# Twitter APIのBearer Tokenをここに設定してください
bearer_token = "YOUR_BEARER_TOKEN_HERE"

# APIリクエストのヘッダー
headers = {
    "Authorization": f"Bearer {bearer_token}"
}

def get_user_followers(user_id):
    url = f"https://api.twitter.com/2/users/{user_id}/followers"
    params = {
        "max_results": 1000,
        "user.fields": "id"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code} {response.text}")
    return [follower['id'] for follower in response.json()['data']]

def get_followers_for_multiple_users(user_ids):
    all_followers = set()
    for user_id in user_ids:
        followers = get_user_followers(user_id)
        all_followers.update(followers)
    return list(all_followers)

def main():
    # スタートユーザーのIDを設定
    start_user_id = "THE_USER_ID_HERE"
    
    # 1ホップのフォロワーを取得
    first_hop_followers = get_user_followers(start_user_id)
    print(f"1 hop followers: {len(first_hop_followers)}")

    # 2ホップのフォロワーを取得
    second_hop_followers = get_followers_for_multiple_users(first_hop_followers)
    print(f"2 hop followers: {len(second_hop_followers)}")

    # 結果を表示（全フォロワーのIDをリストアップ）
    all_followers = first_hop_followers + second_hop_followers
    print(f"All followers (1st and 2nd hop): {len(all_followers)}")

if __name__ == "__main__":
    main()