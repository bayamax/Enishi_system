from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import requests
import time

# --- 設定情報 ---
# ChromeDriverのパス
CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"

# 手動で取得したクッキー情報
cookies = [
    {"name": "auth_token", "value": "9a604e4fff4c52f910bdccdb783178695577f109"},
    {"name": "ct0", "value": "18d9b526a83508c2cc02007bff6b32deb187a92c1b2588bec8da4433fe10bd7b2dcbd0c1c645283189bc1babbbef59a4dfff4703595b8adbe382c04cbf9fd25bc5402f4b4db8b3b7247d2649d163206a"},
    {"name": "twid", "value": "u%3D1782363447843491840"}
]

# ターゲットのフォロワーページURL
TARGET_USER_ID = "cloudproject_ad"  # ユーザーID
FOLLOWER_URL = f"https://twitter.com/{TARGET_USER_ID}/followers"

# ユーザーエージェント設定
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.199 Safari/537.36"
}

# --- Seleniumでクッキー取得＆requests用に渡す ---
def get_twitter_cookies():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.199 Safari/537.36")

    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Twitterのホームページにアクセス
        driver.get("https://twitter.com")
        print("Twitterのホームページにアクセスしました。")

        # クッキーを設定
        #for cookie in cookies:
            #driver.add_cookie(cookie)

        # クッキー適用後、フォロワーページにアクセス
        driver.get(FOLLOWER_URL)
        print("クッキーを適用し、フォロワーページにアクセスしました。")

        # Seleniumで取得したクッキーをrequests用に変換
        #session_cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
        #print("クッキーをrequests用に取得しました。")
        #return session_cookies

    finally:
        driver.quit()

# --- requestsでHTMLを取得＆ファイル保存 ---
def get_followers_html(cookies):
    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value)

    response = session.get(FOLLOWER_URL, headers=HEADERS)
    if response.status_code == 200:
        print("フォロワーページのHTMLを取得しました。")

        # HTMLをファイルに保存して内容を確認
        with open("followers_page.html", "w", encoding="utf-8") as file:
            file.write(response.text)
        print("HTMLをfollowers_page.htmlに保存しました。")

        return response.text
    else:
        print(f"HTML取得に失敗しました。ステータスコード: {response.status_code}")
        return None

# --- HTMLを解析してIDを抽出 ---
def extract_ids_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    ids = []

    # aタグのhref属性を抽出し、IDを取得
    for link in soup.find_all("a", href=True):
        href = link['href']
        if href.startswith("/") and len(href) > 1:  # ユーザーリンクを判定
            user_id = href.split("/")[-1]
            if user_id not in ids:  # 重複防止
                ids.append(user_id)

    print("IDの抽出が完了しました。")
    return ids

# --- メイン処理 ---
if __name__ == "__main__":
    try:
        # SeleniumでTwitterにアクセスし、requests用クッキーを取得
        session_cookies = get_twitter_cookies()

        # requestsでフォロワーページのHTMLを取得
        html_content = get_followers_html(session_cookies)

        # HTMLを解析してフォロワーIDを抽出
        if html_content:
            ids = extract_ids_from_html(html_content)
            print("取得したフォロワーID一覧:")
            print(ids)

    except Exception as e:
        print(f"エラーが発生しました: {e}")