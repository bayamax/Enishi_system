from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# ChromeDriverのパス設定
CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"

# 手動で取得したクッキー情報
cookies = [
    {"name": "auth_token", "value": "9a604e4fff4c52f910bdccdb783178695577f109"},
    {"name": "ct0", "value": "18d9b526a83508c2cc02007bff6b32deb187a92c1c645283189bc1babbbef59a4dfff4703595b8adbe382c04cbf9fd25bc5402f4b4db8b3b7247d2649d163206a"},
    {"name": "twid", "value": "u%3D1782363447843491840"}
]

# ターゲットユーザーID
USERNAME = "cloudproject_ad"
FOLLOWER_URL = f"https://twitter.com/{USERNAME}/followers"

# Seleniumの初期設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

def fetch_followers_with_selenium():
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Twitterのホームページにアクセス
        driver.get("https://twitter.com")
        print("Twitterのホームページにアクセスしました。")

        # クッキーを追加
        for cookie in cookies:
            driver.add_cookie(cookie)

        # クッキー適用後、フォローページにアクセス
        driver.get(FOLLOWER_URL)
        print(f"フォローページにアクセスしました: {FOLLOWER_URL}")

        # リダイレクトされたURLを確認
        current_url = driver.current_url
        if current_url != FOLLOWER_URL:
            print(f"リダイレクトされています: {current_url}")
            return None

        # 要素が読み込まれるまで待機
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//div[@data-testid='UserCell']"))
        )

        # ページのHTMLを取得
        html = driver.page_source
        print("フォローページのHTMLを取得しました。")

        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(html, "html.parser")
        followers = []
        for user_cell in soup.find_all("div", {"data-testid": "UserCell"}):
            user_link = user_cell.find("a", href=True)
            if user_link:
                username = user_link["href"].split("/")[-1]
                followers.append(username)

        return followers

    finally:
        driver.quit()

# メイン処理
if __name__ == "__main__":
    followers = fetch_followers_with_selenium()
    if followers:
        print("取得したフォロワーID一覧:")
        print(followers)
    else:
        print("フォロワーページにアクセスできませんでした。")