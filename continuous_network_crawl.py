from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# Chromeオプション設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # ヘッドレスモード（画面表示なし）
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.199 Safari/537.36")

# ChromeDriverのサービス指定
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Twitterのホームページにアクセス
    driver.get("https://twitter.com")
    print("Twitterのホームページにアクセスしました。")

    # 手動で取得したクッキー情報を追加
    cookies = [
        {"name": "auth_token", "value": "9a604e4fff4c52f910bdccdb783178695577f109"},
        {"name": "ct0", "value": "18d9b526a83508c2cc02007bff6b32deb187a92c1b2588bec8da4433fe10bd7b2dcbd0c1c645283189bc1babbbef59a4dfff4703595b8adbe382c04cbf9fd25bc5402f4b4db8b3b7247d2649d163206a"},
        {"name": "twid", "value": "u%3D1782363447843491840"}
    ]

    for cookie in cookies:
        driver.add_cookie(cookie)

    # クッキーを適用した後、ページを再読み込み
    driver.get("https://twitter.com/home")
    print("クッキーを適用しました。ページを再読み込みします。")

    # フォロワーページにアクセス
    user_id = "1782363447843491840"
    url = f"https://twitter.com/{user_id}/followers"
    driver.get(url)
    print("フォロワーページにアクセスしました。")

    # 要素が表示されるまで待機
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[@data-testid='UserCell']//a"))
    )
    print("フォロワー要素が見つかりました！")

    # フォロワーリンクを取得
    followers = []
    follower_elements = driver.find_elements(By.XPATH, "//div[@data-testid='UserCell']//a")
    for element in follower_elements:
        user_link = element.get_attribute("href")
        print("取得したリンク:", user_link)
        followers.append(user_link.split('/')[-1])

finally:
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)