from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Twitterのログイン情報を入力
USERNAME = "cloudproject_ad"
PASSWORD = "F=M*a+C*v^2"

# Chromeオプション設定
options = webdriver.ChromeOptions()
#options.add_argument("--headless")  # ヘッドレスモード（画面なし）
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# ChromeDriverの起動
service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

try:
    # Twitterのログインページにアクセス
    driver.get("https://twitter.com/login")
    time.sleep(5)  # ページ読み込みを待機

    # ユーザー名を入力
    username_input = driver.find_element(By.NAME, "text")
    username_input.send_keys(USERNAME)
    username_input.send_keys(Keys.RETURN)
    time.sleep(5)

    # パスワードを入力
    password_input = driver.find_element(By.NAME, "password")
    password_input.send_keys(PASSWORD)
    password_input.send_keys(Keys.RETURN)
    time.sleep(5)  # ログイン完了を待機

    # 特定のページにアクセス
    driver.get("https://twitter.com/cloudproject_ad/followers")
    time.sleep(5)

    # クッキーを取得
    cookies = driver.get_cookies()
    print("取得したクッキー:")
    for cookie in cookies:
        print(cookie)

finally:
    # ChromeDriverを終了
    driver.quit()