from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Chromeの設定
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # ヘッドレスモード
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--remote-debugging-port=9222")

# ChromeDriverの起動
service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

# 任意のURLにアクセス
driver.get("https://www.example.com")

# クッキーを取得
cookies = driver.get_cookies()
print("クッキー:", cookies)

# ChromeDriverを終了
driver.quit()
