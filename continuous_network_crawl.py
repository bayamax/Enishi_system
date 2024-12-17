from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# Chromeオプション設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")

# ChromeDriverのサービス指定
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Twitterのフォロワーページにアクセス
    user_id = "1782363447843491840"
    url = f"https://twitter.com/{user_id}/followers"
    print("指定URLにアクセス中:", url)
    driver.get(url)

    # ページソースを確認（デバッグ用）
    print("ページソースの先頭1000文字:")
    print(driver.page_source[:1000])

    # 要素が存在するまで待機
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[@data-testid='UserCell']//a"))
    )
    print("要素が見つかりました！")

finally:
    driver.quit()