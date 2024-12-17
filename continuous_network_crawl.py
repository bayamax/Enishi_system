from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# Chromeオプション設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")

# ChromeDriverのサービス指定
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

# Twitterのフォロワーページにアクセス
user_id = "1782363447843491840"
url = f"https://twitter.com/{user_id}/followers"
print("指定URLにアクセス中:", url)
driver.get(url)

followers = []

try:
    # ページの読み込みが完了するのを待機
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    print("ページの読み込みが完了しました！")

    # デバッグ用: ページソースの先頭2000文字を表示
    print("デバッグ: ページのHTMLソース:")
    print(driver.page_source[:2000])

    # スクロールして全てのフォロワー要素をロード
    print("スクロール処理開始...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    print("スクロール処理完了")

    # 要素の確認
    follower_elements = driver.find_elements(By.XPATH, "//div[@data-testid='UserCell']//a")
    print(f"デバッグ: 見つかった要素数: {len(follower_elements)}")

    # フォロワーのリンクを取得
    for element in follower_elements:
        user_link = element.get_attribute("href")
        print("取得したリンク:", user_link)
        followers.append(user_link.split('/')[-1])

finally:
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)