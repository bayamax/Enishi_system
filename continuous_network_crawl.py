from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# Chromeオプション設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# ヘッドレスモードの有効/無効を切り替え
# コメントアウトすればヘッドレスモードが無効になります
# chrome_options.add_argument("--headless")  # 画面表示なし

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
    # ページソースを出力（デバッグ用）
    print("ページソースの取得開始...")
    print(driver.page_source[:2000])  # 最初の2000文字を表示
    print("ページソースの取得完了")

    # フォロワーリストが表示されるまで待機
    print("要素の待機中...")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[@data-testid='UserCell']//a"))
    )
    print("要素が見つかりました！")

    # スクロールして全てのフォロワーを読み込む
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

    # フォロワーのユーザー名を抽出
    print("フォロワー情報を取得中...")
    follower_elements = driver.find_elements(By.XPATH, "//div[@data-testid='UserCell']//a")
    if not follower_elements:
        print("警告: フォロワー要素が見つかりません。XPathを確認してください。")
    else:
        print(f"見つかった要素数: {len(follower_elements)}")
        for element in follower_elements:
            user_link = element.get_attribute("href")
            print("取得したリンク:", user_link)
            followers.append(user_link.split('/')[-1])

finally:
    print("ブラウザを閉じます...")
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)