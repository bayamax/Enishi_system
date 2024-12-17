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

# ChromeDriverのサービス指定
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

# Twitterのフォロワーページにアクセス
user_id = "1782363447843491840"
url = f"https://twitter.com/{user_id}/followers"
driver.get(url)

followers = []

try:
    # フォロワーリストが表示されるまで待機
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//div[@data-testid='UserCell']//a"))
    )

    # スクロールして全てのフォロワーを読み込む
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # フォロワーのユーザー名を抽出
    follower_elements = driver.find_elements(By.XPATH, "//div[@data-testid='UserCell']//a")
    for element in follower_elements:
        user_link = element.get_attribute("href")
        followers.append(user_link.split('/')[-1])

finally:
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)