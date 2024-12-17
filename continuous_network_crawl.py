from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time

# Chrome WebDriverのパスを指定
driver_path = '/usr/local/bin/chromedriver'
service = Service(driver_path)

# WebDriverの初期化
driver = webdriver.Chrome(service=service)

# ユーザーページにアクセス
user_id = "1782363447843491840"
url = f"https://twitter.com/{user_id}/followers"
driver.get(url)

followers = []

try:
    # フォロワーリストが表示されるまで待機
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='UserCell']"))
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
    follower_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='UserCell']")
    for element in follower_elements:
        user_link = element.find_element(By.CSS_SELECTOR, "a[role='link']").get_attribute('href')
        followers.append(user_link.split('/')[-1])

finally:
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)