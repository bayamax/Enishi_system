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

# ChromeDriverのサービス指定
service = Service('/usr/local/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

# Twitterログイン情報
USERNAME = "cloudproject_ad"
PASSWORD = "F=M*a+C*v^2"

try:
    # Twitterログインページにアクセス
    driver.get("https://twitter.com/login")
    print("Twitterのログインページにアクセスしました。")

    # ユーザー名の入力
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.NAME, "text"))
    ).send_keys(USERNAME)

    driver.find_element(By.XPATH, "//span[text()='Next']").click()

    # パスワードの入力
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.NAME, "password"))
    ).send_keys(PASSWORD)

    driver.find_element(By.XPATH, "//span[text()='Log in']").click()
    print("Twitterにログインしました。")

    # フォロワーページにアクセス
    user_id = "1782363447843491840"
    url = f"https://twitter.com/{user_id}/followers"
    driver.get(url)
    print("フォロワーページにアクセスしました。")

    # ページ全体の読み込みを待機
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    # フォロワー要素を取得
    follower_elements = driver.find_elements(By.XPATH, "//div[@data-testid='UserCell']//a")
    print(f"見つかったフォロワー要素数: {len(follower_elements)}")

    # フォロワーのリンクを取得
    followers = []
    for element in follower_elements:
        user_link = element.get_attribute("href")
        print("取得したリンク:", user_link)
        followers.append(user_link.split('/')[-1])

finally:
    driver.quit()

print(f"フォロワー数: {len(followers)}")
print(followers)