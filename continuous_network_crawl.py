from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os

# ChromeDriverのパス設定
CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"

# 手動で取得したクッキー情報
cookies = [
    {"name": "auth_token", "value": "22c906d59549e6170f01145240c86e20be68e07a"},
    {"name": "ct0", "value": "e2c8a89989ae0b594093a03b70d132d59bdc575c3d80d9436337d1437036b95"},
    {"name": "twid", "value": "u%3D1782363447843491840"}
]

START_ACCOUNT = "@setagayaj_swsc" 

# Seleniumの初期設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--display=:99")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

# 探索済みアカウントを保存するファイル
explored_accounts_file = "explored_accounts.txt"



def load_explored_accounts():
    """
    ファイルから探索済みアカウント名を読み込む。
    """
    if not os.path.exists(EXPLORED_ACCOUNTS_FILE):
        return set()

    with open(EXPLORED_ACCOUNTS_FILE, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file.readlines())

def save_explored_account(account_name):
    """
    探索済みアカウント名をファイルに追加する。
    """
    with open(EXPLORED_ACCOUNTS_FILE, "a", encoding="utf-8") as file:
        file.write(account_name + "\n")

def save_following_list(account_name, following_list):
    """
    フォローリストをアカウント名に基づいたファイルに保存する。
    """
    sanitized_account_name = account_name.replace("@", "").replace("/", "_")  # ファイル名に使える形式に修正
    filename = f"{sanitized_account_name}_following.txt"

    with open(filename, "w", encoding="utf-8") as file:
        for following_account in following_list:
            file.write(following_account + "\n")

def twitter_login(driver):
    """
    Twitterに自動ログインする（電話番号対応版）。
    """
    driver.get("https://x.com/login")
    time.sleep(10)

    try:
        username_input = driver.find_element(By.NAME, "text")
        username_input.send_keys(TWITTER_USERNAME)
        username_input.send_keys(Keys.RETURN)
        time.sleep(5)

        try:
            phone_input = driver.find_element(By.NAME, "text")
            phone_input.send_keys(TWITTER_PHONE)
            phone_input.send_keys(Keys.RETURN)
            time.sleep(5)
        except Exception:
            print("電話番号の入力が求められませんでした。スキップします。")

        password_input = driver.find_element(By.NAME, "password")
        password_input.send_keys(TWITTER_PASSWORD)
        password_input.send_keys(Keys.RETURN)
        time.sleep(10)

        print("Twitterにログインしました。")
    except Exception as e:
        print(f"ログイン中にエラーが発生しました: {e}")

def navigate_to_search_page(driver):
    """
    検索ページに移動する。
    """
    search_page_xpath = '//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[1]/div[2]/nav/a[2]/div'
    search_page_link = driver.find_element(By.XPATH, search_page_xpath)
    search_page_link.click()
    time.sleep(5)

def search_and_open_account(driver, account_name):
    """
    指定のアカウントを検索し、そのアカウントページを開く。
    """
    search_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Search"]'))
    )
    search_input.clear()
    search_input.send_keys(account_name)
    search_input.send_keys(Keys.RETURN)
    time.sleep(5)

    first_account_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/section/div/div/div[3]/div/div/button/div/div[2]/div[1]/div[1]/div/div[1]/a/div/div[1]'
    first_account_link = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, first_account_xpath))
    )
    first_account_link.click()
    time.sleep(5)

def navigate_to_following_list(driver):
    """
    アカウントページからフォローリストページへ移動する。
    """
    following_link_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div/div/div/div[5]/div[1]/a'
    following_link = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, following_link_xpath))
    )
    following_link.click()
    time.sleep(5)

def fetch_following_accounts(driver):
    """
    フォローリストからアカウント名を取得する。
    """
    scroll_pause_time = 2
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    following_accounts = []

    while True:
        accounts = driver.find_elements(By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div/div/div/button/div/div[2]/div[1]/div[1]/div/div[2]/div/a/div/div/span')
        for account in accounts:
            account_name = account.text.strip()
            if account_name.startswith("@") and account_name not in following_accounts:
                following_accounts.append(account_name)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_scroll_height = driver.execute_script("return document.body.scrollHeight")
        if new_scroll_height == scroll_height:
            break
        scroll_height = new_scroll_height

    return following_accounts

def recursive_fetch(driver, account_name, depth=1, max_depth=3, wait_time=6000):
    """
    再帰的にフォローリストを探索する。
    """
    if depth > max_depth:
        return

    explored_accounts = load_explored_accounts()
    if account_name in explored_accounts:
        return

    save_explored_account(account_name)

    try:
        navigate_to_search_page(driver)
        search_and_open_account(driver, account_name)
        navigate_to_following_list(driver)

        following_accounts = fetch_following_accounts(driver)

        # フォローリストが空の場合
        if not following_accounts:
            print(f"{account_name} はフォロワーがいません。スキップします。")
            time.sleep(wait_time)  # 一定時間停止
            return

        save_following_list(account_name, following_accounts)

        for following_account in following_accounts:
            if following_account not in explored_accounts:
                recursive_fetch(driver, following_account, depth=depth + 1, max_depth=max_depth, wait_time=wait_time)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        time.sleep(wait_time)  # エラー時にも一定時間停止してリトライ防止

def main():
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        twitter_login(driver)
        recursive_fetch(driver, START_ACCOUNT)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()