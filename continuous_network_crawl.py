from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os

# ChromeDriverのパス設定
CHROME_DRIVER_PATH = "/opt/homebrew/bin/chromedriver"

# 手動で取得したクッキー情報
cookies = {
    "auth_token": "22c906d59549e6170f01145240c86e20be68e07a",
    "ct0": "e2c8a89989ae0b594093a03b70d132d59bdc575c3d80d9436337d1437036b95",
    "twid": "u%3D1782363447843491840"
}


# Seleniumの初期設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

# 探索済みアカウントを保存するファイル
explored_accounts_file = "explored_accounts.txt"


def load_explored_accounts():
    """
    ファイルから探索済みアカウント名を読み込む。
    """
    if not os.path.exists(explored_accounts_file):
        return set()

    with open(explored_accounts_file, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file.readlines())


def save_explored_account(account_name):
    """
    探索済みアカウント名をファイルに追加する。
    """
    with open(explored_accounts_file, "a", encoding="utf-8") as file:
        file.write(account_name + "\n")


def scroll_until_loaded(driver, max_scrolls=100):
    """
    ページの高さが変化しなくなるまでスクロールを繰り返す。
    max_scrolls: 最大スクロール回数（無限ループ防止用）
    """
    scrolls = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("ページの高さが変化しないため、スクロールを停止します。")
            break
        last_height = new_height
        scrolls += 1
    print(f"最大スクロール回数: {scrolls}/{max_scrolls}")


def fetch_following(account_name):
    """
    指定されたアカウントのフォローリストを取得し、ファイルに保存する。
    """
    explored_accounts = load_explored_accounts()

    if account_name in explored_accounts:
        print(f"すでに探索済みのアカウント: {account_name}")
        return []

    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # フォローリストページのURL
        url = f"https://x.com/{account_name}/following"
        print(f"フォローしている人のページにアクセスします: {url}")

        driver.get("https://x.com")
        time.sleep(10)

        # クッキーを追加
        for cookie in cookies:
            driver.add_cookie(cookie)

        # フォローリストページにアクセス
        driver.get(url)
        time.sleep(10)

        # ページをスクロールして全てのフォローをロード
        scroll_until_loaded(driver, max_scrolls=100)

        # ページのフォロー情報を取得
        following = []
        index = 1

        while True:
            try:
                xpath = f'//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[{index}]/div/div/button/div/div[2]/div[1]/div[1]/div/div[2]/div[1]/a/div/div/span'
                username_element = driver.find_element(By.XPATH, xpath)
                username = username_element.text
                following.append(username)
                print(f"取得したユーザーネーム: {username}")
                index += 1
                time.sleep(1)
            except Exception:
                if not following:
                    print(f"{account_name} のフォローリストが空です。")
                else:
                    print("全てのフォローを取得しました。")
                break

        filename = f"{account_name}_following.txt"
        with open(filename, "w", encoding="utf-8") as file:
            for user in following:
                file.write(user + "\n")
        print(f"取得したフォローのユーザーネームを{filename}に保存しました。")

        # 探索済みアカウントとして記録
        save_explored_account(account_name)

        return following

    finally:
        driver.quit()


def recursive_fetch(account_name, depth=1, max_depth=3):
    """
    再帰的にフォローリストを取得する。
    depth: 現在の探索の深さ
    max_depth: 最大の探索の深さ
    """
    explored_accounts = load_explored_accounts()

    if account_name in explored_accounts:
        print(f"すでに探索済みのアカウント: {account_name}")
        return

    if depth > max_depth:
        print(f"最大深度に到達しました: {account_name}")
        return

    print(f"現在のアカウント: {account_name}, 深度: {depth}")

    following_list = fetch_following(account_name)

    if not following_list:
        print(f"{account_name} のフォローリストが空のため次へ進みます。")
        return

    for following_account in following_list:
        recursive_fetch(following_account, depth=depth + 1, max_depth=max_depth)


if __name__ == "__main__":
    # 初期アカウント名を指定
    start_account = "cloudproject_ad"
    recursive_fetch(start_account)