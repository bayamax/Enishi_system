from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ChromeDriverのパス設定
CHROME_DRIVER_PATH = "/opt/homebrew/bin/chromedriver"

# Twitterのログイン情報
TWITTER_USERNAME = "cloudproject_ad"  # アカウント名
TWITTER_PHONE = "09067372699"  # 電話番号
TWITTER_PASSWORD = "F=M*a+C*v^2"  # パスワード

# スタートとなるアカウント


START_ACCOUNT = "@Lightup_online"  # スタートとなるアカウント名

# 探索済みアカウントを保存するファイル
EXPLORED_ACCOUNTS_FILE = "explored_accounts.txt"

# Seleniumの初期設定
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)


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
        except NoSuchElementException:
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
    global retry_wait_iterations  # グローバル変数を明示的に使用
    try:
        # 検索入力欄を見つけて検索
        search_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Search"]'))
        )
        search_input.clear()
        search_input.send_keys(account_name)
        search_input.send_keys(Keys.RETURN)
        time.sleep(7)

        # People タブに移動
        people_tab_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div[2]/nav/div/div[2]/div/div[3]/a'
        people_tab = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, people_tab_xpath))
        )
        people_tab.click()
        time.sleep(5)

        # 検索結果の最初のアカウントをクリック
        first_account_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/section/div/div/div[1]/div/div/button/div/div[2]/div[1]/div[1]/div/div[1]/a/div/div[1]'
        first_account_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, first_account_xpath))
        )
        first_account_link.click()
        time.sleep(5)

        # アカウントIDを確認
        current_account_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div/div/div/div[2]/div/div/div/div[2]/div/div/div/span'
        current_account_id = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, current_account_xpath))
        ).text.strip()
        if current_account_id != account_name:
            print(f"期待されたアカウント {account_name} ではなく、現在のアカウント {current_account_id} です。戻ります。")
            click_back_button(driver)  # 戻るボタンを押して次のアカウントを探索
            return False
        retry_wait_iterations = 3  # 待機繰り返し回数をリセット
        print(f"正しいアカウント {account_name} に移動しました。")
        return True

    except TimeoutException:
        print(f"アカウント {account_name} の検索結果が見つかりませんでした。スキップします。")
        handle_incremental_wait()
        handle_timeout_and_retry(driver, account_name)

def handle_incremental_wait():
    """
    待機時間を繰り返し処理で実現する。
    """
    global retry_wait_iterations
    max_iterations = 384  # 最大繰り返し数（例: 6回 * 30秒 = 最大180秒）
    sleep_time = 120  # 1回あたりの待機時間（秒）

    print(f"{sleep_time * retry_wait_iterations} 秒待機します...")
    for i in range(retry_wait_iterations):
        time.sleep(sleep_time)  # 固定間隔で待機
        print(f"待機中: {i + 1}/{retry_wait_iterations} 回目完了。")

    # 待機回数を増加（最大値に達するまで）
    if retry_wait_iterations < max_iterations:
        retry_wait_iterations = retry_wait_iterations*2


def click_back_button(driver):
    """
    戻るボタンをクリックする。
    """
    back_button_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div[1]/div/div/div/div/div[1]/button'
    try:
        back_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, back_button_xpath))
        )
        back_button.click()
        time.sleep(3)
    except TimeoutException:
        print("戻るボタンが見つかりませんでした。スキップします。")

def handle_timeout_and_retry(driver, failed_account):
    """
    TimeoutException が発生した際に再ログインと次のアカウント探索を実行する。
    """
    try:
        print("再ログインを試みます...")
        twitter_login(driver)  # 再ログイン
        navigate_to_search_page(driver)  # 検索ページに移動

        # 次のアカウントを探索
        explored_accounts = load_explored_accounts()
        next_account = next(iter(explored_accounts - {failed_account}), None)  # 次の未探索アカウントを取得
        if next_account:
            print(f"次のアカウント {next_account} を探索します。")
            recursive_fetch(driver, next_account, depth=1, max_depth=3)
        else:
            print("探索可能なアカウントがありません。処理を終了します。")
    except Exception as e:
        print(f"再ログインまたは次のアカウント探索中にエラーが発生しました: {e}")

# グローバル変数として待機時間を管理
retry_wait_iterations = 3  # 初期繰り返し回数を1回に設定


def navigate_to_following_list(driver):
    """
    アカウントページからフォローリストページへ移動する。
    """
    following_link_xpath = '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div/div/div/div[5]/div[1]/a'
    try:
        following_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, following_link_xpath))
        )
        following_link.click()
        time.sleep(5)
    except TimeoutException:
        print("フォローリストが見つかりませんでした。スキップします。")


def fetch_following_accounts(driver):
    """
    フォローリストからアカウント名を取得する。
    """
    scroll_pause_time = 3
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

        if new_scroll_height == driver.execute_script("return document.body.scrollHeight"):
            break

    return following_accounts



def recursive_fetch(driver, account_name, depth=1, max_depth=3, retries=0, max_retries=3):
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
            print(f"{account_name} のフォローリストは空です。探索を次のアカウントから再開します。")
            retries += 1
            if retries <= max_retries:
                # 次のアカウントを取得
                next_account = next(iter(explored_accounts - {account_name}), None)  # 次の未探索アカウントを取得
                if next_account:
                    print(f"次のアカウント {next_account} から再開します。")
                    recursive_fetch(driver, next_account, depth=1, max_depth=max_depth, retries=retries, max_retries=max_retries)
                else:
                    print("探索可能なアカウントがありません。処理を終了します。")
            else:
                print(f"最大リトライ回数に達しました。{account_name} の処理を完全にスキップします。")
            return

        save_following_list(account_name, following_accounts)

        for following_account in following_accounts:
            if following_account not in explored_accounts:
                recursive_fetch(driver, following_account, depth=depth + 1, max_depth=max_depth)
    except TimeoutException:
        handle_timeout_and_retry(driver, account_name)
    except Exception as e:
        print(f"フォローリストを取得中にエラーが発生しました: {e}")



def main():
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        twitter_login(driver)
        recursive_fetch(driver, START_ACCOUNT)
    finally:
        driver.quit()

def remove_top_account_and_restart():
    """
    explored_accounts.txt の一番上のIDを削除し、それを START_ACCOUNT としてプログラムを再スタートする。
    """
    if not os.path.exists(EXPLORED_ACCOUNTS_FILE):
        print("探索可能なアカウントがありません。プログラムを終了します。")
        return False

    with open(EXPLORED_ACCOUNTS_FILE, "r", encoding="utf-8") as file:
        accounts = file.readlines()

    if not accounts:
        print("探索可能なアカウントがありません。プログラムを終了します。")
        return False

    # 一番上のアカウントを取得して削除
    next_start_account = accounts.pop(0).strip()
    with open(EXPLORED_ACCOUNTS_FILE, "w", encoding="utf-8") as file:
        file.writelines(accounts)

    print(f"次のアカウント {next_start_account} から再スタートします。")
    run_program(next_start_account)
    return True


def run_program(start_account):
    """
    プログラムを指定された START_ACCOUNT で開始する。
    """
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        twitter_login(driver)
        recursive_fetch(driver, start_account)
    finally:
        driver.quit()
        # プログラム終了時に次のアカウントを取得して再スタート
        remove_top_account_and_restart()

def main():
    """
    メイン関数。
    最初の START_ACCOUNT を指定してプログラムを開始する。
    """
    run_program(START_ACCOUNT)


if __name__ == "__main__":
    main()