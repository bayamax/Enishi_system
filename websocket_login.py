import websocket
import json

# WebSocket URL（curl http://127.0.0.1:9222/json で取得したもの）
websocket_url = "ws://127.0.0.1:9222/devtools/page/579D5F5F5ABB4B8721A1808ECA5B3D5A"

# WebSocket接続を確立
ws = websocket.create_connection(websocket_url)

# ユーザー名を入力
username_script = """
document.querySelector('input[name="text"]').value = 'cloudproject_ad';
"""
ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": username_script}}))

# パスワードを入力
password_script = """
document.querySelector('input[name="password"]').value = 'your_password';
"""
ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {"expression": password_script}}))

# ログインボタンをクリック
login_script = """
document.querySelector('div[data-testid="LoginForm_Login_Button"]').click();
"""
ws.send(json.dumps({"id": 3, "method": "Runtime.evaluate", "params": {"expression": login_script}}))

# WebSocketを閉じる
ws.close()
