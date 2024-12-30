import os
import csv

data_dir = "/Users/oobayashikoushin/Enishi_system/data/"
output_csv = "/Users/oobayashikoushin/Enishi_system/edges.csv"

def clean_handle(raw_handle: str) -> str:
    """
    ファイル中の行から読み込んだハンドル名(@付きなど)を整形。
    先頭の'@'などを削除し、余分な空白も除去。
    """
    handle = raw_handle.strip()
    # 先頭の@のみ削除し、_は削除しない
    if handle.startswith('@'):
        handle = handle[1:]
    return handle

def extract_source_from_filename(filename: str) -> str:
    """
    filename例: '_9743_shukatsu__following.txt' or '@hogehoge_following.txt'
    -> 拡張子 '_following.txt' を除去 -> '_9743_shukatsu_' や '@hogehoge'
    -> 先頭の '@' だけを除去, 先頭の '_' は残す
    """
    # '_following.txt' を除去
    suffix = "_following.txt"
    if filename.endswith(suffix):
        base = filename[:-len(suffix)]
    else:
        base = filename  # 予想外の場合はそのまま
    
    # 先頭の '@' のみ除去 (lstrip('@')), '_'は残す
    # 例: '@hogehoge' -> 'hogehoge', '_9743_shukatsu_' -> '_9743_shukatsu_'
    base = base.lstrip('@')
    
    return base

def main():
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["source", "target"])  # ヘッダー行

        for filename in os.listdir(data_dir):
            if not filename.endswith("_following.txt"):
                continue  # '_following.txt'以外は無視
            
            source_account = extract_source_from_filename(filename)
            input_path = os.path.join(data_dir, filename)
            
            with open(input_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    handle = line.strip()
                    if not handle:
                        continue  # 空行スキップ
                    target_account = clean_handle(handle)
                    
                    # CSVに (source_account, target_account)を書き込む
                    writer.writerow([source_account, target_account])
    
    print(f"完了: {output_csv} にエッジリストを書き出しました。")

if __name__ == "__main__":
    main()