import os
import glob
import zipfile
import datetime
import pandas as pd
import numpy as np
import csv
import keras
keras.__version__
from keras.models import load_model
import itertools

# jsonlファイルを読み込み、numpy配列を返す
def load_json_line(file_path):
    # pandasだと一部のjsonlファイルでエラーが出るためnumpyに変換する
    # np.arrayでは一部のファイルでエラーが出るのでnp.asarrayを使う
    return np.asarray(pd.read_json(file_path, orient="records", lines=True, encoding="utf8", dtype="object"))

# CSVファイルを読み込み、リストを返す
def load_csv(file_path):
    csv_list = []
    # 空白行を無くすためにnewline=""を引数として渡す
    with open(file_path, "r", encoding="utf8", newline="") as f:
        csv_data = csv.reader(f)
        for current_line in csv_data:
            csv_list.append(str(current_line[0]))
    return csv_list
    del csv_list
    del csv_data

# CSVファイルに保存する
def save_csv(file_path, list_data):
    with open(file_path, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        for i in list_data:
            writer.writerow(i)

# 正解ラベルのインデックスを返す
def get_answer_label_index(answer_labels, all_answer_labels):
    index = -1
    for label in all_answer_labels:
        if label in answer_labels:
            index = all_answer_labels.index(label)
    
    return index

# 文字列をコードポイントを返す
def count_code_point(training_data):
    global skip_string
    # valueがnullの時に備えてstr型に変換する
    training_data = str(training_data)
    # Unicodeのコードポイントをインデックスとする配列を用意する
    words = []
    # 一文字ずつUnicodeのコードポイントに変換する
    for word_index in range(len(training_data)):
        temp_str = ord(training_data[word_index])
        if temp_str >= 100000:
            skip_string.append(temp_str)
            continue
        words.append(temp_str)
    return words

# ベクトル化を行う
def vectorize_sequences(sequences, dimension=100000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# □■□■□■□■□■□■□■□■ 設定 □■□■□■□■□■□■□■□■

# 検出する正解ラベルの一覧
all_answer_labels = ["VOCALOID", "演奏してみた", "歌ってみた", "踊ってみた"]

# jsonlファイルの訓練データのキー
training_datas_index = 4 # 動画のタイトル
# training_datas_key = "title" # 動画のタイトル

# jsonlファイルの正解ラベルのキー
answer_labels_index = 7  # 動画のタグ
# answer_labels_key = "tags"  # 動画のタグ

# zipファイルのファイルパス
zip_directory = ".\\zipfiles\\*\\*.zip"

# zipファイルの展開先のディレクトリ
unfolded_zip_directory = ".\\training_data"

# jsonlファイルのファイルパス
jsonl_directory = ".\\training_data\\*.jsonl"

# すべてのJSONLファイルのファイルパスを格納するCSVファイルのファイル名
all_jsonl_file_path_csv_name = "all_jsonl_file_path.csv"

# 読み込むモデルのファイル名
model_file_name = "title_to_tag_model.h5"

# 初期化するときに削除するファイル名一覧
delete_name_list = [all_jsonl_file_path_csv_name]

# □■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■



# 質問の回答を記録する
question01 = input("初期化しますか？ y(Yes) or n(No)>")
while question01 != "y" and question01 != "n":
    question01 = input("初期化しますか？ y(Yes) or n(No)>")

question02 = input("zipファイルを展開しますか？ y(Yes) or n(No)>")
while question02 != "y" and question02 != "n":
    question02 = input("zipファイルを展開しますか？ y(Yes) or n(No)>")

# 最後に経過時間を表示するために開始時刻を記録する
start_time = datetime.datetime.now()

if question01 == "y":
    print(str(datetime.datetime.now()), "初期化しています")
    for name in delete_name_list:
        try:
            os.remove(name)
            print(str(datetime.datetime.now()), name + "を削除しました")
        except:
            print(str(datetime.datetime.now()), name + "を削除できませんでした")

# 何件目のzipファイルを読み込んでいるか
count_current_zip = 0
# 正常に読み込めたzipファイルの件数
count_success_zip = 0
# 読み込めなかったzipファイルの件数
count_error_zip = 0

if question02 == "y":
    print(str(datetime.datetime.now()), "zipファイルを検索しています")
    # zipファイルのファイルパスの一覧を取得し順番に読み込む
    for zip_file_path in sorted(glob.glob(zip_directory)):
        # 何件目のzipファイルを読み込んでいるかインクリメントする
        count_current_zip += 1
        print(str(datetime.datetime.now()), zip_file_path + "を展開しています(" + str(count_current_zip) + "件目) " + str(count_success_zip) + "件成功 " + str(count_error_zip) + "件失敗")
        # try:
        # zipファイルを展開する
        with zipfile.ZipFile(zip_file_path) as sm_zip:
            sm_zip.extractall(unfolded_zip_directory)
        # 正常に読み込めたzipファイルの件数をインクリメントする
        count_success_zip += 1
        # except:
            # 読み込めなかったzipファイルの件数をインクリメントする
            # count_error_zip += 1

# 何件目のjsonlファイルを読み込んでいるか
count_current_jsonl = 0
# 処理済みのファイルパスの一覧を保存するリスト
completed_jsonl_files_info = []
# 読み込めなかったjsonlファイルを記録する
failed_jsonl_list = []
# 読み込めなかった文字列を記録する
skip_string = []
# 処理済みの合計ファイルサイズ
all_file_size = 0

# 読み込んだタイトルの件数
loaded_title_count = 0
# 正解した件数
correct_answer_count = 0

# 学習済みのデータと分類器が存在するか判定する
try:
    print(str(datetime.datetime.now()), model_file_name + "をロードしています")
    # 学習済みのデータと分類器を読み込む
    model = load_model(model_file_name)
except:
    print(str(datetime.datetime.now()), model_file_name + "をロードできませんでした")

try:
    print(str(datetime.datetime.now()), all_jsonl_file_path_csv_name + "をロードしています")
    all_jsonl_file_path_list = load_csv(all_jsonl_file_path_csv_name)
except:
    print(str(datetime.datetime.now()), all_jsonl_file_path_csv_name + "をロードできませんでした")
    print(str(datetime.datetime.now()), "jsonlファイルを検索しています")
    # jsonlファイルのファイルパスの一覧を取得し順番に読み込む
    all_jsonl_file_path_list = []
    for jsonl_file_path in sorted(list(glob.glob(jsonl_directory))):
        all_jsonl_file_path_list.append(jsonl_file_path)
    print(str(datetime.datetime.now()), all_jsonl_file_path_csv_name + "をセーブしています")
    save_csv(all_jsonl_file_path_csv_name, all_jsonl_file_path_list)

for jsonl_file_path in all_jsonl_file_path_list:
    all_file_size += os.path.getsize(jsonl_file_path)
    try:
        # 何件目のjsonlファイルを読み込んでいるかインクリメントする
        count_current_jsonl += 1
        print(str(datetime.datetime.now()), jsonl_file_path + "をロードしています(" + str(count_current_jsonl) + "件目)")
        # jsonlファイルを1つ読み込む
        jsonl_np = load_json_line(jsonl_file_path)
        # sm_list[:, answer_labels_index]とすると一部のjsonlファイルでエラーが出るため、やむを得ずfor文を使う
        print(str(datetime.datetime.now()), "訓練データをコードポイントに変換しています")
        for i in range(len(jsonl_np)):
            answer_label_index = get_answer_label_index(jsonl_np[i][answer_labels_index], all_answer_labels)
            
            # 対象の正解ラベルが含まれていない場合はスキップする
            if answer_label_index == -1:
                continue
            
            # 予測する。コードポイントに変換しそれをベクトル化したものをpredict関数に渡す
            predict_list = model.predict(vectorize_sequences([count_code_point(str(jsonl_np[i][training_datas_index]))]))
            # NumPy配列をリストに直してそれを1次元配列に直す
            predict_list = list(itertools.chain.from_iterable(predict_list.tolist()))
            # 確率が高い正解ラベルのインデックスを求める
            predict_result = predict_list.index(max(predict_list))
            
            # 予測が合っているか確かめる
            loaded_title_count += 1
            if all_answer_labels[predict_result] in jsonl_np[i][answer_labels_index]:
                correct_answer_count += 1
            
            print(str(datetime.datetime.now()), "今までに読み込めなかったjsonlファイルの件数 : " + str(len(failed_jsonl_list)) + "件")
            print(str(datetime.datetime.now()), "今までに読み込めなかった文字数 : " + str(len(skip_string)) + "件")
            print(str(datetime.datetime.now()), "経過時間 : " + str(datetime.datetime.now() - start_time))
            print(str(datetime.datetime.now()), "処理済みの合計ファイルサイズ : " + str(all_file_size / (1024 ** 3)) + "GB")
            print(str(datetime.datetime.now()), "読み込んだタイトルの件数 : " + str(loaded_title_count) + "件")
            print(str(datetime.datetime.now()), "正解した件数 : " + str(correct_answer_count) + "件")
            print(str(datetime.datetime.now()), "現時点での正解率 : " + str(correct_answer_count / loaded_title_count * 100) + "件")
        
        # 処理済みのファイルパスを記録する
        completed_jsonl_files_info.append(jsonl_file_path)
        
    except:
        # 読み込めなかったjsonlファイルの件数をインクリメントする
        failed_jsonl_list.append(jsonl_file_path)

print(str(datetime.datetime.now()), "☆☆☆　結果報告　☆☆☆")
try:
    print(str(datetime.datetime.now()), "正常に展開できたzipファイルの件数 : " + str(count_success_zip))
except:
    print(str(datetime.datetime.now()), "正常に展開できたzipファイルの件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "展開できなかったzipファイルの件数 : " + str(count_error_zip))
except:
    print(str(datetime.datetime.now()), "展開できなかったzipファイルの件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "正常に読み込めたjsonlファイルの件数 : " + str(len(completed_jsonl_files_info)) + " 件")
except:
    print(str(datetime.datetime.now()), "正常に読み込めたjsonlファイルの件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "読み込めなかったjsonlファイルの件数 : " + str(len(failed_jsonl_list)))
except:
    print(str(datetime.datetime.now()), "読み込めなかったjsonlファイルの件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "今までに読み込めなかった文字数 : " + str(len(skip_string)) + " 件")
except:
    print(str(datetime.datetime.now()), "今までに読み込めなかった文字数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "経過時間 : " + str(datetime.datetime.now() - start_time))
except:
    print(str(datetime.datetime.now()), "経過時間 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "処理済みの合計ファイルサイズ : " + str(all_file_size / (1024 ** 3)) + " GB")
except:
    print(str(datetime.datetime.now()), "処理済みの合計ファイルサイズ : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "読み込んだタイトルの件数 : " + str(loaded_title_count) + "件")
except:
    print(str(datetime.datetime.now()), "読み込んだタイトルの件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "正解した件数 : " + str(correct_answer_count) + "件")
except:
    print(str(datetime.datetime.now()), "正解した件数 : " + "表示できません")
try:
    print(str(datetime.datetime.now()), "現時点での正解率 : " + str(correct_answer_count / loaded_title_count * 100) + "件")
except:
    print(str(datetime.datetime.now()), "現時点での正解率 : " + "表示できません")
