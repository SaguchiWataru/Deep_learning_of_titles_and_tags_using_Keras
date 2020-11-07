import os
import glob
import zipfile
import datetime
import pandas as pd
import numpy as np
import csv
import keras
keras.__version__
import matplotlib.pyplot as plt
# from keras.utils.np_utils import to_categorical  # 既存の関数(to_categorical)を使う方法と、自作の関数(to_one_hot)を使う方法の2種類がありますが、今回は仕組みを理解する目的で自作の関数(to_one_hot)を使用するため、無効化します
from keras import models
from keras import layers
from keras.models import load_model

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

def to_one_hot(labels, dimension=128):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, int(label)] = 1.
    return results



# □■□■□■□■□■□■□■□■ 設定 □■□■□■□■□■□■□■□■

# 検出する正解ラベルの一覧
all_answer_labels = ["アニメ", "ゲーム", "エンタメ", "音楽", "スポーツ", "歌ってみた", "踊ってみた", "演奏してみた", "描いてみた", "ニコニコ技術部", "アイドルマスター", "東方", "VOCALOID", "例のアレ", "生放送", "政治", "ニコニコ動画講座", "静画", "動物", "料理", "日記", "自然", "科学", "歴史", "ラジオ"]

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

# 学習が完了したJSONLファイルのファイルパスを格納するCSVファイルのファイル名
completed_jsonl_files_csv_name = "completed_jsonl_files_list.csv"

# 読み込みに失敗したjsonlファイルのファイル名の一覧を保存するCSVファイルのファイル名
failed_jsonl_files_csv_name = "failed_jsonl_files_list.csv"

# 出力するモデルのファイル名
model_file_name = "title_to_tag_model.h5"

# 損失関数のグラフを保存するファイルパス
graph_directory = ".\\graphs\\graph_" + str(datetime.datetime.now()) + "_"
graph_directory = graph_directory.replace("-", "_")
graph_directory = graph_directory.replace(":", "_")
graph_directory = graph_directory.replace(" ", "_")

# インクリメンタル学習を行う間隔
incremental_learning_interval = 13440 # 使用しているパソコンでは16800まで正常に動作しましたが、余裕をもって8割の13440にしました

# 保存を行う間隔(jsonlファイル数)
save_jsonl_interval = 100  # 今回の場合は、jsonlファイルが3411件あるため、作業完了までに35回上書き保存されます

# モデルを定義
model = models.Sequential()
model.add(layers.Dense(1000, activation='relu', input_shape=(100000, )))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(128, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 1回の学習に用いるデータの数
batch_size_num = incremental_learning_interval  # 全てのデータを学習させる。インクリメンタル学習を行う間隔と同じ

# 同じ内容を何回学習させるか 15回以降は学習に効果が見られないため15回で終了 (過学習を避けるため)
epochs_num = 15

# インクリメンタル学習を行う度に人が手動でテストを行う ※自動で継続的に学習させる場合はFalse
test_flag = False

# 学習に使用したデータを最後に表示する ※メモリの使用量が極端に増えるので少量のテスト以外False
converted_backup = False

# 初期化するときに削除するファイル名一覧
delete_name_list = [all_jsonl_file_path_csv_name, completed_jsonl_files_csv_name, model_file_name, failed_jsonl_files_csv_name]

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
# 正常に読み込めたjsonlファイルの件数
count_success_jsonl = 0
# 読み込めなかったjsonlファイルの件数
count_error_jsonl = 0
# すべての正解ラベルから重複を取り除いた集合
all_answer_label_set = set()
# 定期的に経過時間をリセットし、経過時間を表示するために現在時刻を記録する
temp_time = datetime.datetime.now()
# 何件目のjsonlファイルを読み込んでいるか
count_current_jsonl = 0
# 正常に読み込めたjsonlファイルの件数
count_success_jsonl = 0
# 読み込めなかったjsonlファイルの件数
count_error_jsonl = 0

# すべての訓練データ
training_data_list = []
# 訓練データを可視化しやすくする
training_data_list_str = []
# すべての正解ラベル
answer_label_list = []
# 正解ラベルを可視化しやすくする
answer_label_list_str = []

# すべての訓練データ
training_data_list_old = []
# 訓練データを可視化しやすくする
training_data_list_str_old = []
# すべての正解ラベル
answer_label_list_old = []
# 正解ラベルを可視化しやすくする
answer_label_list_str_old = []

# 処理済みの合計ファイルサイズ
all_file_size = 0
# 定期的に経過時間をリセットし、経過時間を表示するために現在時刻を記録する
temp_time = datetime.datetime.now()
# 読み込めなかったjsonlファイルを記録する
failed_jsonl_list = []
# 読み込めなかった文字列を記録する
skip_string = []
# セーブのタイミングにインターバルを設けるために、学習した回数を記録する
learn_count = 0
# 損失関数のグラフを連番で保存する
jpg_count = 0

# 停電対策のために処理済みのjsonlファイルのパスの一覧が記録されているcsvファイルを検索する
try:
    print(str(datetime.datetime.now()), completed_jsonl_files_csv_name + "をロードしています")
    completed_jsonl_files_info = load_csv(completed_jsonl_files_csv_name)
except:
    # ファイルが存在しなかったからファイルを新規作成する
    print(str(datetime.datetime.now()), completed_jsonl_files_csv_name + "をロードできませんでした")
    # 処理済みのファイルパスの一覧を保存するリスト
    completed_jsonl_files_info = []

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
    learn_count += 1
    all_file_size += os.path.getsize(jsonl_file_path)
    # 停電対策のために処理済みのjsonlファイルであるか確認する
    if jsonl_file_path in completed_jsonl_files_info:
        continue
    else:
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
                answer_label_value = all_answer_labels[answer_label_index]
                if answer_label_index == -1:
                    continue
                training_data_list.append(count_code_point(str(jsonl_np[i][training_datas_index])))
                training_data_list_str.append(jsonl_np[i][training_datas_index])
                answer_label_list.append(answer_label_index)
                answer_label_list_str.append(answer_label_value)
                # メモリを節約するために途中でインクリメンタル学習を行う
                if incremental_learning_interval <= len(answer_label_list):
                    print("\n" + str(datetime.datetime.now()), "インクリメンタル学習をしています")
                    


                    # ベクトル化とカテゴリ化
                    x_train = vectorize_sequences(training_data_list)
                    one_hot_train_labels = to_one_hot(answer_label_list)
                    # one_hot_train_labels = to_categorical(answer_label_list)
                    
                    # 学習用と精度の計測用で半分に分ける (今回の場合は、全てのデータを学習させるため、一時的に無効にしています)
                    # slice_index = len(x_train) // 2
                    x_val = x_train
                    # x_val = x_train[:slice_index]
                    partial_x_train = x_train
                    # partial_x_train = x_train[slice_index:]
                    y_val = one_hot_train_labels
                    # y_val = one_hot_train_labels[:slice_index]
                    partial_y_train = one_hot_train_labels
                    # partial_y_train = one_hot_train_labels[slice_index:]
                    
                    # 学習させる
                    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = epochs_num,
                    batch_size = batch_size_num,
                    validation_data=(x_val, y_val)
                    )
                    
                    # 精度を可視化
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                    epochs = range(1, len(loss) + 1)
                    plt.plot(epochs, loss, 'bo', label='Training loss')
                    plt.plot(epochs, val_loss, 'b', label='Validation loss')
                    plt.title('Training and validation loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    jpg_count += 1
                    # 損失関数のグラフを保存する
                    plt.savefig(graph_directory + str(jpg_count).zfill(8) + ".jpg")
                    # リセットする
                    plt.figure()
                    
                    
                    
                    while test_flag == True:
                        test_training_data = str(input("テスト>"))
                        if test_training_data == "exit":
                            test_flag = False
                            continue
                        test_list =  model.predict(vectorize_sequences([count_code_point(test_training_data)]))
                        size = []
                        color_index = []
                        for i in test_list:
                            for i2, value in enumerate(i):
                                size.append(value)
                            size_max = max(size)
                            for i2, value in enumerate(i):
                                if size_max == value:
                                    color_index.append(i2)
                        for i in color_index:
                            print(all_answer_labels[i])
                    
                    # 学習に使用したデータのバックアップ
                    if converted_backup == True:
                        training_data_list_old += training_data_list
                        training_data_list_str_old += training_data_list_str
                        answer_label_list_old += answer_label_list
                        answer_label_list_str_old += answer_label_list_str
                    
                    # 初期化
                    training_data_list = []
                    training_data_list_str = []
                    answer_label_list = []
                    answer_label_list_str = []
            
            if learn_count % save_jsonl_interval == 0:
                print(str(datetime.datetime.now()), model_file_name + "をセーブしています")
                # 学習済みデータを保存する
                model.save(model_file_name)
                # 処理済みのファイルパスをcsvファイルに保存する
                save_csv(completed_jsonl_files_csv_name, completed_jsonl_files_info)
            
            # メモリを節約するために初期化する
            jsonl_np = []
            # 読み込めたjsonlファイルの件数をインクリメントする
            count_success_jsonl += 1
            # 処理済みのファイルパスを記録する
            completed_jsonl_files_info.append(jsonl_file_path)
            
            print(str(datetime.datetime.now()), "今までに読み込めなかったjsonlファイルの件数 : " + str(count_error_jsonl) + "件")
            print(str(datetime.datetime.now()), "今までに読み込めなかった文字数 : " + str(len(skip_string)) + "件")
            print(str(datetime.datetime.now()), "経過時間 : " + str(datetime.datetime.now() - start_time))
            print(str(datetime.datetime.now()), "処理済みの合計ファイルサイズ : " + str(all_file_size / (1024 ** 3)) + "GB")
            print(str(datetime.datetime.now()), "正解ラベルの繰越しの件数 : " + str(len(answer_label_list)) + "件")
        except:
            # 読み込めなかったjsonlファイルの件数をインクリメントする
            count_error_jsonl += 1
            failed_jsonl_list.append(jsonl_file_path)
            save_csv(failed_jsonl_files_csv_name, failed_jsonl_list)

print(str(datetime.datetime.now()), "インクリメンタル学習をしています")



# ベクトル化とカテゴリ化
x_train = vectorize_sequences(training_data_list)
one_hot_train_labels = to_one_hot(answer_label_list)
# one_hot_train_labels = to_categorical(answer_label_list)

# 学習用と精度の計測用で半分に分ける (今回の場合は、全てのデータを学習させるため、一時的に無効にしています)
# slice_index = len(x_train) // 2
x_val = x_train
# x_val = x_train[:slice_index]
partial_x_train = x_train
# partial_x_train = x_train[slice_index:]
y_val = one_hot_train_labels
# y_val = one_hot_train_labels[:slice_index]
partial_y_train = one_hot_train_labels
# partial_y_train = one_hot_train_labels[slice_index:]

# batchサイズをデータと同じ数にする
batch_size_num = len(partial_x_train)

# 学習させる
history = model.fit(partial_x_train,
partial_y_train,
epochs = epochs_num,
batch_size = batch_size_num,
validation_data = (x_val, y_val))

# 精度を可視化
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
jpg_count += 1
# 損失関数のグラフを保存する
plt.savefig(graph_directory + str(jpg_count).zfill(8) + ".jpg")
# リセットする+
plt.figure()



print(str(datetime.datetime.now()), model_file_name + "をセーブしています")
# 学習済みデータを保存する
model.save(model_file_name)
# 処理済みのファイルパスをcsvファイルに保存する
save_csv(completed_jsonl_files_csv_name, completed_jsonl_files_info)

# 学習に使用したデータのバックアップ
if converted_backup == True:
    training_data_list_old += training_data_list
    training_data_list_str_old += training_data_list_str
    answer_label_list_old += answer_label_list
    answer_label_list_str_old += answer_label_list_str

# 初期化
training_data_list = []
training_data_list_str = []
answer_label_list = []
answer_label_list_str = []

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
    print(str(datetime.datetime.now()), "読み込めなかったjsonlファイルの件数 : " + str(count_error_jsonl))
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

# 学習に使用したデータの表示 (少量のデータでのテスト用)
if converted_backup == True:
    print("学習に使用したデータ")
    for i, v in zip(answer_label_list_str_old, training_data_list_str_old):
        print(i, "\t", v)
