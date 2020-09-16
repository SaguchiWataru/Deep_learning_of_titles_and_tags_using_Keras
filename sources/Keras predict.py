import datetime
import numpy as np
import keras
keras.__version__
from keras.models import load_model
import itertools

def vectorize_sequences(sequences, dimension=100000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

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

# □■□■□■□■□■□■□■□■ 設定 □■□■□■□■□■□■□■□■

# 検出する正解ラベルの一覧
all_answer_labels = ["VOCALOID", "演奏してみた", "歌ってみた", "踊ってみた"]

# 読み込むモデルのファイル名
model_file_name = "title_to_tag_model.h5"

# □■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■

try:
    print(str(datetime.datetime.now()), model_file_name + "をロードしています")
    # 学習済みのデータと分類器を読み込む
    model = load_model(model_file_name)
    print(str(datetime.datetime.now()), model_file_name + "をロードしています")
except:
    print(str(datetime.datetime.now()), model_file_name + "をロードできませんでした")

while True:
    # コードポイントに変換しそれをベクトル化したものをpredict関数に渡す
    predict_list = model.predict(vectorize_sequences([count_code_point(str(input("動画のタイトルを入力してください>")))]))
    # 1次元配列に直す
    predict_list = list(itertools.chain.from_iterable(predict_list.tolist()))
    # 確率が高い正解ラベルのインデックスを求める
    answer_index = predict_list.index(max(predict_list))
    print(all_answer_labels[answer_index])
