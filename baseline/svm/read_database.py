"""
text_data = [
    "文本1", "本2", "文本3",
    "文本1", "文本112", "文本3",
    "文本1", "文2本2", "文本3",
    "文本1", "w文本2", "文本3",
]  # 替换为实际文本数据
"""
from Utils.check.check_data import extract_word_from_bio
from Utils.paths import *
import pandas as pd
import jieba


def read_standard_input_bio():
    answer_list = extract_word_from_bio(Standard_Test_BIO)
    text = ''.join(answer_list)
    cant_word_list = read_cant_word()
    for word in cant_word_list:
        jieba.add_word(word)
    seg_list = list(jieba.cut(text))
    label = [1 if word in cant_word_list else 0 for word in seg_list]
    return seg_list, label


def read_cant_word():
    df = pd.read_excel(cant_word_location)
    return list(df['cant'])


def generate_word_to_index(text_data):
    word_to_index = {word: idx for idx, word in enumerate(text_data)}
    return word_to_index


def encode_word(word, word_to_index):
    return word_to_index.get(word, -1)
