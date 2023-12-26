import jieba.posseg as pseg  # 分词工具
import re  # 正则表达式库
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from read_database import generate_word_to_index, read_standard_input_bio, encode_word
from grams import bigrams, unigrams, trigrams, unigrams_in_batch, bigrams_in_batch, trigrams_in_batch
"""
这个SVM分类器是用于一个任务，将其视为一个二元分类问题。它采用支持向量机（SVM）（Cortes和Vapnik，1995）作为学习模型，并提出了以下四类特征：

1. 基本特征：包括字符单字、双字和三字组合以及表面形式；词性标签；字符数量；某些字符是否相同。这些基本特征有助于识别词形候选项的一些共同特征（例如，它们很可能是名词，很不可能包含单个字符）。

2. 字典特征：许多词形是从专有名词派生出来的非规则名词，同时保留了一些特征。比如，“薄督（Governor Bo）”和“吃省（Gourmand Province）”这两个词形分别源自它们的目标实体名称“薄熙来（Bo Xilai）”和“广东省（Guangdong Province）”。因此，采用了一个专有名词的字典，并提出以下特征：词项是否出现在字典中；词项是否以常用姓氏开头，并包含不常用字符作为其名字；词项是否以地理政治实体或组织后缀词结尾，但不在字典中。

3. 音形特征：许多词形是基于音形（在这里是中文拼音）的修改而创建的。例如，“饭饼饼（Rice Cake）”与其目标实体名称“范冰冰（Fan Bingbing）”具有相同的拼音。为了提取基于音形的特征，编制了一个由中文Gigaword语料库中的〈音形转录，词项〉对组成的字典。然后，对于每个词项，检查它是否与字典中的任何条目具有相同的音形转录，但包含不同的字符。

4. 语言建模特征：许多词形很少出现在一般新闻语料库中（例如，“六步郎（Six Step Man）”指的是NBA篮球运动员“勒布朗·詹姆斯（LeBron James）”）。因此，使用从Gigaword训练的基于字符的语言模型来计算每个词项的出现概率，并使用n-gram...

总体来说，该分类器综合利用了基本特征、字典信息、音形相似性和语言模型的概率等特征，以帮助区分和分类词形候选项。
"""


# 基本特征提取函数
def extract_basic_features(text, word_to_index):
    features = []
    # 提取字符 unigram、bigram 和 trigram
    unigrams = [encode_word(char, word_to_index) for char in text]
    bigrams = [encode_word(text[i:i + 2], word_to_index) for i in range(len(text) - 1)]
    trigrams = [encode_word(text[i:i + 3], word_to_index) for i in range(len(text) - 2)]

    # 获取词性标签
    words = pseg.cut(text)
    pos_tags = [word_.flag for word_ in words]

    # 字符数量
    num_characters = len(text)

    # 是否存在相同字符
    has_identical_chars = len(set(text)) < len(text)

    features.extend(unigrams)
    features.extend(bigrams)
    features.extend(trigrams)
    features.extend(pos_tags)
    features.append(num_characters)
    features.append(has_identical_chars)

    return features


# 字典特征提取函数（假设已有一个名为proper_names_dict的专有名词字典）
def extract_dictionary_features(text, proper_names_dict):
    features = []
    # 是否存在于专有名词字典中
    in_proper_names_dict = text in proper_names_dict

    # 检查是否以常见姓氏开头，并包含不常用字符作为名字
    # 此处省略相关代码，需要根据实际数据和要求编写

    # 检查是否以地理政治实体或组织后缀词结尾
    # 此处省略相关代码，需要根据实际数据和要求编写

    features.append(in_proper_names_dict)
    # 添加其他字典特征

    return features


# 音形特征提取函数
def extract_phonetic_features(text, phonetic_dict):
    features = []
    # 从音形字典中检查是否有相同音形但不同字符的条目
    # 此处省略相关代码，需要根据实际数据和要求编写

    return features


# 语言建模特征提取函数
def extract_language_modeling_features(text, language_model):
    features = []
    # 使用语言模型计算每个词项的出现概率或其他语言建模特征
    # 此处省略相关代码，需要根据实际数据和要求编写

    return features


# 示例数据
text_data, labels = read_standard_input_bio()

word_to_index_ = generate_word_to_index(text_data + unigrams_in_batch(text_data) + bigrams_in_batch(text_data) + trigrams_in_batch(text_data))

feature = [[] for _ in range(len(text_data))]


# 此处写特征提取代码
for index, word in enumerate(text_data):
    feature[index].append([len(word), len(word)+1])
# 此处写特征提取代码

X = feature
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练SVM分类器
classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = classifier.predict(X_test)

# 计算评估指标
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
