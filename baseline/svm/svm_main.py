import jieba.posseg as pseg  # 分词工具
from nltk import ngrams
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from read_database import generate_word_to_index, read_standard_input_bio, encode_word
from grams import bigrams, unigrams, trigrams, unigrams_in_batch, bigrams_in_batch, trigrams_in_batch
from Dictionary import common_last_names, uncommon_chars, proper_names_dict
from phonetic import extract_phonetic_features as epf
from LanguageModeling import generate
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
    unig = encode_word(unigrams(text), word_to_index)
    big = encode_word(bigrams(text), word_to_index)
    trig = encode_word(trigrams(text), word_to_index)

    # 获取词性标签
    words = pseg.cut(text)
    pos_tags = [word_.flag for word_ in words]
    # print(pos_tags)
    l = [str(ord(i)) for i in pos_tags[0]]
    pos_tags_embedding = int("".join(l))/100
    # print(pos_tags_embedding)
    # 字符数量
    num_characters = len(text)

    # 是否存在相同字符
    has_identical_chars = int(len(set(text)) < len(text))

    features.append(unig)
    features.append(big)
    features.append(trig)
    features.append(pos_tags_embedding)
    features.append(num_characters)
    features.append(has_identical_chars)

    return features


def extract_dictionary_features(text):
    features = []
    # 是否存在于专有名词字典中
    in_proper_names_dict = text in proper_names_dict
    features.append(in_proper_names_dict)
    words = text.split()
    starts_with_common_last_name = False
    contains_uncommon_chars = False

    if len(words) > 0:
        first_word = words[0]
        for name in common_last_names:
            if first_word.startswith(name):
                starts_with_common_last_name = True
                break

        for char in uncommon_chars:
            if char in first_word:
                contains_uncommon_chars = True
                break

    # 如果以常见姓氏开头并包含不常用字符作为名字，则设置特征值为True，否则为False
    has_unusual_name = starts_with_common_last_name and contains_uncommon_chars
    features.append(has_unusual_name)

    # 检查是否以地理政治实体或组织后缀词结尾
    political_suffixes = ["省", "市", "区", "县", "国", "组织", "集团"]  # 政治实体或组织后缀列表，根据实际情况扩展
    ends_with_political_suffix = False

    if len(words) > 0:
        last_word = words[-1]
        for suffix in political_suffixes:
            if last_word.endswith(suffix):
                ends_with_political_suffix = True
                break

    features.append(ends_with_political_suffix)

    return features


# 音形特征提取函数
def extract_phonetic_features(text):
    return [epf(text)]


# 语言建模特征提取函数

def extract_language_modeling_features(text, corpus_ngrams_, ngram_counts_, n):
    # 将文本转换为 n-gram
    grams = ngrams(text.split(), n)

    # 将 n-gram 组合为元组的列表
    n_grams_list = [' '.join(gram) for gram in grams]

    # 计算文本出现概率
    probability = 1.0
    for ngram in n_grams_list:
        probability *= (ngram_counts_[ngram] + 1) / (len(corpus_ngrams_) + len(set(corpus_ngrams_)))

    return [probability]


# 示例数据
text_data, labels = read_standard_input_bio()

word_to_index_ = generate_word_to_index(text_data + unigrams_in_batch(text_data) + bigrams_in_batch(text_data) + trigrams_in_batch(text_data))

xx = extract_basic_features(text_data[0], word_to_index_)

n = 3  # 选择 n 的大小，表示使用 trigram 模型
corpus_ngrams, ngram_counts = generate(text_data, n)

feature = []
for word in tqdm.tqdm(text_data):
    feature.append(extract_basic_features(word, word_to_index_) + extract_dictionary_features(word) + extract_phonetic_features(word) + extract_language_modeling_features(word, corpus_ngrams, ngram_counts, n))

X = feature
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 创建并训练SVM分类器
classifier = svm.SVC(kernel='rbf', C=100, gamma=1)
classifier.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = classifier.predict(X_test)
"""
# 计算评估指标
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"fake Precision: {precision}")
print(f"fake Recall: {recall}")
print(f"fake F1 Score: {f1}")
"""
# r值应该是做出的预测中正确的样本数量和做出的预测的数量的比值
# p值应该是实际的暗语中被预测到的数量和暗语数量的比值

r_ = 0
for index, label in enumerate(y_pred):
    if not label:
        continue
    if y_test[index] == 1:
        r_ += 1
r_ = r_ / len(y_pred)

p_ = 0
for index, label in enumerate(y_test):
    if not label:
        continue
    if y_pred[index] == 1:
        p_ += 1
p_ = p_ / len(y_test)

f_ = 2 * (p_ * r_) / (p_ + r_)

print(f"f:{f_} p:{p_} r:{r_}")
"""
# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# 创建SVM分类器
svm_model = svm.SVC()

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)

# 计算评估指标
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
"""