import jieba.analyse
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec, word2vec

# 文件位置需要改为自己的存放路径
# 将文本分词
with open('/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model/datasets/NER/wiki/wiki.txt',
          encoding='utf-8') as f:
    document = f.read()

with open('/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model/datasets/NER/PKU/pku_training.utf8',
          encoding='utf-8') as f:
    document1 = f.read().replace("  ", "")

document_cut = jieba.cut(document + document1)
result = ' '.join(document_cut)
with open('./wiki.txt', 'w', encoding="utf-8") as f2:
    f2.write(result)
# 加载语料
sentences = word2vec.LineSentence('./wiki.txt')
# 训练语料
path = get_tmpfile("model/word2vec.model")  # 创建临时文件
model = Word2Vec(sentences, hs=1, min_count=1, window=10)
model.save("model/word2vec.model")