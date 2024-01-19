import warnings

warnings.filterwarnings('ignore')
from gensim import models
model = models.word2vec.Word2Vec.load('model/word2vec.model')


def generateSimilarity(word1, word2):
    try:
        distance = model.wv.similarity(word1, word2)
        return distance
    except:
        return 0.2
