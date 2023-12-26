from collections import Counter
from nltk import ngrams


def generate(corpus, n):
    # 统计语料库中每个 n-gram 的出现次数
    corpus_ngrams = [ngram for sentence in corpus for ngram in ngrams(sentence.split(), n)]
    ngram_counts = Counter(corpus_ngrams)
    return corpus_ngrams, ngram_counts