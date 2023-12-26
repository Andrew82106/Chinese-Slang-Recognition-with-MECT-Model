def unigrams(text):
    return [c for c in text]


def bigrams(text):
    return [text[i:i + 2] for i in range(len(text) - 1)]


def trigrams(text):
    return [text[i:i + 3] for i in range(len(text) - 2)]


def unigrams_in_batch(text_list):
    res = []
    for text in text_list:
        res += unigrams(text)
    return res


def bigrams_in_batch(text_list):
    res = []
    for text in text_list:
        res+= bigrams(text)
    return res


def trigrams_in_batch(text_list):
    res = []
    for text in text_list:
        res += trigrams(text)
    return res


if __name__ == '__main__':
    print(unigrams("你好"))
    print(bigrams("你好"))
    print(trigrams("你好"))