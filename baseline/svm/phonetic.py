from pypinyin import pinyin, Style


def extract_phonetic_features(text):
    pinyin_list = pinyin(text, style=Style.NORMAL)
    phonetic_words = ''.join([''.join(word) for word in pinyin_list])
    res = int(''.join(str(ord(i)) for i in phonetic_words))
    return 10*res/(10**(len(str(res))))


if __name__ == '__main__':
    print(extract_phonetic_features("你好"))
