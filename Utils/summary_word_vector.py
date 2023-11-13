import pickle
import torch
from Utils.paths import *
from Utils.outfitDataset import OutdatasetLst, nameToPath
from Utils.AutoCache import Cache
cache = Cache()


def read_pickle(dataset):
    assert dataset in OutdatasetLst, f"dataset illegal, got {dataset}"
    with open(nameToPath[dataset], 'rb') as f:
        X_dict = pickle.load(f)
    return X_dict


def summary_lex(dataset):
    """
    统计dataset中的各个词语出现次数
    """
    X_dict = read_pickle(dataset)
    word_dict = {}
    for i in X_dict['tokenize']:
        for j in i['wordCutResult']:
            if j not in word_dict:
                word_dict[j] = 1
            else:
                word_dict[j] += 1
    sorted_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict


@cache.cache_result(cache_path='cache_summary_word_in_dataset.pkl')
def summary_word_in_dataset(word, dataset):
    X_Dict = read_pickle(dataset)
    sentence = []
    Index = []
    cnt = -1
    for i in X_Dict['tokenize']:
        cnt += 1
        if word in i['wordCutResult']:
            sentence.append("".join(i['wordCutResult']))
            Index.append(cnt)
    return {"sentence": sentence, "Index": Index}


if __name__ == '__main__':
    datasetLst = OutdatasetLst
    checkLst = summary_word_in_dataset("女人", "anwang")
    print("end")
