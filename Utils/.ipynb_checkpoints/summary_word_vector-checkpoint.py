import pickle
import torch
from Utils.paths import *
from Utils.outfitDataset import OutdatasetLst, nameToPath


def summary_lex(dataset):
    """
    统计dataset中的各个词语出现次数
    """
    assert dataset in OutdatasetLst, f"dataset illegal, got {dataset}"
    with open(nameToPath[dataset], 'rb') as f:
        X_dict = pickle.load(f)
    word_dict = {}
    for i in X_dict['tokenize']:
        for j in i['wordCutResult']:
            if j not in word_dict:
                word_dict[j] = 1
            else:
                word_dict[j] += 1
    sorted_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict