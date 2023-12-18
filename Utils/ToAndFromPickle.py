import pickle
import torch


def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def append_tokenize_to_pickle(path, tokenize):
    res = load_from_pickle(path)
    res['tokenize'].append(tokenize)
    write_to_pickle(path, res)


def append_wordVector_to_pickle(path, wordVector):
    res = load_from_pickle(path)
    res['wordVector'].append(wordVector)
    write_to_pickle(path, res)