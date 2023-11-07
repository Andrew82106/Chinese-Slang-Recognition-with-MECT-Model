import pickle
import torch

def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(path, data):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)

