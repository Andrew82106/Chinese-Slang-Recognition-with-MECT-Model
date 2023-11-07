import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from Utils.paths import *
from Utils.summary_word_vector import summary_lex
import pickle
import torch


def saveFig(X, clusters, name="your_plot_name", xlabel='Feature 1', ylabel='Feature 2', title='DBSCAN Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'cluster_result/{name}.png')
    

def dbscan(X, eps=25, savefig=False, word=None):
    """
    X: input matrix with size N x 256
    """
    X = StandardScaler().fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X)
    if savefig:
        saveFig(
            X, 
            clusters, 
            name=savefig if word is None else f"{word} in {savefig}", 
            title=f"{savefig} clustering" if word is None else f"{savefig} clustering of word {word}"
        )
    return clusters


def mkData():
    # 生成样本数据
    n_samples = 100
    n_features = 256
    X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
    return X


def read_vector(dataset, word):
    assert dataset in ['weibo', 'tieba'], f"dataset illegal, got {dataset}"
    nameToPath = {
        'tieba': tieba_vector,
        'weibo': weibo_vector
    }
    with open(nameToPath[dataset], 'rb') as f:
        X_dict = pickle.load(f)
    X = None
    for ID in range(len(X_dict['tokenize'])):
        sentence_meta = X_dict['tokenize'][ID]
        for wordID in range(len(sentence_meta['wordCutResult'])):
            
            if sentence_meta['wordCutResult'][wordID] == word:
                vector = X_dict['wordVector'][ID][wordID]
                # print(vector.shape)
                if X is None:
                    X = vector.unsqueeze(0)
                else:
                    X = torch.cat((X, vector.unsqueeze(0)), dim=0)
    return X
    

def cluster(dataset, word):
    X = read_vector(dataset, word)
    res = dbscan(X, savefig=dataset, word=word)
    print(f"number of nodes:{len(res)}\nnumber of clusters:{len(set(res))}")
    


if __name__ == "__main__":
    word_sit = summary_lex('weibo')
    word_sit1 = summary_lex('tieba')
    for i in word_sit:
        if i not in word_sit1:
            continue
        if word_sit[i] > 50 and word_sit1[i]>50:
            print(word_sit[i])
            print(word_sit1[i])
            cluster('weibo', i)
            cluster('tieba', i)