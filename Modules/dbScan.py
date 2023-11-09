import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from Utils.paths import *
from Utils.summary_word_vector import summary_lex
from Utils.outfitDataset import OutdatasetLst, nameToPath
from Modules.DWT import *
import tqdm
import pickle
import torch
import functools
plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams['axes.unicode_minus']=False   #These two lines need to be set manually

def saveFig(X, clusters, name="your_plot_name", xlabel='Feature 1', ylabel='Feature 2', title='DBSCAN Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'cluster_result/{name}.png')

def index1(metricA: int, metricB: int):
    """
    按照第一种方法计算指标的相似度
    """
    return abs(metricA-metricB)


def index2(metricLstA: list, metricLstB: list):
    """
    按照二种方法计算指标的相似度
    """
    return naive_DTW(metricLstA, metricLstB)


@functools.lru_cache(1024)
def dbscan(X, eps=25, savefig=False, word=None, dataset=None):
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
    assert dataset in OutdatasetLst, f"dataset illegal, got {dataset}"
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
    

def cluster(dataset, word, eps=25, savefig=False):
    """
    聚类接口api
    """
    X = read_vector(dataset, word)
    res = dbscan(
        X, 
        eps=eps,
        savefig=savefig, 
        dataset=dataset,
        word=word
    )
    num_of_clusters = len(set(res))
    
    metric = num_of_clusters
    
    return metric


def generate_list_with_delta(min_interval, max_interval, delta):
    my_list = []
    current_value = min_interval
    while current_value <= max_interval:
        my_list.append(current_value)
        current_value += delta
    return my_list


def maximize_metric_for_eps(dataset, word, delta=10, min_interval=1, max_interval=100, minndelta=0.001):
    """
    以精度为标准，递归的查找metric的最大值，从而求得聚类的最大值
    """
    maxx_metric = 0
    maxx_met_eps = -1
    
    forLst = generate_list_with_delta(min_interval, max_interval, delta)
    # print(forLst)
    for eps in forLst:
        if eps <= 0:
            continue
        # print(f"eps:{eps} ", end="")
        metric = cluster(dataset, word, eps)
        # print(f"metric:{metric}")
        if maxx_metric < metric:
            maxx_metric = metric
            maxx_met_eps = eps
            
    if delta > minndelta:
        new_delta = delta/5
        # print(f"recursive to next delta:{new_delta}")
        res = maximize_metric_for_eps(dataset, word, new_delta, maxx_met_eps-(2*new_delta), maxx_met_eps+ 2*new_delta)
        if res == 0:
            return maxx_metric
        else:
            return res
    else:
        # return maxx_metric, maxx_met_eps
        return maxx_metric
    
    
def calc_metric_in_steps(dataset, word, delta=1, min_interval=1, max_interval=100):
    """
    平均的对某个范围内的聚类结果进行采样
    """
    # print('calc_metric_in_steps')
    forLst = generate_list_with_delta(min_interval, max_interval, delta)
    res = []
    for eps in tqdm.tqdm(forLst):
        metric = cluster(dataset, word, eps)
        res.append(metric)
    print(f"res:{res}")
    return res


if __name__ == "__main__":
    # best_metric, best_eps = maximize_metric_for_eps("tieba", "你")
    # print(f"best_eps:{best_eps}, best_metric:{best_metric}")
    Lex_tieba = summary_lex("tieba")
    Lex_weibo = summary_lex("PKU")
    count = 50
    aim_word_Lst = [] # 统计一下在两个词典中出现次数都大于count的词组
    for i in tqdm.tqdm(Lex_tieba):
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
    # print(aim_word_Lst,len(aim_word_Lst))
    for i in aim_word_Lst:
        best_metric, best_eps = maximize_metric_for_eps("tieba", i)
        best_metric1, best_eps1 = maximize_metric_for_eps("weibo", i)
        print(f"word:{i}\nbest_metric in tieba:{best_metric}\nbest_metric in weibo:{best_metric1}")
            