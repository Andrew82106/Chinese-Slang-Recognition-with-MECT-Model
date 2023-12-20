import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import time
from datetime import datetime
import numpy as np
"""
try:
    from ConvWordToVecWithMECT import preprocess
except:
    from ..ConvWordToVecWithMECT import preprocess
"""
try:
    from sklearnex import patch_sklearn, unpatch_sklearn
    patch_sklearn()
except:
    print("sklearnex isn't available, skip init sklearnex")

from Utils.paths import *
from sklearn.metrics.pairwise import pairwise_distances
from Utils.outfitDataset import OutdatasetLst, nameToPath
from Modules.DWT import *
import pickle
from Utils.AutoCache import Cache

cache = Cache()
plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually

X_dict = None


def writeLog(Content, init=False):
    if init:
        with open(cluster_Log_Path, "w", encoding='utf-8') as f:
            f.write(Content)
    else:
        with open(cluster_Log_Path, "a", encoding='utf-8') as f:
            f.write(f"TIME:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " + Content + "\n")


def writeResult(Content):
    with open(clusterResult_path, "w", encoding='utf-8') as f:
        f.write(Content + "\n")


def debugInfo(Content, show=0):
    if show:
        print(f"TIME:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: {Content}")


def saveFig(X, clusters, name="your_plot_name", xlabel='Feature 1', ylabel='Feature 2', title='DBSCAN Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(clusterResultPhoto_path, f'cluster_result/{name}.png'))
    plt.show()


def index1(metricA: int, metricB: int):
    """
    按照第一种方法计算指标的相似度
    """
    return abs(metricA - metricB)


def index2(metricLstA: list, metricLstB: list):
    """
    按照二种方法计算指标的相似度
    """
    return naive_DTW(metricLstA, metricLstB)


def dbscan(X, metric, min_samples, eps=25, savefig=False, word=None, dataset=None):
    """
    # 进行dbscan聚类
    Input: X: input matrix with size N x 256，即输入的词的向量集合
    Input: eps: 即ε-邻域的epsilon值
    Input: savefig: 是否保存聚类结果图
    Input: word: 该聚类对应的词语
    Input: dataset: 该聚类对应的词语所在的数据集
    Input: clusters: 聚类结果
    """
    X = StandardScaler().fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clusters = dbscan.fit_predict(X)
    n = len(set(clusters))
    if savefig:
        saveFig(
            X,
            clusters,
            name=(
                     savefig if word is None else f"{word}_in_{dataset}") + f"[with_eps={eps}&n={n}&min_samples={min_samples}]",
            title=(
                      f"{dataset} clustering" if word is None else f"{dataset} clustering of word {word}") + f"[with_eps={eps}&n={n}&min_samples={min_samples}]",
        )
    """
    if not GETRES:
        return clusters
    else:
        return dbscan
    """
    return {'cluster result': clusters, 'dbscan result': dbscan}


def getCenter(clusters, X=None):
    """
    # 获取核心点的索引和坐标
    input: clusters，即聚类结果
    output: core_indices，即核心点的索引
    output: core_points，即核心点的坐标
    """
    if X is not None:
        XX = StandardScaler().fit_transform(X)
        core_indices = clusters.core_sample_indices_
        core_points = XX[core_indices]
    else:
        core_points = clusters.components_
    """
    from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html:
    components_:ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    """
    # core_points = clusters.components_
    return core_points


def is_in_epsilon_neighborhood(new_vector, core_points, epsilon, metric):
    """
    # 判断新向量new_vector是否在核心点core_points的 ε-邻域内
    input: new_vector，即新向量
    input: core_points，即核心点
    input: epsilon，即ε-邻域的epsilon值
    input: metric，即距离计算的函数，默认为euclidean，即欧几里得距离
    output: is_in_epsilon，即判断结果
    note: 传入的new_vector一定要是经过了StandardScaler进行标准化后的才行
    """
    # 计算新向量与核心点之间的距离
    if not isinstance(new_vector, np.ndarray):
        new_vector = new_vector.numpy()
    distances = pairwise_distances(core_points, [new_vector], metric=metric)

    # 判断新向量是否在 ε-邻域内
    is_in_epsilon = np.any(distances <= epsilon)

    return is_in_epsilon


def mkData():
    # 生成样本数据
    n_samples = 100
    n_features = 256
    X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
    return X


def initVector(dataset):
    global X_dict
    with open(nameToPath[dataset], 'rb') as f:
        print("initing x dict")
        X_dict = pickle.load(f)
        print("finish loading x dict")
        time.sleep(0.5)


def read_vector(dataset, word, maxLength=20000):
    """
    # 从dataset数据集读取word词语的词向量
    Input: dataset: 数据集的名称，比如weibo，anwang等
    Input: word: 需要读取的词语
    Output: X: dataset数据集中word词语的词向量集合，size为N x embeddingLength
    maxLength: 用于聚类的最大向量数量
    """
    assert dataset in OutdatasetLst, f"dataset illegal, got {dataset}"
    if X_dict is None:
        initVector(dataset)
    R = X_dict['fastIndexWord'][word]
    if len(R) > maxLength:
        print(f"debug: length={len(R)}")
        R = R[: maxLength]
    return R


@cache.cache_result(cache_path='cache_function_cluster.pkl')
def cluster(dataset, word, eps=25, savefig=False, metric='euclidean', min_samples=5, maxLength=20000):
    """
    聚类接口api
    从dataset中对word进行聚类
    """
    X = read_vector(dataset, word, maxLength=maxLength)
    debugInfo(f"load vec of word {word}")
    try:
        res = dbscan(
            X,
            metric=metric,
            eps=eps,
            savefig=savefig,
            dataset=dataset,
            word=word,
            min_samples=min_samples
        )
    except Exception as e:
        raise e
    num_of_clusters = len(set(res['cluster result']))

    return {
        "num_of_clusters": num_of_clusters,
        "result class instance": res['dbscan result'],
        "cluster result": res['cluster result'],
        'cluster_members': X
    }


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
        """
        return {
            "num_of_clusters": num_of_clusters,
            "result class instance": res['dbscan result'],
            "cluster result": res['cluster result']
        }
        """
        metric = cluster(dataset, word, eps)['num_of_clusters']
        # print(f"metric:{metric}")
        if maxx_metric < metric:
            maxx_metric = metric
            maxx_met_eps = eps

    if delta > minndelta:
        new_delta = delta / 5
        # print(f"recursive to next delta:{new_delta}")
        res = maximize_metric_for_eps(dataset, word, new_delta, maxx_met_eps - (2 * new_delta),
                                      maxx_met_eps + 2 * new_delta)
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
    for eps in forLst:
        """
                return {
                    "num_of_clusters": num_of_clusters,
                    "result class instance": res['dbscan result'],
                    "cluster result": res['cluster result']
                }
        """
        metric = cluster(dataset, word, eps)['num_of_clusters']
        res.append(metric)
    # print(f"res:{res}")
    return res


if __name__ == "__main__":
    # calcSentence()
    # best_metric, best_eps = maximize_metric_for_eps("tieba", "你")
    # print(f"best_eps:{best_eps}, best_metric:{best_metric}")
    """
                    return {
                        "num_of_clusters": num_of_clusters,
                        "result class instance": res['dbscan result'],
                        "cluster result": res['cluster result']
                    }
    """
    """
    Lex_tieba = summary_lex("tieba")
    Lex_weibo = summary_lex("PKU")
    count = 50
    aim_word_Lst = []  # 统计一下在两个词典中出现次数都大于count的词组
    for i in tqdm.tqdm(Lex_tieba):
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
    # print(aim_word_Lst,len(aim_word_Lst))
    for i in aim_word_Lst:
        best_metric, best_eps = maximize_metric_for_eps("tieba", i)
        best_metric1, best_eps1 = maximize_metric_for_eps("weibo", i)
        print(f"word:{i}\nbest_metric in tieba:{best_metric}\nbest_metric in weibo:{best_metric1}")
    """
