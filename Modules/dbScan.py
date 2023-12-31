import os

import matplotlib.pyplot as plt
import torch
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
from umap import UMAP
from Utils.paths import *
from sklearn.metrics.pairwise import pairwise_distances
from Utils.outfitDataset import OutdatasetLst, nameToPath
from Modules.DWT import *
import pickle
# from umap import UMAP
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA
from Utils.AutoCache import Cache

cache = Cache()
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
    # print(f"successfully rewrite file {clusterResult_path}")


def debugInfo(Content, show=0):
    if show:
        print(f"TIME:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: {Content}")


def Dimensionality_reduction(vectors_list, dimension=2, algo='default'):
    """
    降维算法接口
    输入矩阵，输出降维后的矩阵
    所有的降维算法都需要使用这一个接口
    接口输出的值已经经过归一化，无需额外归一化
    """
    if algo == 't-sne':
        assert vectors_list.shape[0] < 20000, 'too many points for t-sne!'
        U = TSNE(n_components=dimension, random_state=42, perplexity=1)
    elif algo == 'umap':
        U = UMAP(n_components=dimension, random_state=42)
    else:
        U = PCA(n_components=dimension, random_state=42)
    vectors = U.fit_transform(vectors_list)
    return StandardScaler().fit_transform(vectors)


def dimensionReduce(Matrix, dimension=2, algo='default'):
    """
    该接口本身是容错接口，但是这个容错在新算法中不应发生
    这个接口最好在未来被抛弃
    """
    # print(Matrix.shape)
    """
    if Matrix.shape[0] <= dimension:  # 如果词向量矩阵只有一行
        origin_shape = Matrix.shape[0]
        s_matrix = (Matrix for _ in range(dimension+1))
        vectors_matrix = torch.cat(tuple(s_matrix), dim=0)
        # print(vectors_matrix.shape)
        reduced_matrix = Dimensionality_reduction(vectors_matrix, dimension, algo=algo)
        reduced_matrix = reduced_matrix[0:origin_shape]
    else:
        vectors_matrix = Matrix
        reduced_matrix = Dimensionality_reduction(vectors_matrix, dimension, algo=algo)
    return StandardScaler().fit_transform(reduced_matrix)
    """
    reduced_matrix = Dimensionality_reduction(Matrix, dimension, algo=algo)
    return reduced_matrix
    # return reduced_matrix


def merge_matrix_and_reduce_dimension(base_data_matrix, test_data_matrix, dimension=2, algo='default'):
    """
    将base dataset中的词语和test dataset中的词语合并起来，进行降维操作，并且返回降维后的两个数据集中的向量
    """
    merged_matrix = torch.cat((base_data_matrix, test_data_matrix), dim=0)
    # print(base_data_matrix.shape, test_data_matrix.shape)
    # print(merged_matrix.shape)
    new_matrix = Dimensionality_reduction(merged_matrix, dimension, algo=algo)
    assert base_data_matrix.shape[0] + test_data_matrix.shape[0] == new_matrix.shape[0], f'降维后矩阵行数出错: {base_data_matrix.shape[0]}, {test_data_matrix.shape[0]}, {new_matrix.shape[0]}'
    new_base_data_matrix = new_matrix[0:base_data_matrix.shape[0]]
    new_test_data_matrix = new_matrix[base_data_matrix.shape[0]:base_data_matrix.shape[0] + test_data_matrix.shape[0]]
    return new_base_data_matrix, new_test_data_matrix


def draw_cluster_res_of_single_word(word, vectorList1, vectorList2=None):
    """
    对于 dataset 中的 word，对其进行降维并且画出其降维结果
    传入处理好的矩阵即可
    """
    # 处理 vectorList1
    # New_X1 = Dimensionality_reduction(vectorList1)
    New_X1 = vectorList1
    labels1 = np.zeros(New_X1.shape[0])  # 给 vectorList1 的数据打标签，用 0 表示

    if vectorList2 is not None:
        # 处理 vectorList2
        # New_X2 = Dimensionality_reduction(vectorList2)
        New_X2 = vectorList2
        labels2 = np.ones(New_X2.shape[0])  # 给 vectorList2 的数据打标签，用 1 表示
        # 合并数据和标签
        combined_data = np.vstack((New_X1, New_X2))
        combined_labels = np.concatenate((labels1, labels2))
        combined_sizes = np.concatenate((np.ones(New_X1.shape[0]) * 10, np.ones(New_X2.shape[0]) * 20))  # 调整不同类别点的大小
    else:
        combined_data = New_X1
        combined_labels = labels1
        combined_sizes = np.ones(New_X1.shape[0]) * 10  # 设置点的大小

    # 创建一个图
    fig, ax = plt.subplots(figsize=(6, 4))

    # 根据标签绘制散点图，并为不同类别的数据点着色
    scatter = ax.scatter(combined_data[:, 0], combined_data[:, 1], c=combined_labels, cmap='coolwarm', s=combined_sizes, label=['wiki', '暗语'])
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # 计算不同标签的样本数量
    num_label0 = np.sum(combined_labels == 0)
    num_label1 = np.sum(combined_labels == 1)

    # 在图中添加标签
    ax.text(0.1, 0.9, f'wiki数据集样本数: {num_label0}', transform=ax.transAxes)
    ax.text(0.1, 0.85, f'暗语样本数: {num_label1}', transform=ax.transAxes)

    ax.set_title(f'Word: {word}')
    plt.tight_layout()
    plt.savefig(os.path.join(LabCachePath1, f"降维算法比较图_词语{word}.png"))
    print(f"successfully save graph to path:{os.path.join(LabCachePath1, f'降维算法比较图_词语：{word}.png')}")


def saveFig(X, clusters, name="your_plot_name", xlabel='Feature 1', ylabel='Feature 2', title='DBSCAN Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(clusterResultPhoto_path, f'cluster_result/{name}.png'))
    plt.show()


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


def initVector(dataset, refresh=False):
    global X_dict
    if X_dict is None or refresh:
        with open(nameToPath[dataset], 'rb') as f:
            print("initing x dict")
            X_dict = pickle.load(f)
            print("finish loading x dict")
            time.sleep(0.5)


def read_vector(dataset, word, maxLength=None, refresh=True):
    """
    # 从dataset数据集读取word词语的词向量
    Input: dataset: 数据集的名称，比如weibo，anwang等
    Input: word: 需要读取的词语
    Output: X: dataset数据集中word词语的词向量集合，size为N x embeddingLength
    maxLength: 用于聚类的最大向量数量
    """
    assert dataset in OutdatasetLst, f"dataset illegal, got {dataset}"
    # if X_dict is None:
    initVector(dataset, refresh=refresh)
    if word not in X_dict['fastIndexWord']:
        raise KeyError(f"{word} not in pkl of dataset {dataset}")
    R = X_dict['fastIndexWord'][word]
    if maxLength is not None and len(R) > maxLength:
        # print(f"debug: length={len(R)}")
        R = R[: maxLength]
    return R


@cache.cache_result(cache_path='cache_function_cluster.pkl')
def cluster(dataset, word, eps=25, savefig=False, metric='euclidean', min_samples=5, maxLength=20000, refresh=True, dimension_d=False):
    """
    聚类接口api
    从dataset中对word进行聚类
    dimension_d: 是否使用降维算法
    """
    X = read_vector(dataset, word, maxLength=maxLength, refresh=refresh)
    if dimension_d:
        X = Dimensionality_reduction(X)
    # X = Dimensionality_reduction(X)
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


if __name__ == "__main__":
    X = read_vector('wiki', "数据")
    X1 = read_vector('test', "数据")
    draw_cluster_res_of_single_word('数据', X, X1)