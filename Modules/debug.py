import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
import pandas as pd
import tqdm

from dbScan import read_vector, torch, dbscan, getCenter, cluster, StandardScaler, is_in_epsilon_neighborhood, silhouette_score


def read_vector0(dataset, word):
    print(f"read vector in {dataset}, {word}")
    clusterA = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 2]
    ]
    clusterB = [
        [10, 10],
        [11, 10],
        [11, 11],
        [10, 11]
    ]
    clusterC = [
        [-10, -10],
        [-11, -10],
        [-11, -11],
        [-10, -11]
    ]
    clusterLst = clusterA + clusterB + clusterC
    # plt.scatter([i[0] for i in clusterLst], [i[1] for i in clusterLst])
    return torch.tensor(clusterLst)


def debug1(dataset, word, eps=25, savefig=False, metric='euclidean', min_samples=5):
    X = read_vector0(dataset, word)
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
        # {'cluster result': clusters, 'dbscan result': dbscan}
    except Exception as e:
        print(f"eps:{eps}\nX:{X}\ndataset:{dataset}\nword:{word}")
        raise e
    num_of_clusters = len(set(res['cluster result']))
    center = getCenter(res['dbscan result'], X)
    print(center, len(center))
    return {
        "num_of_clusters": num_of_clusters,
        "result class instance": res['dbscan result'],
        "cluster result": res['cluster result']
    }


def debug2():
    dataset = 'wiki'
    word = '我们'
    eps = 15
    metric = 'euclidean'
    clustera = cluster(dataset, word, savefig=False, eps=eps, metric=metric)
    X = read_vector(dataset, word)
    center = getCenter(clustera['result class instance'], X)
    classify = clustera['cluster result']
    count_N1 = sum([1 if i == -1 else 0 for i in classify])
    Count_N1 = 0
    for i in classify:
        if i == -1:
            Count_N1 += 1
    assert count_N1 == Count_N1, "ERROR"
    print(f"聚类结果离群点数：{count_N1}")
    res = []
    X = StandardScaler().fit_transform(X)
    for C in range(len(X)):
        res.append(is_in_epsilon_neighborhood(X[C], center, epsilon=eps, metric=metric))
    print(f"基于距离公式计算的离群点数：{len(res) - sum(res)}")
    print(f"聚类结果聚类数量：{clustera['num_of_clusters']}")


def debug3_1(dataset, word, eps=15, minsample=4, metric='euclidean'):
    clustera = cluster(dataset, word, savefig=False, eps=eps, metric=metric, min_samples=minsample)
    print(f"聚类数量：{clustera['num_of_clusters']}|", end='')
    X = read_vector(dataset, word)
    center = getCenter(clustera['result class instance'], X)
    if not len(center):
        print(f"结果展示————参数:eps:{eps},minsample:{minsample},dataset:{dataset},word:{word},无有效聚类")
        return {'eps': eps, 'minsample': minsample}
    if clustera['num_of_clusters'] == 1 and clustera['cluster result'][0] != -1:
        print(f"结果展示————参数:eps:{eps},minsample:{minsample},dataset:{dataset},word:{word},聚类数量为1")
        return {'eps': eps, 'minsample': minsample}
    classify = clustera['cluster result']
    count_N1 = sum([1 if i == -1 else 0 for i in classify])
    Count_N1 = 0
    for i in classify:
        if i == -1:
            Count_N1 += 1
    assert count_N1 == Count_N1, "ERROR"
    res = []
    X = StandardScaler().fit_transform(X)
    for C in range(len(X)):
        res.append(is_in_epsilon_neighborhood(X[C], center, epsilon=eps, metric=metric))
    assert len(res) - sum(res) - count_N1 == 0, "聚类结果离群点数计算错误"
    metric_ = silhouette_score(X, classify, metric='euclidean')
    print(f"结果展示————参数:eps:{eps},minsample:{minsample},dataset:{dataset},word:{word},metric:{metric};平均轮廓系数为{metric_}")
    return {'eps': eps, 'minsample': minsample, 'metric': metric, '轮廓系数': metric_}


def debug3():
    start = 14
    end = 29
    delta = 1
    datasetLst = ['wiki', 'PKU', 'anwang', 'tieba']
    wordLst = ['我们', '你们', '中国', '工作']
    df = {'eps': [], 'minsample': [], 'metric': [], '轮廓系数': [], 'word': [], 'dataset': []}
    for word in wordLst:
        for dataset_ in datasetLst:
            eps = start
            while eps <= end:
                res = debug3_1(dataset_, word, eps=eps, metric='euclidean')
                if '轮廓系数' in res:
                    df['轮廓系数'].append(res['轮廓系数'])
                    df['metric'].append(res['metric'])
                else:
                    df['轮廓系数'].append("-1")
                    df['metric'].append("-1")
                df['eps'].append(res['eps'])
                df['minsample'].append(res['minsample'])
                df['word'].append(word)
                df['dataset'].append(dataset_)
                eps += delta
            pd.DataFrame(df).to_csv("./lab.csv")


def drawResultDebug3():
    df = pd.read_csv("./lab.csv")
    datasetLst = ['wiki', 'PKU', 'anwang', 'tieba']
    wordLst = ['我们', '你们', '中国', '工作']
    for word in wordLst:
        for dataset in datasetLst:
            df1 = df[(df['dataset'] == dataset) & (df['word'] == word)]
            X = df1['eps']
            Y = df1['轮廓系数']
            plt.plot(X, Y)
    plt.xlabel('eps', fontproperties=font)
    plt.ylabel('轮廓系数', fontproperties=font)
    plt.title(f"不同数据集不同词语的轮廓系数汇总", fontproperties=font)
    plt.show()


if __name__ == '__main__':
    # xx = debug1("wiki", "123", savefig=1, eps=1.42, min_samples=4)
    # print(xx)
    # debug3()
    drawResultDebug3()