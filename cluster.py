from Modules.dbScan import *
from Utils.summary_word_vector import summary_lex
from Utils.evaluateCluster import evaluateDBScanMetric
from ConvWordToVecWithMECT import preprocess
import tqdm
import os
import argparse

metrics_calc_function = [
    calc_metric_in_steps,
    maximize_metric_for_eps
]  # 计算指标的函数，包括最大化函数、取样函数

evaluate_function = [
    index2,
    index1
]  # 衡量指标的函数，分别对应最大化函数、取样函数

function_name = [
    "取样函数",
    "最大化聚类函数"
]


def Find_many_word(dataset1, dataset2, mes=False):
    """
    在两个数据集中找到都出现过大于50个的词语
    """
    Lex_tieba = summary_lex(dataset1)
    Lex_weibo = summary_lex(dataset2)
    count = 50
    aim_word_Lst = []  # 统计一下在两个词典中出现次数都大于count的词组
    for i in Lex_tieba:
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
            if mes:
                print(
                    f"word {i} occurs in dataset {dataset1}:{Lex_tieba[i]} and occurs in dataset {dataset2}:{Lex_weibo[i]}")
    # print(aim_word_Lst,len(aim_word_Lst))
    return aim_word_Lst


def compare(word, datasetLst):
    file = "clusterRes/"
    for FunctionID in range(0, 2):  # 对不同指标函数的结果进行测试
        metricLst = []
        for i in datasetLst:
            # print(function_name[FunctionID], (i, word))
            metricLst.append(metrics_calc_function[FunctionID](i, word))
        for i in range(len(metricLst)):
            for j in range(i, len(metricLst)):
                if i == j:
                    continue
                index = evaluate_function[FunctionID](metricLst[i], metricLst[j])
                log = f"word {word} in dataset {datasetLst[i]} and {datasetLst[j]} with function {function_name[FunctionID]}: difference is {index}"
                # print(log)
                with open(os.path.join(file, "clusterRes.txt"), "a", encoding='utf-8') as f:
                    f.write(str(log) + "\n")


def calcSentence(baseDatabase='wiki', eps=18, metric='euclidean', min_samples=4):
    print("starting cutting Result")
    writeLog("", init=1)
    cutResult = preprocess()
    # 这里cutResult存的是待标记数据集的向量化结果
    tokenizeRes = cutResult['tokenize']
    wordVector = cutResult['wordVector']
    res = []
    word = ""
    initVector(baseDatabase)
    for ID in tqdm.tqdm(range(len(tokenizeRes)), desc='processing'):
        for wordID in range(len(tokenizeRes[ID]['wordCutResult'])):
            # for wordID in tqdm.tqdm(range(len(tokenizeRes[ID]['wordCutResult'])), desc=f'running sentence with ID:{ID}'):
            try:
                word = tokenizeRes[ID]['wordCutResult'][wordID]
                if word in ".,!。，":
                    res.append([word, True])
                    writeResult(str(res))
                    continue
                Vector = wordVector[ID][wordID]
                # 拿到word和对应的Vector
                debugInfo(f'clustering word:{word}')
                writeLog(f"TIME:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: clustering word:{word}\n")
                clustera = cluster(baseDatabase, word, savefig=False, eps=eps, metric=metric,
                                   min_samples=min_samples)
                debugInfo(f'success running cluster function with word {word}')
                writeLog(
                    f"TIME:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} success running cluster function with word {word}")
                # 计算出聚类结果

                classify = clustera['cluster result']
                count_N1 = sum([1 if i == -1 else 0 for i in classify])
                # print(f"聚类结果离群点数：{count_N1}")
                writeLog(f"聚类结果离群点数：{count_N1}\n")
                # print(f"聚类结果聚类数量：{clustera['num_of_clusters']}")
                writeLog(f"聚类结果聚类数量：{clustera['num_of_clusters']}")
                center = getCenter(clustera['result class instance'])
                if clustera['num_of_clusters'] == 1:
                    center = clustera['cluster_members']

                res.append(
                    [word, is_in_epsilon_neighborhood(Vector, center, epsilon=eps, metric=metric)]
                )
                debugInfo(f" append {word} in res")
                writeResult(f"{res}")
            except Exception as e:
                debugInfo(f"clustering word {word} with error {e}", show=1)
                writeLog(f"INFO: clustering word {word} with error {e}")
                res.append(
                    [word, 404]
                )
                writeResult(f"{res}")
    writeResult(f"{res}")
    # print(cutResult)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str, choices=['test', 'generate'], default='test')
parser.add_argument('--eps', type=int, default=18, help='聚类所使用的eps值')
parser.add_argument('--metric', type=str, default='euclidean', help='聚类所使用的距离算法')
parser.add_argument('--min_samples', type=int, default=4, help='聚类所使用的min_samples参数')
args = parser.parse_args()


if args.mode == 'test':
    evaluateDBScanMetric()
elif args.mode == 'generate':
    calcSentence(
        eps=args.eps,
        metric=args.metric,
        min_samples=args.min_samples
    )