from Modules.dbScan import *
from Utils.summary_word_vector import summary_lex
from Utils.evaluateCluster import evaluateDBScanMetric
from ConvWordToVecWithMECT import preprocess
import tqdm
from Utils.LLMDataExpand import Baidu, dataConvert
import os
import argparse
from Utils.Lab.lab_of_lowdimension import main


def Find_many_word(dataset1, dataset2, mes=False, Count=50):
    """
    在两个数据集中找到都出现过大于50个的词语
    """
    Lex_tieba = summary_lex(dataset1)
    Lex_weibo = summary_lex(dataset2)
    count = Count
    aim_word_Lst = []  # 统计一下在两个词典中出现次数都大于count的词组
    for i in Lex_tieba:
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
            if mes:
                print(
                    f"word {i} occurs in dataset {dataset1}:{Lex_tieba[i]} and occurs in dataset {dataset2}:{Lex_weibo[i]}")
    # print(aim_word_Lst,len(aim_word_Lst))
    return aim_word_Lst


def DrawWordCompare():
    # print(Find_many_word('wiki', 'test', Count=5))
    cant_wordList = ['开心', '男生', '数据', '客场', '此案', '通讯员']
    normal_wordList = ['我', '的', '自己', '警方']
    for word in cant_wordList + normal_wordList:
        try:
            X_ = read_vector('wiki', word)
            X1_ = read_vector('test', word)
            print(X_.shape)
            print(X1_.shape)
            draw_cluster_res_of_single_word(word, X_, X1_)
        except Exception as e:
            raise e


def calcSentence(baseDatabase='wiki', eps=18, metric='euclidean', min_samples=4, maxLength=20000):
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
                # Vector = Dimensionality_reduction(wordVector[ID][wordID].reshape(1, -1))
                Vector = wordVector[ID][wordID]
                # 拿到word和对应的Vector
                debugInfo(f'clustering word:{word}')
                writeLog(f"INFO: clustering word:{word}")
                clustera = cluster(
                    baseDatabase,
                    word,
                    savefig=False,
                    eps=eps,
                    metric=metric,
                    min_samples=min_samples,
                    maxLength=maxLength,
                    refresh=False
                )
                debugInfo(f'success running cluster function with word {word}')
                writeLog(f"success running cluster function with word {word}")
                # 计算出聚类结果

                classify = clustera['cluster result']
                count_N1 = sum([1 if i == -1 else 0 for i in classify])
                debugInfo(f"词语{word}聚类结果离群点数：{count_N1}")
                writeLog(f"词语{word}聚类结果离群点数：{count_N1}")
                debugInfo(f"词语{word}聚类结果聚类数量：{clustera['num_of_clusters']}")
                writeLog(f"词语{word}聚类结果聚类数量：{clustera['num_of_clusters']}")
                center = getCenter(clustera['result class instance'])
                if clustera['num_of_clusters'] == 1:
                    center = clustera['cluster_members']

                res.append(
                    [word, is_in_epsilon_neighborhood(Vector, center, epsilon=eps, metric=metric)]
                )
                debugInfo(f"append {word} in res")
                writeLog(f"append {word} in res")
                writeResult(f"{res}")
            except Exception as e:
                debugInfo(f"clustering word {word} with error {e}")
                writeLog(f"clustering word {word} with error {e}")
                res.append(
                    [word, 404]
                )
                writeResult(f"{res}")
    writeResult(f"{res}")


args_list = [
    {'name': '--mode', 'type': str, 'default': 'generate'},
    {'name': '--eps', 'type': int, 'default': 18, 'help': '聚类所使用的eps值'},
    {'name': '--metric', 'type': str, 'default': 'euclidean', 'help': '聚类所使用的距离算法'},
    {'name': '--min_samples', 'type': int, 'default': 4, 'help': '聚类所使用的min_samples参数'},
    {'name': '--maxLength', 'type': int, 'default': 20000, 'help': '聚类所用的最多的向量数量'}
]


parser = argparse.ArgumentParser(description='Process some integers.')
for arg in args_list:
    arg_name = arg['name']
    del arg['name']
    parser.add_argument(arg_name, **arg)


args = parser.parse_args()


if args.mode == 'test':
    evaluateDBScanMetric()
elif args.mode == 'generate':
    calcSentence(
        eps=args.eps,
        metric=args.metric,
        min_samples=args.min_samples,
        maxLength=args.maxLength
    )
elif args.mode == 'lowDimensionLab':
    main()
elif args.mode == 'CompareSensitiveWordLab':
    DrawWordCompare()
elif args.mode == 'expandBaseData':
    Baidu.Expand()
    dataConvert.Convert()