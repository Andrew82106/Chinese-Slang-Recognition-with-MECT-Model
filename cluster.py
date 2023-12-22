import torch

from Modules.dbScan import *
from Utils.paths import *
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
    # 需要比较聚类结果的词语列表
    cant_wordList = ['开心', '男生', '数据', '客场', '此案', '通讯员']
    normal_wordList = ['我', '的', '自己', '警方']

    # 对每个词语进行处理
    for word in cant_wordList + normal_wordList:
        try:
            # 读取两个数据集中特定词语的向量
            X_ = read_vector('wiki', word)
            X1_ = read_vector('test', word)

            # 打印两个向量的形状
            print(X_.shape)
            print(X1_.shape)

            # 绘制单个词语的聚类结果
            draw_cluster_res_of_single_word(word, X_, X1_)

        except Exception as e:
            raise e  # 如果出现异常，将异常向上抛出


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


def mergeVectorsByWordWithIndices(tokenizeRes, wordVector):
    """
    合并词向量并记录每个词语在原数据集中的索引位置

    参数:
    - tokenizeRes: 原始数据集的分词结果
    - wordVector: 对应数据集的词向量

    返回值:
    - merged_matrices: 包含词向量及索引位置的字典
    """
    word_dict = {}  # 创建一个字典用于存储单词及其对应的向量
    word_indices = {}  # 创建一个字典用于存储单词及其对应的索引位置
    word_list = []  # 存储所有单词的列表
    word_index = 0  # 用于记录单词的整体索引位置

    # 遍历分词结果
    for ID in tqdm.tqdm(range(len(tokenizeRes)), desc='合并待打标数据集矩阵中'):
        for wordID in range(len(tokenizeRes[ID]['wordCutResult'])):
            word = tokenizeRes[ID]['wordCutResult'][wordID]  # 获取单词
            word_list.append(word)  # 将单词添加到列表中
            Vector = wordVector[ID][wordID]  # 获取单词对应的向量

            # 如果单词不在字典中，创建一个新的键值对
            if word not in word_dict:
                word_dict[word] = [Vector]
                word_indices[word] = [word_index]  # 记录单词的索引位置
            else:
                word_dict[word].append(Vector)  # 如果单词已存在，将向量追加到现有列表中
                word_indices[word].append(word_index)  # 记录单词的索引位置
            word_index += 1  # 更新整体索引位置

    merged_matrices = {}  # 存储合并后的矩阵
    # 对于每个单词及其对应的向量列表，将向量列表堆叠成矩阵，并存储到字典中
    for word, vectors in word_dict.items():
        merged_matrix = torch.stack(vectors)
        merged_matrices[word] = {
            'vectors': merged_matrix,  # 存储合并后的矩阵到字典中
            'indices': word_indices[word]  # 存储对应单词的索引位置列表
        }

    # debug部分：确保合并的词语与索引位置匹配
    for word in merged_matrices:
        indices = merged_matrices[word]['indices']
        for index in indices:
            if word_list[index] != word:
                raise Exception(f"Word ERROR:词语合并时索引出错！索引得到词语：{word_list[index]} 实际词语：{word}")

    return merged_matrices  # 返回包含词向量及索引位置的字典


def reduceDimensionsForMatrices(merged_matrices):
    reduced_matrices = {}  # 用于存储降维后的矩阵

    # 遍历 merged_matrices 中的每个词语及其对应的向量矩阵
    for word, data in tqdm.tqdm(merged_matrices.items(), desc='对数据进行降维'):
        vectors_matrix = data['vectors']  # 获取词向量矩阵

        # 对词向量矩阵进行降维处理
        reduced_matrix = Dimensionality_reduction(vectors_matrix)

        # 将降维后的矩阵存储到 reduced_matrices 中
        reduced_matrices[word] = reduced_matrix

    return reduced_matrices


def calcSentenceWithDimensionDecline(baseDatabase='wiki', eps=18, metric='euclidean', min_samples=4, maxLength=20000):
    """
    用降维进行聚类
    """
    print("starting cutting Result")
    writeLog("", init=1)
    cutResult = preprocess()
    # 这里cutResult存的是待标记数据集的向量化结果
    tokenizeRes = cutResult['tokenize']
    wordVector = cutResult['wordVector']
    merged_matrices = mergeVectorsByWordWithIndices(tokenizeRes, wordVector)
    reduced_matrices = reduceDimensionsForMatrices(merged_matrices)

    # 在这里对每个词语的信息进行汇总
    word_info_list = []  # 存储每个词语的信息
    for word, data in tqdm.tqdm(merged_matrices.items(), desc='存储每个词语的信息'):
        if word in reduced_matrices:  # 确保降维后的向量也存在
            original_vectors = data['vectors']  # 原始向量矩阵
            indices = data['indices']  # 词语在原数据集中的索引位置
            reduced_vectors = reduced_matrices[word]  # 降维后的向量矩阵

            # 汇总每个词语的信息
            word_info = {
                'word': word,
                'original_vectors': original_vectors,
                'indices': indices,
                'reduced_vectors': reduced_vectors
                # 还可以根据需要添加其他信息
            }
            word_info_list.append(word_info)  # 将每个词语的信息添加到列表中

    # 在这里使用 word_info_list 中的信息进行聚类操作或其他处理
    # ...

    return word_info_list  # 返回每个词语的信息列表，用于后续操作


args_list = [
    {'name': '--mode', 'type': str, 'default': 'test_dimension_decline'},
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
elif args.mode == 'test_dimension_decline':
    calcSentenceWithDimensionDecline(
        eps=args.eps,
        metric=args.metric,
        min_samples=args.min_samples,
        maxLength=args.maxLength
    )
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
elif args.mode == 'ConvertExpandedData':
    dataConvert.Convert()
elif args.mode == 'clean_function_cache':
    del_cluster_cache_path()
elif args.mode == 'clean_model_cache':
    clean_cache_path()