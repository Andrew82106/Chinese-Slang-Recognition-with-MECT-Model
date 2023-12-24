import torch
from Modules.dbScan import (cluster, read_vector, initVector, writeLog, writeResult, draw_cluster_res_of_single_word,
                            debugInfo, getCenter,
                            is_in_epsilon_neighborhood, dimensionReduce, merge_matrix_and_reduce_dimension)
from Utils.paths import *
from Utils.summary_word_vector import summary_lex
from Utils.evaluateCluster import evaluateDBScanMetric
from ConvWordToVecWithMECT import preprocess
import tqdm
from Utils.outfitDataset import OutdatasetLst
from Utils.LLMDataExpand import Baidu, dataConvert, merge_data
import argparse
from Utils.Lab.lab_of_lowdimension import main
import matplotlib as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually

args_list = [
    {'name': '--mode', 'type': str, 'default': 'test_dimension_decline'},
    {'name': '--eps', 'type': float, 'default': 18, 'help': '聚类所使用的eps值'},
    {'name': '--metric', 'type': str, 'default': 'euclidean', 'help': '聚类所使用的距离算法'},
    {'name': '--min_samples', 'type': int, 'default': 4, 'help': '聚类所使用的min_samples参数'},
    {'name': '--maxLength', 'type': int, 'default': 1000, 'help': '聚类所用的最多的向量数量'}
]

parser = argparse.ArgumentParser(description='Process some integers.')
for arg in args_list:
    parser.add_argument(arg['name'], type=arg['type'], default=arg['default'], help=arg.get('help', None))

parser.add_argument('--status', default='generate', choices=['train', 'run', 'generate'])
parser.add_argument('--extra_datasets', default='None', choices=OutdatasetLst)
parser.add_argument('--msg', default='_')
parser.add_argument('--train_clip', default=False, help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--see_convergence', default=False)
parser.add_argument('--see_param', default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--test_train', default=False)
parser.add_argument('--number_normalized', type=int, default=0,
                    choices=[0, 1, 2, 3], help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name', default='yj', choices=['lk', 'yj'])
parser.add_argument('--update_every', default=1, type=int)
parser.add_argument('--use_pytorch_dropout', type=int, default=0)

parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)

parser.add_argument('--word_min_freq', default=1, type=int)

# hyper of training
parser.add_argument('--early_stop', default=40, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=20, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='uniform', help='norm|uniform')
parser.add_argument('--self_supervised', default=False)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)

parser.add_argument('--warmup', default=0.1, type=float)

# hyper of model
parser.add_argument('--use_bert', type=int)
parser.add_argument('--model', default='transformer', help='lstm|transformer')
parser.add_argument('--lattice', default=1, type=int)
parser.add_argument('--use_bigram', default=1, type=int)
parser.add_argument('--hidden', default=-1, type=int)
parser.add_argument('--ff', default=3, type=int)
parser.add_argument('--layer', default=1, type=int)
parser.add_argument('--head', default=8, type=int)
parser.add_argument('--head_dim', default=20, type=int)
parser.add_argument('--scaled', default=False)
parser.add_argument('--ff_activate', default='relu', help='leaky|relu')

parser.add_argument('--k_proj', default=False)
parser.add_argument('--q_proj', default=True)
parser.add_argument('--v_proj', default=True)
parser.add_argument('--r_proj', default=True)

parser.add_argument('--attn_ff', default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos', default=False)
parser.add_argument('--use_rel_pos', default=True)
# 相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared', default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm', default=False)
parser.add_argument('--rel_pos_init', default=1)
parser.add_argument('--four_pos_shared', default=True, help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion', default='ff_two', choices=['ff', 'attn', 'gate', 'ff_two', 'ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared', default=True, help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，所以可以使4个位置的初始embedding size缩小，最后融合时回到正常的hidden size即可')  # modify this to decrease the use of gpu memory

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='nda')

over_all_dropout = -1
parser.add_argument('--embed_dropout_before_pos', default=False)
parser.add_argument('--embed_dropout', default=0.5, type=float)
parser.add_argument('--gaz_dropout', default=0.5, type=float)
parser.add_argument('--char_dropout', default=0, type=float)
parser.add_argument('--output_dropout', default=0.3, type=float)
parser.add_argument('--pre_dropout', default=0.5, type=float)
parser.add_argument('--post_dropout', default=0.3, type=float)
parser.add_argument('--ff_dropout', default=0.15, type=float)
parser.add_argument('--ff_dropout_2', default=-1, type=float)
parser.add_argument('--attn_dropout', default=0, type=float)
parser.add_argument('--embed_dropout_pos', default='0')
parser.add_argument('--abs_pos_fusion_func', default='nonlinear_add',
                    choices=['add', 'concat', 'nonlinear_concat', 'nonlinear_add', 'concat_nonlinear',
                             'add_nonlinear'])

parser.add_argument('--dataset', default='msra', help='weibo|resume|ontonotes|msra|tieba')
parser.add_argument('--label', default='all', help='ne|nm|all')
args = parser.parse_args()


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
    # cant_wordList = ['开心', '数据', '日本', '你在开玩笑', '立体', '金姐', '啥也看不见', '绞丝']
    cant_wordList = ['椅子', '新西兰', '友情', '巧克力', '张韶涵', '骨折', '尘土帽子', '中间部分', '马上', '海绵',
                     '绿箭口香糖', '努力', '一种称呼', '维生素', '清凉', '现在', '温暖', '宜家']
    # normal_wordList = ['我', '的', '自己', '可以', '运动', '时尚']
    normal_wordList = ['腹部', '内裤', '；', '丰胸', '胸部', '放入', '中医', '经络', '靠', '或者', '大脑', '导语',
                       '排除', '装扮', '消耗', '缺氧', '颈部', '卡路里', '头晕', '欧令奋', '鼻', '医师',
                       '妹', '皮鞋']

    # 对每个词语进行处理
    for word in cant_wordList + normal_wordList:
        try:
            # 读取两个数据集中特定词语的向量
            X_ = read_vector('wiki', word, maxLength=args.maxLength)
            X1_ = read_vector('test', word)
            X_, X1_ = merge_matrix_and_reduce_dimension(X_, X1_, dimension=2, algo='default')
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


def reduceDimensionsForMatrices(merged_matrices, dimension=2, algo='default'):
    """
    仅对于merged_matrices进行降维
    """
    reduced_matrices = {}
    for word, data in tqdm.tqdm(merged_matrices.items(), desc='对数据进行降维'):
        vectors_matrix = data['vectors']  # 获取词向量矩阵
        # print(vectors_matrix.shape)
        reduced_matrix = dimensionReduce(vectors_matrix, dimension=dimension, algo=algo)
        reduced_matrices[word] = reduced_matrix
    return reduced_matrices


def convertResDictToResList(ResDict):
    ResList = []
    for Index in range(len(ResDict)):
        assert isinstance(list(ResDict.keys())[0], int), f'KEY TYPE ERROR:{type(list(ResDict.keys())[0])}'
        label = ResDict[Index]
        ResList.append(label)
    return ResList


def calcSentenceWithDimensionDecline(
        baseDatabase='wiki',
        eps=18,
        metric='euclidean',
        min_samples=4,
        maxLength=2000,
        dimension=2,
        algo='t-sne'
):
    """
    用降维直接进行距离计算，不使用聚类算法
    """
    print("starting cutting Result")
    cutResult = preprocess(args)
    # 这里cutResult存的是待标记数据集的向量化结果
    tokenizeRes = cutResult['tokenize']
    wordVector = cutResult['wordVector']
    merged_matrices = mergeVectorsByWordWithIndices(tokenizeRes, wordVector)
    initVector(baseDatabase)
    resDict = {}
    cnt_404 = 0
    cnt_false = 0
    cnt_true = 0
    for word in tqdm.tqdm(merged_matrices, desc='processing word with immediate distance calculation'):
        if (cnt_true + cnt_404 + cnt_false) % 3000 == 0:
            print(
                f"now has {cnt_404} 404 words and {cnt_false} false label and {cnt_true} true label")
        test_word_matrix = merged_matrices[word]['vectors']
        test_word_indices = merged_matrices[word]['indices']
        try:
            base_dataset_word_matrix = read_vector('wiki', word, refresh=False, maxLength=maxLength)
            new_base_dataset_word_matrix, new_test_word_matrix = merge_matrix_and_reduce_dimension(
                base_dataset_word_matrix, test_word_matrix, algo=algo, dimension=dimension)

            for index_, indices_ in enumerate(test_word_indices):
                Vector = new_test_word_matrix[index_]
                label = is_in_epsilon_neighborhood(Vector, new_base_dataset_word_matrix, epsilon=eps, metric=metric)
                assert indices_ not in resDict, f"indices_ {indices_} in resDict"
                resDict[indices_] = [word, label]
                if not label:
                    cnt_false += 1
                else:
                    cnt_true += 1
        except Exception as e:
            if not isinstance(e, KeyError):
                # print(f'processing word with error {e}')
                raise e
            for indices_ in test_word_indices:
                cnt_404 += 1
                assert indices_ not in resDict, f"indices_ {indices_} in resDict"
                resDict[indices_] = [word, 404]
            continue
    res = convertResDictToResList(resDict)
    writeResult(str(res))


def calcSentenceWithDimensionDecline_with_cluster(
        baseDatabase='wiki',
        eps=18,
        metric='euclidean',
        min_samples=4,
        maxLength=20000,
        dimension=4,
        algo='default'
):
    """
    用降维进行聚类
    """
    print("starting cutting Result")
    cutResult = preprocess(args)
    # 这里cutResult存的是待标记数据集的向量化结果
    tokenizeRes = cutResult['tokenize']
    wordVector = cutResult['wordVector']
    merged_matrices = mergeVectorsByWordWithIndices(tokenizeRes, wordVector)
    reduced_matrices = reduceDimensionsForMatrices(merged_matrices, dimension=dimension, algo=algo)
    initVector(baseDatabase)
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
            }
            word_info_list.append(word_info)  # 将每个词语的信息添加到列表中
    resDict = {}
    cnt_404 = 0
    cnt_false = 0
    cnt_true = 0
    for word_instance in tqdm.tqdm(word_info_list, desc='processing cluster algorithm'):
        if (cnt_true + cnt_404 + cnt_false) % 1000 == 0:
            print(
                f"now has {cnt_404} 404 words and {cnt_false} false label and {cnt_true} true label")
        """
        try:
            wiki_cluster_result = cluster(
                        baseDatabase,
                        word_instance['word'],
                        savefig=False,
                        eps=eps,
                        metric=metric,
                        min_samples=min_samples,
                        maxLength=maxLength,
                        refresh=False,
                        dimension_d=True
                    )
        except:
            for indices_ in word_instance['indices']:
                cnt_404 += 1
                assert indices_ not in resDict, f"indices_ {indices_} in resDict"
                resDict[indices_] = [word_instance['word'], 404]
            continue

        center = getCenter(wiki_cluster_result['result class instance'])
        # if wiki_cluster_result['num_of_clusters'] == 1:
        if wiki_cluster_result['num_of_clusters']:
            center = wiki_cluster_result['cluster_members']
        # center = dimensionReduce(center)
        """
        try:
            center = dimensionReduce(
                read_vector(baseDatabase, word_instance['word'], maxLength=maxLength, refresh=False),
                dimension=dimension, algo=algo
            )
        except:
            for indices_ in word_instance['indices']:
                cnt_404 += 1
                assert indices_ not in resDict, f"indices_ {indices_} in resDict"
                resDict[indices_] = [word_instance['word'], 404]
            continue
        for index_, indices_ in enumerate(word_instance['indices']):
            Vector = word_instance['reduced_vectors'][index_]
            label = is_in_epsilon_neighborhood(Vector, center, epsilon=eps, metric=metric)
            assert indices_ not in resDict, f"indices_ {indices_} in resDict"
            resDict[indices_] = [word_instance['word'], label]
            if not label:
                cnt_false += 1
            else:
                cnt_true += 1
    print(f"there are {cnt_404} 404 words and {cnt_false} false label and {cnt_true} true label")
    res = convertResDictToResList(resDict)
    writeResult(str(res))


"""
print(Find_many_word('wiki', 'test', Count=30))
exit()
"""

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
elif args.mode == 'strengthen_wiki':
    merge_data.merge_LLM_data()
elif args.mode == 'clean_model_cache':
    clean_cache_path()
