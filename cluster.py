from Modules.dbScan import *
import os

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
    aim_word_Lst = [] # 统计一下在两个词典中出现次数都大于count的词组
    for i in Lex_tieba:
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
            if mes:
                print(f"word {i} occurs in dataset {dataset1}:{Lex_tieba[i]} and occurs in dataset {dataset2}:{Lex_weibo[i]}")
    # print(aim_word_Lst,len(aim_word_Lst))
    return aim_word_Lst


def compare(word, datasetLst):
    file = "clusterLog/"
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
                print(log)
                with open(os.path.join(file, "clusterLog.txt"), "a", encoding='utf-8') as f:
                    f.write(str(log) + "\n")


if __name__ == "__main__":
    d = Find_many_word("PKU", "tieba", 1)
    print(d)
    compare("身体", ["PKU", "tieba"])
    compare("大学", ["PKU", "tieba"])
    compare("运动员", ["PKU", "tieba"])
    compare("工资", ["PKU", "tieba"])
    compare("垃圾", ["PKU", "tieba"])
    compare("美国", ["PKU", "tieba"])
    compare("中国", ["PKU", "tieba"])