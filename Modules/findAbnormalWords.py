try:
    from Utils.paths import clusterLog_path
except:
    from ..Utils.paths import clusterLog_path
import re
import numpy as np


def read_log():
    with open(clusterLog_path, "r", encoding='utf-8') as f:
        cont = f.read()
    return cont.split("\n")


def reFind(cont):
    datavalue = {"diff1": [], "diff2": [], "word": []}
    cnt = 0
    for i in cont:
        pattern = r"word (\w+) in dataset \w+ and \w+ with function \w+: difference is (\w+)"
        match = re.search(pattern, i)

        if match:
            word = match.group(1)
            diff = float(match.group(2))
            if cnt % 2 == 0:
                datavalue['diff1'].append(diff)
                datavalue['word'].append(word)
            else:
                datavalue['diff2'].append(diff)
            cnt += 1
        else:
            print("未找到匹配项")
            print(i)
    return datavalue


def findOutersUsingBoxPlot(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in range(len(data)) if data[x] < lower_bound or data[x] > upper_bound]
    return outliers


def find_outliers(data=None):
    """
    对于data表寻找异常值，使用的方法是箱线图的异常值排查方法
    """
    if data is None:
        cont = read_log()
        data = reFind(cont)
    BoxOut1 = findOutersUsingBoxPlot(data['diff1'])
    BoxOut2 = findOutersUsingBoxPlot(data['diff2'])
    ID = set(BoxOut1 + BoxOut2)
    wordSet = []
    for i in ID:
        wordSet.append(data['word'][i])
    wordSet = set(wordSet)
    return wordSet


if __name__ == '__main__':
    cont = read_log()
    datavalue = reFind(cont)
    print(find_outliers())
    print("end")
