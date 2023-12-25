import requests
import tqdm
import json
import pprint
import pandas as pd
from Utils.paths import *
from Utils.ToAndFromPickle import *


cant_table = pd.read_excel(cant_word_location)
cant_word_list = list(cant_table['cant'])
wordList = cant_word_list
test_pkl = load_from_pickle(test_vector)
test_pkl_word_list = list(set(test_pkl['fastIndexWord'].keys()))
wordList = set(wordList + test_pkl_word_list)


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    AppID = 44592571
    ApiKey = 'tEUIklndR5n0EtmZBCTXGIMZ'
    SK = 'lUKitI49tkevMOiMl9NUIP6kpA7osNxN'

    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={ApiKey}&client_secret={SK}"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def main(word):
    # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=' + get_access_token()
    cnt = 20
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"""
                用'{word}'这个词语造一个{cnt}句子，要求如下：
                1.你只需要输出这{cnt}个句子就行
                2.句子不能重复，越多样化越好
                3.给你的词汇不能只在开头或者结尾出现，需要在句子中间出现
                4.这{cnt}个句子中词语'{word}'出现的越多越好，最好超过100次
                5.句子中必须包含词语{word}，并且词语{word}必须原封不动的出现，否则增强无效！
                """
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "a", encoding='utf-8') as f:
        res = eval(response.text.replace("false", 'False').replace("true", 'True'))
        try:
            f.write(res['result'])
            print(f"增强词语：{word}，内容：{res['result']}")
        except Exception as e:
            print(res)
            raise e
    # print(res['result'])


def check_dataGenerate_txt(cont):
    """
    检测LLM_dataGenerate.txt中包含test.pkl中词语的数量，从而实时把控数据增强有效性
    """
    zero = 0
    for word in wordList:
        cont1 = cont
        zero += int(word not in cont1)
    print(f"TEST INTO：当前增强文本总长度：{len(cont.replace(' ', ''))}")
    print(f"TEST INTO：test中词语在增强数据集中出现次数为0的词语数量为:{zero}")


def Expand():
    res = []
    summaryDict = {}
    cnt = 0
    with open(wiki_txt_path, "r", encoding='utf-8') as f1:
        wiki = f1.read()
    for i in tqdm.tqdm(wordList, desc='生成数据集增强文本中'):
        cnt += 1
        with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "r", encoding='utf-8') as f:
            cont = f.read() + wiki
            # print(cont.count('成龙'))
            # exit(0)
            if cnt % 2000 == 0:
                check_dataGenerate_txt(cont)
            if cont.count(str(i)) >= 15:
                continue
        main(i)
        with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "r", encoding='utf-8') as f:
            cont = f.read() + wiki
            if cont.count(i) < 15:
                # raise Exception(f"词语：{i} 增强失败")
                print(f"词语：{i} 增强失败")
            else:
                print(f"词语：{i} 增强成功")
    with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "r", encoding='utf-8') as f:
        cont1 = f.read() + wiki
        for i in tqdm.tqdm(wordList, desc='测试文本结果中'):
            if cont1.count(str(i)) not in summaryDict:
                summaryDict[cont1.count(str(i))] = 0
            summaryDict[cont1.count(str(i))] += 1
            res.append(cont1.count(str(i)))
    print(f"增强文本中平均包含暗语词汇数量：{sum(res) / len(res)}\n增强文本中最少包含暗语词汇数量：{min(res)}\n增强文本中最多包含暗语词汇数量：{max(res)}")
    print("增强文本词频统计细则：")
    pprint.pprint(summaryDict)