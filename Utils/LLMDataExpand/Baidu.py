import requests
import tqdm
import json
import pprint
import pandas as pd
from Utils.paths import *

cant_table = pd.read_excel(cant_word_location)
cant_word_list = list(cant_table['cant'])
wordList = cant_word_list


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
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"""
                用'{word}'这个词语造一个25句子，要求如下：
                1.你只需要输出这25个句子就行
                2.句子不能重复，越多样化越好
                3.给你的词汇不能只在开头或者结尾出现，需要在句子中间出现
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
        except Exception as e:
            print(res)
            raise e
    # print(res['result'])


def Expand():
    res = []
    summaryDict = {}
    cnt = 0
    for i in tqdm.tqdm(wordList, desc='生成数据集增强文本中'):
        cnt += 1
        with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "r", encoding='utf-8') as f:
            cont = f.read()
            if cnt % 20 == 0:
                print(f"当前增强文本总长度：{len(cont.replace(' ', ''))}")
            if cont.count(str(i)) >= 1:
                # print(f"词语:{i} 在已有增强文本中计数为:{cont.count(str(i))}，无需增强，跳过")
                continue
        main(i)
    with open(os.path.join(LLM_data_expand_path, "LLM_dataGenerate.txt"), "r", encoding='utf-8') as f:
        cont1 = f.read()
        for i in tqdm.tqdm(wordList, desc='测试文本结果中'):
            if cont1.count(str(i)) not in summaryDict:
                summaryDict[cont1.count(str(i))] = 0
            summaryDict[cont1.count(str(i))] += 1
            res.append(cont1.count(str(i)))
    print(f"增强文本中平均包含暗语词汇数量：{sum(res) / len(res)}\n增强文本中最少包含暗语词汇数量：{min(res)}\n增强文本中最多包含暗语词汇数量：{max(res)}")
    print("增强文本词频统计细则：")
    pprint.pprint(summaryDict)