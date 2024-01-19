import requests
import json
from wordResolving.convertCantBioToSentence import readBIO, readCantWord
import random
import tqdm
from wordResolving.similarity import generateSimilarity
import jieba


API_KEY = "tEUIklndR5n0EtmZBCTXGIMZ"
SECRET_KEY = "lUKitI49tkevMOiMl9NUIP6kpA7osNxN"


def segment_text(text):
    seg_list = list(jieba.cut(text, cut_all=False))
    return seg_list


def sendMessage(message):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    # url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=' + get_access_token()

    payload = json.dumps(message)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return eval(response.text.replace("false", "False").replace("true", "True"))


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def main():
    CantListMap, ReferListMap = readCantWord()
    sentenceList = readBIO()
    with open("./LLM_Res.txt", "w", encoding='utf-8') as f:
        summary = []
        for sentence in tqdm.tqdm(sentenceList):
            rawSentence = sentence[0]
            cantWordList = sentence[-1]
            if len(cantWordList) != 1:
                continue
            if cantWordList[0] not in CantListMap:
                continue
            refer = CantListMap[cantWordList[0]]
            # referWordList = [CantListMap[cantWordList[0]]]
            referWordList = []
            while len(referWordList) <= 10:
                referWordList.append(list(ReferListMap.keys())[random.randint(0, len(ReferListMap) - 1)])
            random.shuffle(referWordList)
            message = {
                "messages": [
                    {
                        "role": "user",
                        "content": """
                            请以一位语言学家的身份完成我的下述任务：
                            我会给你一个句子，句子中预留了一个空位，用下划线填充。你需要用一个词语填上这个空位。
                            你只需要输出你选中的词语即可，不用输出其他多余的东西。
                            样例1：今天_很好
                            样例输出1：天气
                            样例2：我们去_吃饭吧
                            样例输出2：餐厅
                            待填充句：{}
                            """.format(rawSentence)
                    }
                ]
            }

            res = sendMessage(message)
            summary0 = {
                "res": res,
                "rawSentence": rawSentence,
                "referWordList": referWordList,
                "refer": refer
            }
            summary.append(summary0)
            f.write(f"Summary: {json.dumps(summary, ensure_ascii=False)}\n")
    with open("./LLM_Res.txt", "w", encoding='utf-8') as f:
        f.write(f"Summary: {json.dumps(summary, ensure_ascii=False)}\n")


def main1():
    CantListMap, ReferListMap = readCantWord()
    sentenceList = readBIO()
    with open("./LLM_Res.txt", "w", encoding='utf-8') as f:
        summary = []
        for sentence in tqdm.tqdm(sentenceList):
            rawSentence = sentence[0]
            cantWordList = sentence[-1]
            if len(cantWordList) != 1:
                continue
            if cantWordList[0] not in CantListMap:
                continue
            refer = CantListMap[cantWordList[0]]
            referWordList = [CantListMap[cantWordList[0]]]
            # referWordList = []
            while len(referWordList) <= 10:
                referWordList.append(list(ReferListMap.keys())[random.randint(0, len(ReferListMap) - 1)])
            random.shuffle(referWordList)
            message = {
                "messages": [
                    {
                        "role": "user",
                        "content": """
                            请以一位语言学家的身份完成我的下述任务：
                            我会给你一个句子，句子中预留了一个空位，用下划线填充。你需要用一个词语填上这个空位。
                            我会给你一些备选词语，答案就在备选词语中。
                            你只需要输出你选中的词语即可，不用输出其他多余的东西。
                            待填充句：{}
                            备选词：{}
                            """.format(rawSentence, referWordList)
                    }
                ]
            }

            res = sendMessage(message)
            summary0 = {
                "res": res,
                "rawSentence": rawSentence,
                "referWordList": referWordList,
                "refer": refer
            }
            summary.append(summary0)
            f.write(f"Summary: {json.dumps(summary, ensure_ascii=False)}\n")
    with open("./LLM_Res.txt", "w", encoding='utf-8') as f:
        f.write(f"Summary: {json.dumps(summary, ensure_ascii=False)}\n")


def test():
    with open("./LLM_Res.txt", "r", encoding='utf-8') as f:
        data = f.read().split("Summary: ")
    data = eval(data[-1].strip().replace("false", "False").replace("true", "True"))
    suc = 0
    fai = 0
    for index, instance in enumerate(data):
        """
        if not len(instance.strip()):
            continue
        summary = eval(instance.strip().replace("false", "False").replace("true", "True"))[0]
        """
        summary = instance
        llmAnswer = summary['res']['result']
        refer = summary['refer']
        lst = segment_text(llmAnswer)
        flag = False
        for word in lst:
            word = word.strip()
            if not len(word):
                continue
            simi = generateSimilarity(refer, word)
            # print(refer, word, simi)
            if simi >= 0.7:
                suc += 1
                flag = True
                break
        if not flag:
            fai += 1
        if (fai + suc) % 10 == 0:
            print(fai, suc)
    print("准确度：" + str(suc/(suc+fai)))


if __name__ == '__main__':
    main1()
    test()
