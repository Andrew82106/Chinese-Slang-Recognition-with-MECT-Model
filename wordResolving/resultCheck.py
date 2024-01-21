import jieba.posseg as pseg
from wordResolving.similarity import generateSimilarity
import jieba


def segment_text(text):
    seg_list = list(jieba.cut(text, cut_all=False))
    return seg_list


def get_word_pos_in_sentence(sentence, target_word):
    # 进行中文分词和词性标注
    words = pseg.cut(sentence)

    # 查找目标词语的词性
    for word, pos in words:
        if word == target_word:
            return pos

    # 如果目标词语不在句子中，默认返回 None
    return None


def ConvertList2Map(Lst):
    res = {}
    for i in Lst:
        if i not in res:
            res[i] = 0
        res[i] += 1
    return res


def test():
    with open("./LLM_Res_3mention_word_EBOT4.txt", "r", encoding='utf-8') as f:
        data = f.read().split("Summary: ")
    data = eval(data[-1].strip().replace("false", "False").replace("true", "True"))
    suc = 0
    fai = 0
    sucLst = []
    failst = []
    for index, instance in enumerate(data):
        summary = instance
        sentence = instance['rawSentence']
        llmAnswer = summary['res']['result']
        referWordList = instance['referWordList']
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
                word_label = get_word_pos_in_sentence(sentence.replace("_", refer), refer)
                sucLst.append(word_label)
                break
        if not flag:
            word_label = get_word_pos_in_sentence(sentence.replace("_", refer), refer)
            if word_label == 'n':
                print(sentence, refer, referWordList, llmAnswer)
            failst.append(word_label)
            fai += 1
    print("准确度：" + str(suc/(suc+fai)))
    print(ConvertList2Map(sucLst))
    print(ConvertList2Map(failst))


if __name__ == '__main__':
    test()