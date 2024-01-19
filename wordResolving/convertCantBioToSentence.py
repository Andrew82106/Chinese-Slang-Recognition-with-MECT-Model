from Utils.paths import *
import pandas as pd


def readBIO():
    with open(Standard_Test_BIO, "r", encoding='utf-8') as f:
        cont = f.read()
    contLst = cont.replace("。 O\n\n", "，	O\n").split("，	O")
    sentenceList = []
    for i in contLst:
        if len(i.strip()) == 0:
            continue
        sentenceList.append(i.strip())


    for index, sentence in enumerate(sentenceList):
        sentence_components = sentence.replace("\n", "\t").split("\t")
        sentence_pairs = []
        assert len(sentence_components) % 2 == 0, "ERROR SPLIT!"
        raw_sentence = ""
        cant_word_list = []
        for i in range(0, len(sentence_components), 2):
            sentence_pairs.append([sentence_components[i],  sentence_components[i+1]])
            if sentence_components[i + 1] == "O":
                raw_sentence += sentence_components[i]
            else:
                if i == 0 or sentence_components[i - 1] == "O":
                    raw_sentence += "_"
                    cant_word_list.append("")
                cant_word_list[-1] += sentence_components[i]

        sentenceList[index] = [raw_sentence, cant_word_list]
    """
    sentenceList:
    [
        [[我, O],[是, O],[是, O],[我, O]],
        [[..],[..],[..],[..]],
        [[..],[..],[..],[..]],
    ]
    """
    zero = 0
    one = 0
    up = 0
    for sentence in sentenceList:
        if len(sentence[-1]) == 0:
            zero += 1
        elif len(sentence[-1]) == 1:
            one += 1
        else:
            up += 1
    print(f"result:zero={zero} one={one} up={up}")
    return sentenceList


def readCantWord():
    df = pd.read_excel(cant_word_location)
    CantList = list(df['cant'])
    ReferList = list(df['word'])
    CantListMap = {}
    ReferListMap = {}
    for index, word in enumerate(CantList):
        CantListMap[word] = ReferList[index]
    for index, word in enumerate(ReferList):
        ReferListMap[word] = CantList[index]
    return CantListMap, ReferListMap


if __name__ == '__main__':
    readBIO()
    readCantWord()