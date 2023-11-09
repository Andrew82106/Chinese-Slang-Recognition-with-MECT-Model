from dbScan import *


def Find_many_word(dataset1, dataset2):
    """
    在两个数据集中找到都出现过大于50个的词语
    """
    Lex_tieba = summary_lex(dataset1)
    Lex_weibo = summary_lex(dataset2)
    count = 50
    aim_word_Lst = [] # 统计一下在两个词典中出现次数都大于count的词组
    for i in tqdm.tqdm(Lex_tieba):
        if Lex_tieba[i] > count and (i in Lex_weibo and Lex_weibo[i] > count):
            aim_word_Lst.append(i)
    # print(aim_word_Lst,len(aim_word_Lst))
    return aim_word_Lst            
    

if __name__ == "__main__":
    Find_many_word("weibo", "tieba")