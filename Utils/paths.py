import os
import sys

if sys.platform == 'darwin':
    rootPth = "/Users/andrewlee/Desktop/Projects/Chinese-Slang-Recognition-with-MECT-Model"
elif sys.platform == 'linux':
    rootPth = "/home/ubuntu/Project/Chinese-Slang-Recognition-with-MECT-Model"
else:
    rootPth = "B:\Chinese-Slang-Recognition-with-MECT-Model"

embeddings = os.path.join(rootPth, "datasets/embeddings")
charinfo = os.path.join(rootPth, "datasets/charinfo")
NER = os.path.join(rootPth, "datasets/NER")
vector = os.path.join(rootPth, "datasets/pickle_data")
cache_path = os.path.join(rootPth, "cache")
Utils_path = os.path.join(rootPth, "Utils")

yangjie_rich_pretrain_unigram_path = os.path.join(embeddings, 'gigaword_chn.all.a2b.uni.ite50.vec')
yangjie_rich_pretrain_bigram_path = os.path.join(embeddings, 'gigaword_chn.all.a2b.bi.ite50.vec')
yangjie_rich_pretrain_word_path = os.path.join(embeddings, 'ctb.50d.vec')

# this path is for the output of preprocessing
yangjie_rich_pretrain_char_and_word_path = os.path.join(charinfo, 'yangjie_word_char_mix.txt')

# This is the path of the file with radicals
# radical_path = '/home/ws/data/char_info.txt'
radical_path = os.path.join(charinfo, 'chaizi-jt.txt')
radical_eng_path = os.path.join(charinfo, 'radicalEng.json')

ontonote4ner_cn_path = '/home/ws/data/OntoNote4NER'
msra_ner_cn_path = os.path.join(NER, 'MSRA_NER')
resume_ner_path = os.path.join(NER, 'resume_NER')
weibo_ner_path = os.path.join(NER, 'Weibo_NER')
demo_ner_path = os.path.join(NER, 'Demo_NER')
tieba_path = os.path.join(NER, 'tieba')
PKU_path = os.path.join(NER, 'PKU')
wiki_path = os.path.join(NER, 'wiki')
anwang_path = os.path.join(NER, 'anwang')
test_path = os.path.join(NER, 'test')

tieba_vector = os.path.join(vector, 'tieba.pkl')
weibo_vector = os.path.join(vector, 'weibo.pkl')
msra_vector = os.path.join(vector, 'msra.pkl')
PKU_vector = os.path.join(vector, 'PKU.pkl')
anwang_vector = os.path.join(vector, 'anwang.pkl')
wiki_vector = os.path.join(vector, 'wiki.pkl')
test_vector = os.path.join(vector, 'test.pkl')

cluster_path = os.path.join(rootPth, "clusterRes")
# 聚类结果存放总路径
cluster_Log_Path = os.path.join(cluster_path, "clusterLog.txt")
# 聚类算法运行记录存放路径
clusterResult_path = os.path.join(cluster_path, "Result.txt")
# 初步存放聚类结果
clusterResultPhoto_path = os.path.join(cluster_path, "clusterLogs")
# 存放聚类结果图片
clusterResultBio_path = os.path.join(cluster_path, "Result.bio")
# 存放BIO格式的聚类结果
Standard_Test_BIO = os.path.join(test_path, 'input.bio')

cant_word_location = os.path.join(test_path, "cant_word.xlsx")
# 自定义词表的位置

custom_word_table = os.path.join(vector, '.pkl')

LLM_data_expand_path = os.path.join(Utils_path, "LLMDataExpand")

LabPath = os.path.join(Utils_path, "Lab")
LabCachePath = os.path.join(LabPath, "LabCache")

sys.path.append(rootPth)
sys.path.append(embeddings)
sys.path.append(charinfo)
sys.path.append(NER)
sys.path.append(vector)
sys.path.append(cache_path)
if __name__ == '__main__':
    print(charinfo)
