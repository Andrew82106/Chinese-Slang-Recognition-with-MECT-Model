import os
import sys


def find_root_path():
    current_path = os.getcwd()
    while True:
        if os.path.exists(os.path.join(current_path, 'Chinese-Slang-Recognition-with-MECT-Model')):
            return os.path.join(current_path, 'Chinese-Slang-Recognition-with-MECT-Model')
        parent_path = os.path.dirname(current_path)
        if current_path == parent_path:
            return None
        current_path = parent_path


rootPth = find_root_path()
if rootPth:
    print(f"Project root path found: {rootPth}")
else:
    raise Exception("Unable to find the project root path.")

embeddings = os.path.join(rootPth, "datasets/embeddings")
charinfo = os.path.join(rootPth, "datasets/charinfo")
NER = os.path.join(rootPth, "datasets/NER")
vector = os.path.join(rootPth, "datasets/pickle_data")
cache_path = os.path.join(rootPth, "cache")
Utils_path = os.path.join(rootPth, "Utils")
wordResolving_path = os.path.join(rootPth, "wordResolving")

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
wiki_file_old_path = os.path.join(wiki_path, 'wiki_old.bmes')
# 未增强的wiki文件路径
wiki_file_path = os.path.join(wiki_path, 'wiki.bmes')
# 增强后的wiki文件路径
wiki_txt_path = os.path.join(wiki_path, 'wiki.txt')
# wiki.txt文件路径
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
LLM_data_expand_bio_file_path = os.path.join(LLM_data_expand_path, "LLM_dataGenerate.bio")
# 使用大模型进行增强的bio格式的文本

LabPath = os.path.join(Utils_path, "Lab")
LabCachePath = os.path.join(LabPath, "LabCache")
LabCachePath1 = os.path.join(LabPath, "LabCache1")

cluster_cache_path = os.path.join(cache_path, "cluster_cache")
# 缓存函数存放位置

sys.path.append(rootPth)
sys.path.append(embeddings)
sys.path.append(charinfo)
sys.path.append(NER)
sys.path.append(vector)
sys.path.append(cache_path)
sys.path.append(wordResolving_path)


def del_cluster_cache_path():
    """
    删除 cluster_cache_path 下所有文件
    """
    try:
        # 列出目录下的所有文件和文件夹
        files = os.listdir(cluster_cache_path)
        # 遍历目录下的所有文件和文件夹
        for file in files:
            # 拼接文件的完整路径
            file_path = os.path.join(cluster_cache_path, file)
            # 如果是文件，就删除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        print("All files deleted successfully!")
    except Exception as e:
        print(f"Error while deleting files: {e}")


def clean_cache_path():
    """
    删除cache_path下所有非文件夹文件
    """
    try:
        # 列出目录下的所有文件和文件夹
        files = os.listdir(cache_path)
        # 遍历目录下的所有文件和文件夹
        for file in files:
            # 拼接文件的完整路径
            file_path = os.path.join(cache_path, file)
            # 如果是文件并且不是文件夹，就删除
            if os.path.isfile(file_path) and not os.path.isdir(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        print("All non-directory files deleted successfully!")
    except Exception as e:
        print(f"Error while deleting non-directory files: {e}")


if __name__ == '__main__':
    print(charinfo)
